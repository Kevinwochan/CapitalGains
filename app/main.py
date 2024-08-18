"""Trade Tracker"""

import pandas as pd
import streamlit as st
import yfinance as yf
from aus import financial_year

st.set_page_config(
    page_title="Trade Tracker",
    page_icon=":bar_chart:",
    layout="wide",
)

NORMALISED_COLUMNS = [
    "date",
    "action",
    "reference",
    "code",
    "units",
    "avg_price",
    "consideration",
    "brokerage",
    "source",
    "currency",
]

COLUMUMN_CONFIG = {
    "date": st.column_config.DateColumn(
        "Date",
    ),
    "weight_pct": st.column_config.NumberColumn(
        "Weight (%)",
        format="%.1f",
    ),
}

ACTIONS = ["Buy", "Sell", "DRP", "SPP"]


def parse_csv(csv):
    trades = pd.read_csv(
        csv,
        usecols=NORMALISED_COLUMNS,
        converters={
            "date": pd.to_datetime,
            "units": int,
            "avg_price": float,
        },
    )
    trades["source"].fillna("Manual")
    trades["date"] = pd.to_datetime(trades["date"])
    return trades


def parse_commsec_csv(csv, currency="AUD"):
    """Date	Reference	Details	Debit($)	Credit($)	Balance($)."""
    transactions = pd.read_csv(
        csv,
        usecols=["Date", "Reference", "Details", "Debit($)", "Credit($)", "Balance($)"],
    )
    # 'C' means this is a contract transaction, only use those
    trades = transactions[transactions["Reference"].str.contains("C")]

    def parse_commsec_row(row):
        """Translate to normalised row."""
        commsec_actions = {
            "B": "Buy",
            "S": "Sell",
        }
        new_row = [col for col in NORMALISED_COLUMNS]
        try:
            action, units, code, _, avg_price = row["Details"].split(" ")
            consideration = row["Debit($)"] if action == "B" else row["Credit($)"]
            new_row[0] = pd.to_datetime(row["Date"], format="%d/%m/%Y")
            new_row[1] = commsec_actions[action]
            new_row[2] = row["Reference"]
            new_row[3] = code + ".AX"
            new_row[4] = int(units)
            new_row[5] = float(avg_price)
            new_row[6] = float(consideration)
            new_row[7] = abs(float(consideration) - (float(units) * float(avg_price)))
            new_row[8] = "Commsec CSV"
            new_row[9] = currency
        except:
            st.error(
                "Please upload a valid Commsec CSV file from Portfolio -> Accounts -> Transactions -> Set your From and To dates -> Downloads -> CSV",
            )
        return new_row

    new_trades = pd.DataFrame.from_records(
        [parse_commsec_row(row) for _, row in trades.iterrows()],
        columns=NORMALISED_COLUMNS,
    )
    return new_trades


def parse_selfwealth_csv(csv, currency="AUD"):
    """Trade Date	Settlement Date	Action	Reference	Code	Name	Units	Average Price	Consideration	Brokerage	Total
    16/08/2024 0:00	20/08/2024 0:00	Buy	1993950	GOLD	GBLX GOLD	141	34.25	4829.25	9.5	4838.75

    """
    trades = pd.read_csv(
        csv,
        usecols=[
            "Trade Date",
            "Action",
            "Reference",
            "Code",
            "Units",
            "Average Price",
            "Consideration",
            "Brokerage",
        ],
        converters={
            "Trade Date": lambda x: pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S"),
            "Units": int,
            "Code": lambda x: x + ".AX" if currency == "AUD" else x,
        },
    )
    trades["source"] = "Selfwealth CSV"
    trades["currency"] = currency
    trades.columns = NORMALISED_COLUMNS
    # translate this: 2021-05-28 00:00:00 to a Datetime object
    return trades


# match trades with buy parcels
CAPITAL_GAIN_METHODS = [
    "FIFO",
    "LIFO",
    "Minimise Capital Gain Tax",
]


def find_buy_sell_parcels(trades):
    """Find buy and sell parcels."""
    buy_parcels = []
    sell_parcels = []
    for _, row in trades.iterrows():
        if row["action"] == "Buy" or row["action"] == "DRP" or row["action"] == "SPP":
            buy_parcels.append(row)
        elif row["action"] == "Sell":
            sell_parcels.append(row)

    return buy_parcels, sell_parcels


def apply_cgt_discount(buy_parcels, date):
    discounted_buy_parcels = []
    for buy in buy_parcels:
        buy["contract_price"] = buy["avg_price"]
        if buy["date"] - date > pd.Timedelta(
            days=365,
        ):
            buy["avg_price"] = buy["avg_price"] / 2
        discounted_buy_parcels.append(buy)

    discounted_buy_parcels.sort(
        key=lambda x: x["avg_price"],
        reverse=True,
    )
    return discounted_buy_parcels


def calculate_capital_gains(trades_df):
    """Calculate capital gains by taking the highest possible buy parcel for each trade and matching it with the highest possible sell parcel."""
    if trades_df.empty:
        return trades_df

    capital_gain_events = []
    # Progressively making these pairs per year ensures the remaining buy parcels can still be used for the next year
    for code, trades_on_code in trades_df.groupby("code"):
        [buy_parcels, sell_parcels] = find_buy_sell_parcels(trades_on_code)

        # most recent sells first
        sell_parcels.sort(key=lambda x: x["date"], reverse=True)
        # expensive buys first
        buy_parcels.sort(key=lambda x: x["avg_price"], reverse=True)

        for sell_parcel in sell_parcels:
            units_remaining = sell_parcel["units"]
            # filter out invalid buy parcels and apply the cgt discount
            discounted_buy_parcels = apply_cgt_discount(
                list(
                    filter(
                        lambda x: x["date"] <= sell_parcel["date"] and x["units"] > 0,
                        buy_parcels,
                    ),
                ),
                sell_parcel["date"],
            )

            cgt_event = [sell_parcel]
            for buy_parcel in discounted_buy_parcels:
                units_sold = min(units_remaining, buy_parcel["units"])
                units_remaining -= units_sold
                buy = buy_parcel.copy(deep=True)
                buy["units"] = units_sold
                cgt_event.append(buy)
                buy_parcel["units"] -= units_sold

                if units_remaining == 0:
                    break
            cgt_event.sort(key=lambda x: x["date"])
            capital_gain_events.append(
                {
                    "code": code,
                    "trades": cgt_event,
                    "FY": sell_parcel["FY"],
                },
            )

    return capital_gain_events


def display_capital_gains(
    trades_df,
    selected_codes=[],
):
    trades_df["FY"] = trades_df["date"].apply(lambda x: financial_year(x))
    cgt_events = calculate_capital_gains(trades_df)
    trading_years = sorted(trades_df["FY"].unique().tolist())
    for year in trading_years:
        cgt_events_in_year = list(filter(lambda x: x["FY"] == year, cgt_events))
        if not cgt_events_in_year:
            continue

        gain = sum(
            [
                sum(
                    [
                        -(x["avg_price"] * x["units"])
                        if x["action"] == "Buy"
                        else (x["avg_price"] * x["units"])
                        for x in cgt_event["trades"]
                    ],
                )
                for cgt_event in cgt_events_in_year
            ],
        )
        holdings = find_current_holdings(trades_df[trades_df["FY"] < year])
        holding_sum = holdings["total_cost"].sum()
        st.subheader(f"{year}: {gain:,.2f} ({(gain/holding_sum*100):,.2f} %)")

        for cgt_event in cgt_events_in_year:
            st.dataframe(
                cgt_event["trades"],
                column_config={
                    "date": st.column_config.DateColumn(
                        "Date",
                    ),
                },
                column_order=[
                    "code",
                    "action",
                    "date",
                    "units",
                    "avg_price",
                    "consideration",
                    "reference",
                ],
                use_container_width=True,
                hide_index=True,
            )

            capital_gain = sum(
                [
                    -(x["avg_price"] * x["units"])
                    if x["action"] == "Buy"
                    else (x["avg_price"] * x["units"])
                    for x in cgt_event["trades"]
                ],
            )
            st.write(f"Capital Gain: ${capital_gain:,.2f}")


def find_current_holdings(trades):
    current_holdings = pd.DataFrame()
    for code, sub_df in trades.groupby("code"):
        sorted_trades = sub_df.sort_values("date", ascending=True)  # FIFO
        units = 0
        total_cost = 0
        for _, row in sorted_trades.iterrows():
            if (
                row["action"] == "Buy"
                or row["action"] == "DRP"
                or row["action"] == "SPP"
            ):
                units += row["units"]
                total_cost += row["avg_price"] * row["units"]
            elif row["action"] == "Sell":
                total_cost -= row["avg_price"] * row["units"]
                units -= row["units"]

        if units > 0:
            holding_summmary = pd.DataFrame.from_records(
                data=[
                    {
                        "code": code,
                        "units": units,
                        "avg_price": total_cost / units,
                        "total_cost": total_cost,
                    },
                ],
                columns=[
                    "code",
                    "units",
                    "avg_price",
                    "total_cost",
                ],
            )
            current_holdings = pd.concat(
                [current_holdings, holding_summmary],
                ignore_index=True,
            )

    return current_holdings


def get_current_value(holdings):
    if holdings.empty:
        return holdings
    holdings["current_price"] = holdings.apply(
        lambda x: yf.Ticker(x["code"]).info["previousClose"],
        axis=1,
    )
    holdings["current_value"] = holdings["current_price"] * holdings["units"]
    holdings["profit"] = (
        holdings["current_price"] * holdings["units"] - holdings["total_cost"]
    )
    return holdings


def display_current_holdings(corrected_trades):
    """To find the current holdings, buy/sell parcels are matched with the FIFO method"""
    current_holdings = pd.DataFrame(
        columns=["code", "units", "total_cost", "avg_price"],
    )
    current_holdings = find_current_holdings(corrected_trades)
    current_holdings = get_current_value(current_holdings)

    if current_holdings.empty:
        st.write("No current holdings found")
        return
    st.write(
        f"Portfolio value: ${current_holdings['current_value'].sum():,.2f}",
    )
    st.write(f"Total cost: ${current_holdings['total_cost'].sum():,.2f}")
    st.write(
        f"Total Profit: ${current_holdings['current_value'].sum() - current_holdings['total_cost'].sum():,.2f}",
    )

    current_holdings["weight_pct"] = (
        current_holdings["current_value"] / current_holdings["current_value"].sum()
    ) * 100

    st.dataframe(
        current_holdings,
        use_container_width=True,
        column_config=COLUMUMN_CONFIG,
    )


def has_sold_more_units_than_bought(sub_df):
    """Check if there are more units sold than bought"""
    total_buys = sum(
        p["units"]
        for idx, p in sub_df.iterrows()
        if p["action"] == "Buy" or p["action"] == "DRP" or p["action"] == "SPP"
    )
    total_sells = sum(
        p["units"] for idx, p in sub_df.iterrows() if p["action"] == "Sell"
    )
    return total_sells > total_buys


def has_missing_price_data(sub_df):
    """Check if there is missing price data"""
    return sub_df["avg_price"].isna().any()


def display_invalid_holdings(trades) -> None:
    """Validate trades"""
    # make sure there are enough buys to cover all sells
    for code, sub_df in trades.groupby("code"):
        error_messages = []
        if has_sold_more_units_than_bought(sub_df):
            error_messages.append(
                "More units sold than bought, you can ignore specific trades or fill in missing data",
            )
        elif has_missing_price_data(sub_df):
            error_messages.append(
                "Missing avg_price, this is required to calculate the capital gain",
            )

        if len(error_messages) > 0:
            st.subheader(code)
            for msg in error_messages:
                st.warning(msg)
            st.dataframe(
                sub_df,
                use_container_width=True,
                column_config=COLUMUMN_CONFIG,
            )


def to_editable_trades(trades):
    """Convert trades to editable format"""
    trades["ignore"] = False
    trades = trades.sort_values(by=["code", "date", "avg_price"], ascending=True)
    return trades


# tax calculations


# handle CGT events that have already been reported to ATO

st.title("Trade Tracker")

trades = pd.DataFrame(columns=NORMALISED_COLUMNS)

# file upload CSV
st.header("1. Upload reports")

st.subheader("Previous export")
prev_report = st.file_uploader(
    "From the last time you used this tool",
    type=["csv"],
    key="prev",
)
if prev_report:
    movement = parse_csv(prev_report)
    trades = pd.concat([trades, movement], ignore_index=True)

st.subheader("Selfwealth AUD")
sw_report = st.file_uploader(
    "Trading Account -> Movements -> Set your time period -> Export X Rows -> As CSV",
    type=["csv"],
    key="sw",
)
if sw_report:
    movement = parse_selfwealth_csv(sw_report, currency="AUD")
    trades = pd.concat([trades, movement], ignore_index=True)

st.subheader("Selfwealth USD")
sw_report = st.file_uploader(
    "Trading Account -> Movements -> Set your time period -> Export X Rows -> As CSV",
    type=["csv"],
    key="sw-usd",
)
if sw_report:
    movement = parse_selfwealth_csv(sw_report, currency="USD")
    trades = pd.concat([trades, movement], ignore_index=True)

st.subheader("Commsec")
cba_report = st.file_uploader(
    "Portfolio -> Accounts -> Transactions -> Set your From and To dates -> Downloads -> CSV",
    type=["csv"],
    key="cba",
)
if cba_report:
    movement = parse_commsec_csv(cba_report)
    trades = pd.concat([trades, movement], ignore_index=True)


st.session_state["original_trades"] = trades
st.dataframe(trades, use_container_width=True, column_config=COLUMUMN_CONFIG)


st.markdown("---")


st.header("2. Consolidate, filter and correct data")
st.write(
    """
    Here you might want to:
     - add data that might be missing 
     - update DRP distributions with a Buy price
     - filter out closed positions
     - remove transfererred between accounts
     - update buy prices that took place before a share split

     It may be easier for you to download the consolidated sheet and resolve the below issues in a sheets editor like excel or google sheets before uploading again for the reports
    """,
)

codes = trades["code"].unique().tolist()
selected_codes = st.multiselect("Select holdings to include", codes, codes)
trades = trades[trades["code"].isin(selected_codes)]
trades = trades.reset_index(drop=True)

trades = st.data_editor(
    to_editable_trades(trades),
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    column_config=COLUMUMN_CONFIG,
)


corrected_trades = trades[trades["ignore"] == False]
# Highlight missing data
display_invalid_holdings(corrected_trades)


st.markdown("---")


st.header("3. Reports")

st.header("Portfolio Summary")
display_current_holdings(corrected_trades)


st.header("FY Capital Gains")
display_capital_gains(
    trades,
    selected_codes=selected_codes,
)
