"""Trade Tracker"""

import datetime

import pandas as pd
import streamlit as st

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
]

COLUMUMN_CONFIG = {
    "date": st.column_config.DateColumn(
        "Date",
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


def parse_commsec_csv(csv):
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
            new_row[3] = code
            new_row[4] = int(units)
            new_row[5] = float(avg_price)
            new_row[6] = float(consideration)
            new_row[7] = abs(float(consideration) - (float(units) * float(avg_price)))
            new_row[8] = "Commsec CSV"
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


def parse_selfwealth_csv(csv):
    """Trade Date	Settlement Date	Action	Reference	Code	Name	Units	Average Price	Consideration	Brokerage	Total
    16/08/2024 0:00	20/08/2024 0:00	Buy	1993950	GOLD	GBLX GOLD	141	34.25	4829.25	9.5	4838.75

    """
    try:
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
            },
        )
        trades["source"] = "Selfwealth CSV"

        trades.columns = NORMALISED_COLUMNS
        # translate this: 2021-05-28 00:00:00 to a Datetime object
        return trades
    except Exception as e:
        st.error(
            "Please upload a valid Selfwealth CSV file from Trading Account -> Movements -> Set your time period -> Export X Rows -> As CSV",
        )
        print(e)
    return pd.DataFrame()


# match trades with buy parcels
def calculate_capital_gains(
    trades_df,
    selected_codes=[],
    year=int(datetime.date.today().year),
):
    """Calculate capital gains by taking the highest possible buy parcel for each trade and matching it with the highest possible sell parcel.
    trades_df could be empty
    """
    if trades_df.empty:
        return trades_df
    # filter trades by year
    trades_df = trades_df[trades_df["date"].dt.year == year]

    remaining_holdings = {}
    for code, sub_df in trades_df.groupby("code"):
        buy_parcels = []
        sell_parcels = []
        for idx, row in sub_df.iterrows():
            if (
                row["action"] == "Buy"
                or row["action"] == "DRP"
                or row["action"] == "SPP"
            ):
                buy_parcels.append(row)
            elif row["action"] == "Sell":
                sell_parcels.append(row)

        # match sell parcels with buy parcels
        capital_gain_events = []

        # most recent sells first
        sell_parcels.sort(key=lambda x: x["date"], reverse=True)
        # expensive buys first
        buy_parcels.sort(key=lambda x: x["avg_price"], reverse=True)

        for sell_parcel in sell_parcels:
            units_remaining = sell_parcel["units"]
            for buy_parcel in buy_parcels:
                # ensure the buy date is before the sell date
                if (
                    not buy_parcel["date"] <= sell_parcel["date"]
                    and buy_parcel["units"] > 0
                ):
                    continue
                # update
                units_sold = min(units_remaining, buy_parcel["units"])
                capital_gain_events.append(
                    {
                        "code": code,
                        "buy": {
                            **buy_parcel,
                            "units": units_sold,
                            "value": units_sold * buy_parcel["avg_price"],
                        },
                        "sell": {
                            **sell_parcel,
                            "value": units_sold * sell_parcel["avg_price"],
                        },
                        "capital_gain": units_sold
                        * (sell_parcel["avg_price"] - buy_parcel["avg_price"]),
                    },
                )
                units_remaining -= units_sold
                buy_parcel["units"] -= units_sold
                # all units sold
                if units_remaining < 1:
                    break

        # create output table
        if not capital_gain_events:
            continue

        st.subheader(code)
        for event in capital_gain_events:
            table = pd.DataFrame.from_records(
                data=[
                    event["buy"],
                    event["sell"],
                    {"capital_gain": event["capital_gain"]},
                ],
                columns=[
                    "date",
                    "action",
                    "reference",
                    "units",
                    "avg_price",
                    "brokerage",
                    "value",
                    "capital_gain",
                ],
            )
        remaining_buy_parcels = [p for p in buy_parcels if p["units"] > 0]
        if remaining_buy_parcels:
            remaining_holdings[code] = remaining_buy_parcels
    return remaining_holdings


def display_current_holdings(corrected_trades):
    current_holdings = pd.DataFrame(
        columns=["code", "units", "total_cost", "avg_price"],
    )
    for code, sub_df in corrected_trades.groupby("code"):
        units = 0
        total_cost = 0
        for _, row in sub_df.iterrows():
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
                columns=["code", "units", "avg_price", "total_cost"],
            )
            current_holdings = pd.concat(
                [current_holdings, holding_summmary],
                ignore_index=True,
            )
    st.dataframe(current_holdings)


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

st.subheader("Selfwealth")
sw_report = st.file_uploader(
    "Trading Account -> Movements -> Set your time period -> Export X Rows -> As CSV",
    type=["csv"],
    key="sw",
)
if sw_report:
    movement = parse_selfwealth_csv(sw_report)
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
# select years
years = [] if trades.empty else trades["date"].dt.year.unique().tolist()

selected_year = st.selectbox(
    "Select a year to filter by",
    sorted(years),
    index=len(years) - 1,
    key="year",
)

remaining_holdings = calculate_capital_gains(
    trades,
    selected_codes=selected_codes,
    year=selected_year,
)

calculation_method = st.selectbox(
    "Select gains calculation method",
    ["LIFO", "FIFO", "minimise CGT"],
    index=0,
    key="calculation_method",
)
