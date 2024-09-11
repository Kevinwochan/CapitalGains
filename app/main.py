"""Trade Tracker"""

import pandas as pd
import streamlit as st
import yfinance as yf
from aus import financial_year, nearest_business_day

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
    "FY",
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
EFFECTIVE_ACTON = {
    "Buy": "Buy",
    "DRP": "Buy",
    "SPP": "Buy",
    "Sell": "Sell",
    "In": "",
    "Out": "",
}


def parse_csv(csv):
    trades = pd.read_csv(
        csv,
        converters={
            "date": pd.to_datetime,
            "units": int,
            "avg_price": float,
        },
    )
    trades["source"].fillna("Manual")
    trades["date"] = pd.to_datetime(trades["date"])
    trades["FY"] = trades["date"].apply(lambda x: financial_year(x))
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
            new_row[10] = financial_year(new_row[0])
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
    trades["FY"] = trades["Trade Date"].apply(lambda x: financial_year(x))
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
        if EFFECTIVE_ACTON[row["action"]] == "Buy":
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
    trades_df = trades_df.sort_values(by=["code", "date", "action"])
    capital_gain_events = []
    # Progressively making these pairs per year ensures the remaining buy parcels can still be used for the next year
    for code, trades_on_code in trades_df.groupby("code"):
        [buy_parcels, sell_parcels] = find_buy_sell_parcels(trades_on_code)

        # oldest sells first
        sell_parcels.sort(key=lambda x: x["date"])
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
            cgt_event.sort(key=lambda x: x["action"])
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
    cgt_events = calculate_capital_gains(trades_df)
    trading_years = sorted(trades_df["FY"].unique().tolist(), reverse=True)
    for year in trading_years:
        cgt_events_in_year = list(filter(lambda x: x["FY"] == year, cgt_events))
        if not cgt_events_in_year:
            continue
        year_gain = sum(
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
        holding_sum = (
            holdings["total_cost"].sum() if holdings["total_cost"].sum() > 0 else 1
        )
        if year_gain > 0:
            st.markdown(
                f"### {year}: :green[{year_gain:,.2f} (+{(year_gain/holding_sum*100):,.2f}%)]",
            )
        else:
            st.markdown(
                f"### {year}: :red[{year_gain:,.2f} ({(year_gain/holding_sum*100):,.2f} %)]",
            )
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

            total_cost = sum(
                [
                    x["avg_price"] * x["units"]
                    for x in cgt_event["trades"]
                    if EFFECTIVE_ACTON[x["action"]] == "Buy"
                ],
            )
            sold_value = sum(
                [
                    (x["avg_price"] * x["units"])
                    for x in cgt_event["trades"]
                    if EFFECTIVE_ACTON[x["action"]] == "Sell"
                ],
            )
            capital_gain = sold_value - total_cost
            if total_cost == 0:
                total_cost = 1
            if capital_gain > 0:
                st.markdown(
                    f":green[+{capital_gain:,.2f} ({capital_gain/total_cost*100:.2f}%) ]",
                )
            else:
                st.markdown(
                    f":red[{capital_gain:,.2f} ({capital_gain/total_cost*100:.2f}%)]",
                )


def calculate_new_holdings(
    trade,
    previous_holdings=None,
):
    """Takes a holdings summary and applies the trade deltas to calculate a summary of the new position"""
    if previous_holdings is None:
        previous_holdings = pd.DataFrame(
            columns=["code", "units", "total_cost", "avg_price"],
        )

    code = trade["code"]
    units = (
        0
        if previous_holdings.loc[
            previous_holdings["code"] == code,
            "units",
        ].empty
        else previous_holdings.loc[
            previous_holdings["code"] == code,
            "units",
        ].sum()
    )

    total_cost = (
        0
        if previous_holdings.loc[
            previous_holdings["code"] == code,
            "total_cost",
        ].empty
        else previous_holdings.loc[
            previous_holdings["code"] == code,
            "total_cost",
        ].sum()
    )

    if EFFECTIVE_ACTON[trade["action"]] == "Buy":
        units += trade["units"]
        total_cost += trade["avg_price"] * trade["units"]
    elif trade["action"] == "Sell":
        total_cost -= trade["avg_price"] * trade["units"]
        units -= trade["units"]

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
        new_holdings = pd.concat(
            [previous_holdings[previous_holdings["code"] != code], holding_summmary],
            ignore_index=True,
        )
    else:
        new_holdings = previous_holdings[previous_holdings["code"] != code]

    return new_holdings


def find_current_holdings(trades):
    current_holdings = pd.DataFrame(
        columns=["code", "units", "total_cost", "avg_price"],
    )
    for code, sub_df in trades.groupby("code"):
        sorted_trades = sub_df.sort_values("date", ascending=True)  # FIFO
        units = 0
        total_cost = 0
        for _, row in sorted_trades.iterrows():
            if EFFECTIVE_ACTON[row["action"]] == "Buy":
                units += row["units"]
                total_cost += row["avg_price"] * row["units"]
            elif row["action"] == "Sell":
                total_cost -= row["avg_price"] * row["units"]
                units -= row["units"]
            if units == 0:
                total_cost = 0
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
        lambda x: yf.Ticker(x["code"]).info.get("previousClose", 0),
        axis=1,
    )
    holdings["current_value"] = holdings["current_price"] * holdings["units"]
    holdings["profit"] = (
        holdings["current_price"] * holdings["units"] - holdings["total_cost"]
    )

    return holdings


def color_gain(value):
    return "color: green" if value > 0 else "color: red"


def display_current_holdings(corrected_trades):
    """To find the current holdings, buy/sell parcels are matched with the FIFO method"""
    current_holdings = pd.DataFrame(
        columns=["code", "units", "total_cost", "avg_price"],
    )
    current_holdings = find_current_holdings(corrected_trades)
    current_holdings = get_current_value(current_holdings)

    if current_holdings.empty:
        st.info("No current holdings found")
        return
    st.subheader(
        f"Market value: ${current_holdings['current_value'].sum():,.2f}",
    )
    st.subheader(f"Cost: ${current_holdings['total_cost'].sum():,.2f}")

    profit = (
        current_holdings["current_value"].sum() - current_holdings["total_cost"].sum()
    )
    if profit > 0:
        st.markdown(
            f"### Unrealised profit: {profit:,.2f} (:green[+{(profit/current_holdings["current_value"].sum()*100):,.2f}%])",
        )
    else:
        st.markdown(
            f"### Unrealised profit: {profit:,.2f} (:red[({(profit/current_holdings["current_value"].sum()*100):,.2f} %])",
        )
    current_holdings["weight_pct"] = (
        current_holdings["current_value"] / current_holdings["current_value"].sum()
    ) * 100
    current_holdings["profit"] = current_holdings["profit"].map("{:.2f}".format)

    # profit is $1,123
    styled_current_holdings = current_holdings.style.map(
        lambda x: "color:red" if float(x) < 0 else "color:green",
        subset="profit",
    )
    st.dataframe(
        styled_current_holdings,
        use_container_width=True,
        column_config=COLUMUMN_CONFIG,
    )


def has_sold_more_units_than_bought(sub_df):
    """Check if there are more units sold than bought"""
    total_buys = sum(
        p["units"]
        for idx, p in sub_df.iterrows()
        if EFFECTIVE_ACTON[p["action"]] == "Buy"
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
    trades = trades[trades["ignore"] == False]
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
                column_order=[
                    "date",
                    "code",
                    "action",
                    "units",
                    "avg_price",
                    "reference",
                    "source",
                ],
                hide_index=True,
                column_config=COLUMUMN_CONFIG,
            )


def display_editable_trade_table(trades):
    codes = trades["code"].unique().tolist()
    selected_codes = st.multiselect("Select holdings to include", codes, codes)
    trades = trades[trades["code"].isin(selected_codes)]
    trades = st.data_editor(
        to_editable_trades(trades),
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config=COLUMUMN_CONFIG,
        column_order=[
            "code",
            "action",
            "units",
            "avg_price",
            "brokerage",
            "date",
            "source",
            "ignore",
        ],
    )
    return trades


def to_editable_trades(trades):
    """Convert trades to editable format"""
    trades["ignore"] = False
    trades = trades.sort_values(by=["code", "date", "avg_price"], ascending=True)
    trades = trades.reset_index(drop=True)
    return trades


def display_import_options():
    """Display import options"""
    trades = pd.DataFrame(columns=NORMALISED_COLUMNS)
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

    st.subheader("Commsec")
    cba_report = st.file_uploader(
        "Portfolio -> Accounts -> Transactions -> Set your From and To dates -> Downloads -> CSV",
        type=["csv"],
        key="cba",
    )
    if cba_report:
        movement = parse_commsec_csv(cba_report)
        trades = pd.concat([trades, movement], ignore_index=True)
    st.dataframe(
        trades,
        use_container_width=True,
        column_config=COLUMUMN_CONFIG,
        column_order=[
            "date",
            "action",
            "code",
            "units",
            "avg_price",
            "brokerage",
            "source",
        ],
    )
    return trades


def daterange(start_date, end_date):
    days = int((end_date - start_date).days)
    for n in range(days):
        yield start_date + pd.Timedelta(n)


def get_market_price(historical_market_data, day, code):
    """Required when only one ticker is provided to yfinance.download"""
    indexes = historical_market_data.loc[day].index
    if code in indexes:
        return historical_market_data.loc[day][code]["Close"]
    return historical_market_data.loc[day]["Close"]


def display_historical_portfolio(corrected_trades):
    if corrected_trades.empty:
        st.info("Please upload a transactions to view your historical portfolio")
        return

    report_type = st.selectbox("Select report type", ["By Financial Year", "By Date"])
    codes = corrected_trades["code"].unique()
    historical_market_data = pd.DataFrame()
    try:
        historical_market_data = yf.download(
            " ".join(codes),
            start=corrected_trades["date"].min(),
            end=pd.Timestamp.now(),
            group_by="ticker",
        )
        historical_market_data.fillna(method="bfill", inplace=True)
    except Exception as e:
        st.error(e)
        historical_market_data = pd.DataFrame()

    if report_type == "By Financial Year":
        holdings_by_year = pd.DataFrame(columns=["FY", "cost"])
        starting_year = corrected_trades["FY"].min()
        if pd.isna(starting_year):
            starting_year = financial_year(pd.Timestamp.now()) - 1
        for year in range(
            starting_year,
            financial_year(pd.Timestamp.now()) + 1,
        ):
            trades_before_year = corrected_trades[corrected_trades["FY"] < year]
            holdings = find_current_holdings(trades_before_year)
            market_value = 0
            cost = 0
            for _, holding in holdings.iterrows():
                day = nearest_business_day(pd.Timestamp(year=year, month=6, day=30))
                cost += holding["units"] * holding["avg_price"]
                market_unit_price = get_market_price(
                    historical_market_data,
                    day,
                    holding["code"],
                )
                if not pd.isna(market_unit_price):
                    market_value += holding["units"] * market_unit_price
                else:
                    market_value += cost
            holdings_by_year = pd.concat(
                [
                    holdings_by_year,
                    pd.DataFrame.from_records(
                        [
                            {
                                "FY": str(year),
                                "cost": float(
                                    "{:.2f}".format(sum(holdings["total_cost"])),
                                ),
                                "value": market_value,
                            },
                        ],
                    ),
                ],
            )
        st.area_chart(
            holdings_by_year,
            x="FY",
            y=["value", "cost"],
            stack=False,
            use_container_width=True,
        )
    else:
        dates = corrected_trades["date"].unique()
        new_index = pd.date_range(dates.min(), dates.max(), freq="B")

        holdings_by_date = pd.DataFrame(
            columns=["date", "cost"],
        )

        holdings = pd.DataFrame(columns=["code", "units", "total_cost", "avg_price"])
        corrected_trades.sort_values("date", inplace=True)
        # for every day traded
        for date, trade_window in corrected_trades.groupby("date"):
            # find the resulting holdings at the end of the day
            for _, trade in trade_window.iterrows():
                holdings = calculate_new_holdings(
                    trade,
                    previous_holdings=holdings,
                )
            market_value = 0
            cost = 0
            for _, holding in holdings.iterrows():
                if holding["units"] == 0:
                    continue
                cost += holding["units"] * holding["avg_price"]
                market_unit_price = historical_market_data.loc[date, "Close"][
                    holding["code"]
                ]
                if not pd.isna(market_unit_price):
                    market_value += holding["units"] * market_unit_price
                else:
                    market_value += cost

            holdings_by_date = pd.concat(
                [
                    holdings_by_date,
                    pd.DataFrame.from_records(
                        [
                            {
                                "date": date,
                                "cost": cost,
                                "value": market_value,
                            },
                        ],
                    ),
                ],
            )
        holdings_by_date = holdings_by_date.set_index("date", drop=True)
        holdings_by_date.reindex(new_index, method="bfill")
        holdings_by_date["cost"] = holdings_by_date["cost"].apply(
            lambda x: float(
                f"{x:.2f}",
            ),
        )
        st.area_chart(holdings_by_date, stack=False)


st.title("Trade Tracker")


# file upload CSV
st.header("1. Upload reports")
trades = display_import_options()
st.markdown("---")


st.header("2. Consolidate, filter and correct data")
st.write(
    """
    Here you might want to:
     - add data that might be missing
     - update DRP distributions with a Buy price
     - filter out closed positions
     - remove transfers between accounts
     - update buy prices that took place before a share split
    """,
)
st.image(
    "images/download.png",
    caption="You can download the consolidated sheet using the download button",
    width=300,
)
# show an editable table
corrected_trades = display_editable_trade_table(trades)
# Highlight missing data
display_invalid_holdings(corrected_trades)
st.markdown("---")


st.header("3. Reports")
st.header("Historical Portfolio")
st.write(
    "Historical value from yahoo finance",
)

display_historical_portfolio(corrected_trades)

st.header("Portfolio Summary")
display_current_holdings(corrected_trades)
st.header("FY Capital Gains")
display_capital_gains(corrected_trades)
