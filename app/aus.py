import pandas as pd
def financial_year(date):
    """Return the Australian financial year from a datetime object"""
    if date.month < 7:
        return date.year - 1
    return date.year

def nearest_business_day(date):
    """Return the nearest business day from a datetime object"""
    if date.weekday() < 5:
        return date
    return date - pd.Timedelta(days=date.weekday() - 4)