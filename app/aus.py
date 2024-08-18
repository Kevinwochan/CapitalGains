def financial_year(date):
    """Return the Australian financial year from a datetime object"""
    if date.month < 7:
        return date.year - 1
    return date.year
