import datetime


def create_dates_range(current_date, period, reversed=False):
    start_date = current_date - period
    if reversed:
        return [current_date - datetime.timedelta(days=(x + 1)) for x in range((current_date - start_date).days)]
    return [start_date + datetime.timedelta(days=x) for x in range((current_date - start_date).days)]