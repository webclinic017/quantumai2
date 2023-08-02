import pdblp
import pandas as pd
from datetime import datetime
from pandas.tseries.offsets import BDay
from pandas.tseries.offsets import DateOffset


# Create a BCon object
# con = pdblp.BCon(debug=False, port=8194, timeout=5000)


def bbg_download_data_asset(tickers, fields, start_date, end_date, con):
    # Convert dates to 'YYYYMMDD' format
    start_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d')

    # Start the connection
    # con.start()

    data_df = pd.DataFrame()
    num_failures = 0

    for ticker in tickers:
        try:
            temp_df = con.bdh(ticker, fields, start_date, end_date)
            temp_df = temp_df.stack(level=0)
            temp_df['type'] = 'asset'
            if len(temp_df) > 0:
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                num_failures += 1
        except Exception as e:
            print(f"Error fetching data for ticker {ticker}: {e}")
    # Stop the connection
    # con.stop()

    if num_failures == len(tickers):
        raise ValueError("No data fetched")
    #! next c0olum wohl aktivieren zum factorizzen
    # data_df.index = data_df.date.factorize()[0]
    data_df = data_df.reset_index()
    field_names = ['date', 'tic'] + data_df.columns[2:].tolist()
    data_df.columns = field_names

    return data_df


def bbg_download_data_eco_indicators(tickers, fields, start_date, end_date, con):
    # Convert dates to 'YYYYMMDD' format
    start_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d')

    # Start the connection
    # con.start()

    data_dict = {}
    num_failures = 0

    for ticker in tickers:
        try:
            temp_df = con.bdh(ticker, fields, start_date, end_date)
            # Convert DataFrame to Series
            data_dict[ticker] = temp_df.iloc[:, 0]
        except Exception as e:
            print(f"Error fetching data for ticker {ticker}: {e}")
            num_failures += 1
    # Stop the connection
    # con.stop()

    if num_failures == len(tickers):
        raise ValueError("No data fetched")

    # Combine all Series into a DataFrame
    data_df = pd.DataFrame(data_dict)
    data_df = data_df.reset_index()

    return data_df


def bbg_data_intraday(tickers, start_date, end_date, interval):
    con = pdblp.BCon(debug=False, port=8194, timeout=5000)
    con.start()
    # Generate list of business days between start_date and end_date
    business_days = pd.date_range(start_date, end_date, freq=BDay())

    data_df = pd.DataFrame()

    for ticker in tickers:
        for start_day in business_days:
            # Shift start_day by one business day to get end_day
            end_day = start_day + DateOffset(days=1)

            # Convert start_day and end_day to strings in the correct format
            start = str(start_day.strftime('%Y-%m-%dT09:30:00'))
            end = str(end_day.strftime('%Y-%m-%dT15:30:00'))

            try:
                temp_df = con.bdib(ticker, start, end,
                                   'TRADE', interval=interval)
                temp_df["tic"] = ticker
                if len(temp_df) > 0:
                    data_df = pd.concat([data_df, temp_df], axis=0)
            except Exception as e:
                print(
                    f"Error fetching data for ticker {ticker} on {start} to {end}: {e}")

    return data_df


def bbg_data_intraday_simple(tickers):
    con = pdblp.BCon(debug=False, port=8194, timeout=5000)
    con.start()
    # Generate list of business days between start_date and end_date
    # business_days = pd.date_range(start_date, end_date, freq=BDay())

    # Generate list of start and end dates for each two-day period
    # two_day_periods = [(business_days[i], business_days[i+1])
    #                    for i in range(len(business_days)-1)]

    data_df = pd.DataFrame()

    for ticker in tickers:
        # for start_day, end_day in two_day_periods:
        #     start_day_str = start_day.strftime('%Y-%m-%d')
        #     end_day_str = end_day.strftime('%Y-%m-%d')
        start = '2023-06-19T09:30:00'
        end = '2023-07-19T15:30:00'
        try:
            temp_df = con.bdib(ticker, start, end,
                               'TRADE', interval=60)
            temp_df["tic"] = ticker
            if len(temp_df) > 0:
                data_df = pd.concat([data_df, temp_df], axis=0)
        except Exception as e:
            print(
                f"Error fetching data for ticker {ticker} on {start} to {end}: {e}")
    con.stop()

    return data_df
