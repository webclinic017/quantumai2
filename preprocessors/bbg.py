import pdblp
import pandas as pd
from datetime import datetime


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
