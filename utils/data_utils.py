import pandas as pd


def load_csv(csv_file):
    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %I:%M:%S %p')
    df['Date'] = df['Date'].astype(str)
    df = df.sort_values(['Date'])

    return df


def round_up(num_to_round, multiple):
    if multiple == 0:
        return num_to_round

    remainder = abs(num_to_round) % multiple
    if remainder == 0:
        return num_to_round

    if num_to_round < 0:
        return -(abs(num_to_round) - remainder)
    else:
        return num_to_round + multiple - remainder


def split_data(data, train_per=0.8):
    train_size = round_up(int(round(train_per * data.shape[0])), 375)
    train = data[:train_size]
    test = data[train_size:]
    return train,test
