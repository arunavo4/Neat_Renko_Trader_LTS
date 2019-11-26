import math
import talib
import pandas as pd
import numpy as np


def get_pattern_columns():
    # initialize a random Dataframe
    random_df = pd.DataFrame(np.random.randint(0, 100, size=(10, 5)),
                             columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    random_df = add_indicators(random_df, tech_in=False, patterns=True)
    random_df = random_df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

    return random_df.columns


def awesome_osc(high, low, s=5, len=34):
    """Awesome Oscillator
    MEDIAN PRICE = (HIGH+LOW)/2
    AO = SMA(MEDIAN PRICE, 5)-SMA(MEDIAN PRICE, 34)
    """
    mp = 0.5 * (high + low)
    ao = talib.SMA(mp, s) - talib.SMA(mp, len)
    return ao


def get_hma(price, timeperiod=14):
    # HMA= WMA(2*WMA(n/2) − WMA(n)),sqrt(n))
    return (talib.WMA(
        2 * talib.WMA(price, timeperiod=math.floor(timeperiod / 2)) - talib.WMA(price, timeperiod=timeperiod),
        timeperiod=math.sqrt(timeperiod)))


def add_indicators(df, tech_in=True, patterns=True):
    # Technical Indicators
    if tech_in:
        df["ema_5_by_10"] = talib.EMA(df['Close'], 5) / talib.EMA(df['Close'], 10)

        df["ema_10_by_20"] = talib.EMA(df['Close'], 10) / talib.EMA(df['Close'], 20)

        df["sma_5_by_10"] = talib.SMA(df['Close'], 5) / talib.SMA(df['Close'], 10)

        df["sma_10_by_20"] = talib.SMA(df['Close'], 10) / talib.SMA(df['Close'], 20)

        df["hma_9_by_18"] = get_hma(df['Close'], 9) / get_hma(df['Close'], 18)

        df["bop"] = talib.BOP(df['Open'], df['High'], df['Low'], df['Close'])

        df["beta"] = talib.BETA(df['High'], df['Low'])

        df["rsi"] = talib.RSI(df['Close'])

        df["adi"] = talib.ADX(df['High'], df['Low'], df['Close'])

        df["natr"] = talib.NATR(df['High'], df['Low'], df['Close'])

        df["mom"] = talib.MOM(df['Close'])

        macd, macdsignal, macdhist = talib.MACD(df['Close'])

        df["macd"] = macd

        df["macdsignal"] = macdsignal

        df["macdhist"] = macdhist

        fastk, fastd = talib.STOCHRSI(df['Close'])

        df["fastk"] = fastk

        df["fastd"] = fastd

        df["ulti"] = talib.ULTOSC(df['High'], df['Low'], df['Close'])

        df["awesome"] = awesome_osc(df['High'], df['Low'])

        df["wills_r"] = talib.WILLR(df['High'], df['Low'], df['Close'])

    if patterns:
        # pattern recognition
        df["three_line_strike"] = talib.CDL3LINESTRIKE(df['Open'], df['High'], df['Low'], df['Close'])

        df["three_black_crows"] = talib.CDL3BLACKCROWS(df['Open'], df['High'], df['Low'], df['Close'])

        df["doji_star"] = talib.CDLDOJISTAR(df['Open'], df['High'], df['Low'], df['Close'])

        df["evening_doji_star"] = talib.CDLEVENINGDOJISTAR(df['Open'], df['High'], df['Low'], df['Close'])

        df["morning_doji_star"] = talib.CDLMORNINGDOJISTAR(df['Open'], df['High'], df['Low'], df['Close'])

        df["morning_star"] = talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])

        df["evening_star"] = talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])

        df["shooting_star"] = talib.CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])

        df["engulfing_patt"] = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])

        df["hammer"] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])

        df["inverted_hammer"] = talib.CDLINVERTEDHAMMER(df['Open'], df['High'], df['Low'], df['Close'])

        df["hanging_man"] = talib.CDLHANGINGMAN(df['Open'], df['High'], df['Low'], df['Close'])

        df["harami"] = talib.CDLHARAMI(df['Open'], df['High'], df['Low'], df['Close'])

        df["harami_cross"] = talib.CDLHARAMICROSS(df['Open'], df['High'], df['Low'], df['Close'])

        df["piercing"] = talib.CDLPIERCING(df['Open'], df['High'], df['Low'], df['Close'])

    df.fillna(method='bfill', inplace=True)

    return df
