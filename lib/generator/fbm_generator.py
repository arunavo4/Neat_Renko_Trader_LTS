import pandas as pd
import numpy as np

from stochastic.continuous import FractionalBrownianMotion
from stochastic.noise import GaussianNoise


class FBMExchange:
    """A simulated instrument exchange, in which the price history is based off a fractional brownian motion
    model with supplied parameters.
    """

    def __init__(self, **kwargs):
        # super().__init__(data_frame=None, **kwargs)

        self._base_price = kwargs.get('base_price', np.random.randint(500, 1000))
        self._base_volume = kwargs.get('base_volume', 1)
        self._start_date = kwargs.get('start_date', '2010-01-01')
        self._start_date_format = kwargs.get('start_date_format', '%Y-%m-%d')
        self._times_to_generate = kwargs.get('times_to_generate', 111056)
        self._hurst = kwargs.get('hurst', round(np.random.uniform(0.3, 0.45), 2))
        self._timeframe = kwargs.get('timeframe', '1min')

    def _generate_price_history(self):
        try:
            price_fbm = FractionalBrownianMotion(t=self._times_to_generate, hurst=self._hurst)
            volume_gen = GaussianNoise(t=self._times_to_generate)
        except:
            self._generate_price_history()

        start_date = pd.to_datetime(self._start_date, format=self._start_date_format)

        price_volatility = price_fbm.sample(self._times_to_generate, zero=False)
        prices = price_volatility + self._base_price
        volume_volatility = volume_gen.sample(self._times_to_generate)
        volumes = volume_volatility * price_volatility + self._base_volume

        price_frame = pd.DataFrame([], columns=['date', 'price'], dtype=float)
        volume_frame = pd.DataFrame(
            [], columns=['date', 'volume'], dtype=float)

        price_frame['date'] = pd.date_range(
            start=start_date, periods=self._times_to_generate, freq="1min")
        price_frame['price'] = abs(prices)

        volume_frame['date'] = price_frame['date'].copy()
        volume_frame['volume'] = abs(volumes)

        price_frame.set_index('date')
        price_frame.index = pd.to_datetime(price_frame.index, unit='m', origin=start_date)

        volume_frame.set_index('date')
        volume_frame.index = pd.to_datetime(volume_frame.index, unit='m', origin=start_date)

        data_frame = price_frame['price'].resample(self._timeframe).ohlc()
        data_frame['volume'] = volume_frame['volume'].resample(self._timeframe).sum()

        self.data_frame = data_frame.astype(np.float16)

    def reset(self):
        self._generate_price_history()
