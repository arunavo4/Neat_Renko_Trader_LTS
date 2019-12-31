"""
    ****** Universal Version of the Stock Trader ******
    This version of the Env has specific Enhancements for Intra-day Trading
    *** This is the most Optimized version till date ***
"""

# logging
import logging
import math
from collections import deque
from statistics import mean

import cv2
import gym
import numpy as np
import pandas as pd
import talib
from gym import spaces

from lib.generator.static_generator import StaticExchange
# from lib.renko.Renko import get_optimal_box_size


def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""
    formatter = logging.Formatter('%(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


# ***** Zerodha Brokerage *****
# Func to calculate brokerage
def cal_profit_w_brokerage(buy_price, sell_price, qty):
    turnover = (buy_price * qty) + (sell_price * qty)
    brokerage = ((0.01 / 100) * turnover)
    stt = math.ceil((0.025 / 100) * (sell_price * qty))
    exn_trn = ((0.00325 / 100) * turnover)
    gst = (18 / 100) * (brokerage + exn_trn)
    sebi = (0.000001 * turnover)

    return ((sell_price - buy_price) * qty) - (brokerage + stt + exn_trn + gst + sebi)


# Delete this if debugging
np.warnings.filterwarnings('ignore')


class StockTraderEnv(gym.Env):
    """A Stock trading environment for Stock Market"""
    metadata = {'render.modes': ['human', 'system', 'none']}
    viewer = None

    def __init__(self, config):
        super(StockTraderEnv, self).__init__()

        self.initial_balance = config["initial_balance"]

        self.exchange = StaticExchange(config=config)

        self.enable_logging = config['enable_env_logging']
        if self.enable_logging:
            self.logger = setup_logger('env_logger', 'env.log')
            self.logger.info('Env Logger!')

        # Stuff from Renko
        self.source_prices = []
        self.renko_prices = []
        self.renko_directions = []
        self.brick_size = 10.0
        self.brick_size_per = 0.0

        # Leverage
        self.leverage = 1
        self.use_leverage = config['use_leverage']

        if self.use_leverage:
            self.leverage = 1   # Currently disabled

        # Stuff from the Env Before
        self.decay_rate = 1e-2
        self._is_auto_hold = False
        self.done = False
        self.day_step_size = config['day_step_size']
        self.current_step = int(self.day_step_size)
        self.wins = int(0)
        self.losses = int(0)
        self.qty = int(0)
        self.short = False
        self.tradable = True
        self.market_open = True
        self.balance = self.initial_balance
        self.qty = int(0)
        self.profit_per = float(0.0)
        self.daily_profit_per = []
        self.profits = int(0)
        self.positions = deque([], maxlen=1)
        self.position_value = int(0)
        self.action_record = ""
        self.position_record = ""
        self.rewards = deque(np.zeros(1, dtype=float))
        self.net_worth = deque([self.initial_balance], maxlen=1)
        self.stock_name = 'NAN'
        self.sum = 0.0
        self.denominator = np.exp(-1 * self.decay_rate)

        self.look_back_window_size = config['look_back_window_size'] or self.day_step_size * 10
        self.obs_window = config['observation_window'] or 32
        self.hold_reward = config['hold_reward']

        # Frame Stack
        self.stack_size = config['frame_stack_size'] or 1
        self.frames = deque([], maxlen=self.stack_size)

        # Actions of the format Buy, Sell , Hold .
        self.action_space = spaces.Discrete(3)

        # Observes the price action, indicators, account action, price forecasts
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.obs_window, self.obs_window, self.stack_size),
                                            dtype=np.uint8)

    # Setting brick size. Auto mode is preferred, it uses history
    def set_brick_size(self, hlc_history=None, auto=True, brick_size=10.0):
        if auto:
            self.brick_size = self.__get_optimal_brick_size(hlc_history)
        else:
            self.brick_size = brick_size

    def __renko_rule(self, last_price):
        # Get the gap between two prices
        gap_div = int(float(float(last_price) - float(self.renko_prices[-1])) / self.brick_size)
        is_new_brick = False
        start_brick = 0
        num_new_bars = 0

        # When we have some gap in prices
        if gap_div != 0:
            # Forward any direction (up or down)
            if (gap_div > 0 and (int(self.renko_directions[-1]) > 0 or int(self.renko_directions[-1]) == 0)) or (
                    gap_div < 0 and (int(self.renko_directions[-1]) < 0 or int(self.renko_directions[-1]) == 0)):
                num_new_bars = gap_div
                is_new_brick = True
                start_brick = 0
            # Backward direction (up -> down or down -> up)
            elif np.abs(gap_div) >= 2:  # Should be double gap at least
                num_new_bars = gap_div
                num_new_bars -= np.sign(gap_div)
                start_brick = 2
                is_new_brick = True
                self.renko_prices.append(
                    str(float(self.renko_prices[-1]) + 2 * float(self.brick_size) * int(np.sign(gap_div))))
                self.renko_directions.append(str(np.sign(gap_div)))
            # else:
            # num_new_bars = 0

            if is_new_brick:
                # Add each brick
                for d in range(start_brick, np.abs(gap_div)):
                    self.renko_prices.append(
                        str(float(self.renko_prices[-1]) + float(self.brick_size) * int(np.sign(gap_div))))
                    self.renko_directions.append(str(np.sign(gap_div)))

        return num_new_bars

    # Getting renko on history
    def build_history(self, prices):
        if len(prices) > 0:
            # Init by start values
            self.source_prices = prices
            self.renko_prices.append(prices.iloc[0])
            self.renko_directions.append(0)

            # For each price in history
            for p in self.source_prices[1:]:
                self.__renko_rule(p)

        return len(self.renko_prices)

    # Getting next renko value for last price
    def do_next(self, last_price):
        if len(self.renko_prices) == 0:
            self.source_prices.append(last_price)
            self.renko_prices.append(last_price)
            self.renko_directions.append(0)
            return 1
        else:
            self.source_prices.append(last_price)
            return self.__renko_rule(last_price)

    # Simple method to get optimal brick size based on ATR
    def __get_optimal_brick_size(self, hlc_history, atr_timeperiod=14):
        brick_size = 0.0

        # If we have enough of data
        if hlc_history.shape[0] > atr_timeperiod:
            brick_size = np.median(talib.ATR(high=np.double(hlc_history.high),
                                             low=np.double(hlc_history.low),
                                             close=np.double(hlc_history.close),
                                             timeperiod=atr_timeperiod)[atr_timeperiod:])

        return round(brick_size, 4)

    def get_renko_prices(self):
        return self.renko_prices

    def get_renko_directions(self):
        return self.renko_directions

    def reset_reward(self):
        self.rewards.clear()
        self.sum = 0.0

    def get_reward(self, reward):
        if len(self.rewards) == 0:
            stale_reward = 0
        else:
            stale_reward = self.rewards.popleft()
        self.sum = self.sum - np.exp(-1 * self.decay_rate) * stale_reward
        self.sum = self.sum * np.exp(-1 * self.decay_rate)
        self.sum = self.sum + reward
        self.rewards.append(reward)
        return self.sum / self.denominator

    def set_qty(self, price):
        self.qty = int((self.balance * self.leverage) / price)

        if self.qty == 0:
            self.done = True

    def _generate_color_graph(self, gap_window=0):

        renko_graph_directions = [float(i) for i in self.renko_directions]

        renko_graph_directions = renko_graph_directions[-(self.obs_window - gap_window):]

        color_graph = np.zeros([self.obs_window, self.obs_window, 3], dtype=np.uint8)

        fill_color = [[255, 0, 0], [0, 255, 0], [255, 255, 255]]

        i = init_i = math.ceil((color_graph.shape[0] / 2))

        values_of_i = []

        for j in range(1, len(renko_graph_directions)):
            i = i - 1 if renko_graph_directions[j] == 1 else i + 1
            values_of_i.append(i)

        spread_of_i = max(values_of_i) - min(values_of_i)

        dist_btw_min_i = init_i - min(values_of_i) if min(values_of_i) > 0 else init_i - (min(values_of_i) + 1)

        i = int((color_graph.shape[0] - spread_of_i) / 2) + dist_btw_min_i
        color_graph[i, 0] = fill_color[1] if renko_graph_directions[0] == 1 else fill_color[0]

        for j in range(1, len(renko_graph_directions)):
            i = i - 1 if renko_graph_directions[j] == 1 else i + 1
            color_graph[i, j] = fill_color[1] if renko_graph_directions[j] == 1 else fill_color[0]

        return color_graph

    def _get_ob(self):
        assert len(self.frames) == self.stack_size
        obs = np.zeros([self.obs_window, self.obs_window, self.stack_size], dtype=np.uint8)
        for i in range(self.stack_size):
            obs[:, :, i] = self.frames[i]
        obs = np.ndarray.flatten(obs)
        return obs

    def _next_observation(self):
        while True:
            observations = self.exchange.data_frame.iloc[self.current_step]
            new_renko_bars = self.do_next(pd.Series([observations['close']]))
            if new_renko_bars == 0 and self.market_open:
                if self.current_step + 1 <= len(self.exchange.data_frame) - 1:
                    # Automatically perform hold
                    if self.enable_logging:
                        self.logger.info(
                            "{} Auto Hold : Renko_bars: {} : Brick_size: {} : Brick_size_per: {}".format(
                                self._current_timestamp(), new_renko_bars, self.brick_size, self.brick_size_per))

                    self._is_auto_hold = True
                    self._take_action(action=0)
                    self._is_auto_hold = False
                    self.current_step += 1

                else:
                    self.done = True
                    break
            else:
                break

        self.frames.append(
            self._transform_obs(self._generate_color_graph(), width=self.obs_window, height=self.obs_window))
        return self._get_ob()

    def _transform_obs(self, obs, resize=False, width=32, height=32, binary=False):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        if binary:
            (thr, obs) = cv2.threshold(obs, 10, 255, cv2.THRESH_BINARY)

        if resize:
            obs = cv2.resize(obs, (width, height), interpolation=cv2.INTER_AREA)

        return obs

    def _current_price(self):
        return self.exchange.data_frame['close'].values[self.current_step]

    def _current_timestamp(self):
        return self.exchange.data_frame.index[self.current_step]

    def _current_date(self):
        return pd.to_datetime(self._current_timestamp()).date()

    def _next_date(self):
        if self.current_step + 1 != len(self.exchange.data_frame) - 1:
            return pd.to_datetime(self.exchange.data_frame.index[self.current_step+1]).date()

    def _is_day_over(self):
        if self.current_step + 1 != len(self.exchange.data_frame) - 1:
            return self._current_date() != self._next_date()
        else:
            return True

    def _take_action(self, action):
        current_price = self._current_price()

        action_type = int(action)  # [0, 1, 2]

        reward = 0
        self.position_record = ""
        self.action_record = ""
        # set next time
        if self._is_day_over():
            # auto square-off at 3:20 pm and skip to next day
            # Check of trades taken
            self.tradable = False
            if len(self.positions) != 0:
                if self.enable_logging:
                    self.logger.info("{} Auto Square-Off".format(self._current_timestamp()))
                if self.short:
                    action_type = 1
                else:
                    action_type = 2

        # act = 0: hold, 1: buy, 2: sell
        if action_type == 1 and self.market_open:  # buy
            if len(self.positions) == 0:
                if self.tradable:
                    # Going Long
                    self.positions.append(float(current_price))
                    self.set_qty(float(current_price))
                    self.short = False
                    self.balance -= mean(self.positions) * self.qty
                    message = "{}: Action: {} ; Reward: {}".format(self._current_timestamp(), "Long",
                                                                   round(reward, 3))
                    self.action_record = message
                    if self.enable_logging:
                        self.logger.info(message)

            elif not self.short and len(self.positions) != 0:
                # If stock has been already long
                reward = 0
                message = "{}: Don't try to go long more than once!".format(self._current_timestamp())
                self.action_record = message
                if self.enable_logging:
                    self.logger.info(message)

            else:
                # exit from Short Sell
                profits = cal_profit_w_brokerage(float(current_price), mean(self.positions), self.qty)
                profit_per_stock = profits / self.qty
                profit_percent = (profit_per_stock / mean(self.positions)) * 100
                self.balance += (mean(self.positions) * self.qty) + profits
                if profit_percent > 0.0:
                    self.wins += 1
                else:
                    self.losses += 1
                self.profit_per += round(profit_percent, 3)
                reward += self.get_reward(profit_percent)

                # Save the record of exit
                self.position_record = "{}: Qty : {} ; Avg: {} ; Ltp: {} ; P&L: {} ; %Chg: {}".format(
                    self._current_timestamp(),
                    self.qty * -1,
                    round(mean(self.positions), 2),
                    round(float(current_price), 2),
                    round(profits, 2),
                    round(profit_percent, 3))
                self.profits += profits
                self.positions.clear()
                self.short = False
                self.reset_reward()
                self.qty = int(0)
                message = "{}: Action: {} ; Reward: {} ; Profit_Per: {}".format(self._current_timestamp(),
                                                                                "Exit Short",
                                                                                round(reward, 3),
                                                                                round(profit_percent, 2))
                self.action_record = message

                # Set Optimal Box Size
                self._set_optimal_box_size()

                if self.enable_logging:
                    self.logger.info(self.position_record)
                    self.logger.info(message)

        elif action_type == 0 and self.hold_reward:  # hold
            if len(self.positions) > 0:
                profits = cal_profit_w_brokerage(float(current_price), mean(self.positions), self.qty)
                profit_per_stock = profits / self.qty
                profit_percent = (profit_per_stock / mean(self.positions)) * 100

                reward += self.get_reward(profit_percent) * 0.01  # 1/100 of the real profit

                message = "{}: Action: {} ; Reward: {}".format(self._current_timestamp(), "Hold",
                                                               round(reward, 3))
                self.action_record = message
                if self.enable_logging and not self._is_auto_hold:
                    self.logger.info(message)

            else:
                self.action_record = "Thinking for next move!" if self.market_open else "##-------------##"
                message = "{}: {}".format(self._current_timestamp(), self.action_record)
                if self.enable_logging and not self._is_auto_hold:
                    self.logger.info(message)

        elif action_type == 2 and self.market_open:  # sell
            if len(self.positions) == 0:
                # Going Short
                if self.tradable:
                    self.positions.append(float(current_price))
                    self.set_qty(float(current_price))
                    self.short = True
                    self.balance -= mean(self.positions) * self.qty
                    message = "{}: Action: {} ; Reward: {}".format(self._current_timestamp(), "Short",
                                                                   round(reward, 3))
                    self.action_record = message
                    if self.enable_logging:
                        self.logger.info(message)

            elif self.short and len(self.positions) != 0:
                # If stock has been already short
                reward = 0
                message = "{}: Don't try to short more than once!".format(self._current_timestamp())
                self.action_record = message
                if self.enable_logging:
                    self.logger.info(message)

            else:
                # exit from the Long position
                profits = cal_profit_w_brokerage(mean(self.positions), float(current_price), self.qty)
                profit_per_stock = profits / self.qty
                profit_percent = (profit_per_stock / mean(self.positions)) * 100
                self.balance += (mean(self.positions) * self.qty) + profits

                if profit_percent > 0.0:
                    self.wins += 1
                else:
                    self.losses += 1
                self.profit_per += round(profit_percent, 3)
                reward += self.get_reward(profit_percent)

                # Save the record of exit
                self.position_record = "{}: Qty : {} ; Avg: {} ; Ltp: {} ; P&L: {} ; %Chg: {}".format(
                    self._current_timestamp(),
                    self.qty,
                    round(mean(self.positions), 2),
                    round(float(current_price), 2),
                    round(profits, 2),
                    round(profit_percent, 3))
                self.profits += profits
                self.positions.clear()
                self.short = False
                self.reset_reward()
                self.qty = int(0)
                message = "{}: Action: {} ; Reward: {} ; Profit_Per: {}".format(self._current_timestamp(),
                                                                                "Exit Long",
                                                                                round(reward, 3),
                                                                                round(profit_percent, 2))
                self.action_record = message

                # Set Optimal Box Size
                self._set_optimal_box_size()

                if self.enable_logging:
                    self.logger.info(self.position_record)
                    self.logger.info(message)

        if self._is_day_over():
            # close Market at 3:20 pm and skip to next day
            if self.enable_logging:
                self.logger.info("{} Market Closed".format(self._current_timestamp()))
            self.market_open = False

        if self._is_day_over():
            self.market_open = True
            self.tradable = True
            # Log for the day
            if self.enable_logging:
                self.logger.info(
                    "{}: Net_worth: {} Total Profits: {} Total Profit_Per: {}".format(
                        self._current_timestamp(),
                        round(self.net_worth[0], 2),
                        round(self.profits, 2),
                        round(self.profit_per, 3), ))

            self.daily_profit_per.append(round(self.profit_per, 3))
            self.profit_per = 0.0
            # Reset Profits for the day
            self.profits = 0.0
            # Set Optimal Box Size
            self._set_optimal_box_size()

        self.position_value = 0
        self.position_value += (float(current_price)) * self.qty
        self.net_worth.append(self.balance + self.position_value)

        if self.market_open:
            if self.enable_logging and not self._is_auto_hold:
                self.logger.info(
                    "{}: Balance: {} Net_worth: {} Stk_Qty: {} Pos_Val: {} Profits: {} Profit_Per: {}".format(
                        self._current_timestamp(),
                        round(self.balance, 2),
                        round(self.net_worth[0], 2),
                        round(self.qty),
                        round(self.position_value, 2),
                        round(self.profits, 2),
                        round(self.profit_per, 3), ))

        # clip reward
        reward = round(self.clip_reward(reward), 3)

        return reward

    def clip_reward(self, reward):
        return reward / 1.0

    def _done(self):
        self.done = self.net_worth[0] < self.initial_balance / 2 or self.current_step == len(self.exchange.data_frame)-1
        return self.done

    def _set_history(self):
        current_idx = self.current_step
        past_data = self.exchange.data_frame[-self.look_back_window_size + current_idx:current_idx]

        self.build_history(past_data['close'])

    def _set_optimal_box_size(self):
        current_idx = self.current_step
        past_data = self.exchange.data_frame[-self.look_back_window_size + current_idx:current_idx]

        # self.set_brick_size(auto=False, brick_size=get_optimal_box_size(past_data))  # Using Optimal_brick_size
        self.set_brick_size(auto=True, hlc_history=past_data)  # Using ATR
        self.brick_size_per = round((self.brick_size/self._current_price()) * 100, 4)

    def reset(self):
        self.balance = self.initial_balance
        self.source_prices = []
        self.renko_prices = []
        self.renko_directions = []
        self.brick_size = 10.0
        self.brick_size_per = 0.0

        self.frames.clear()

        if int(self.look_back_window_size / self.day_step_size) > 1:
            self.current_step = int(self.day_step_size) * int(self.look_back_window_size / self.day_step_size)
        else:
            self.current_step = int(self.day_step_size)

        self.stock_name = self.exchange.reset()

        self._set_optimal_box_size()
        self._set_history()

        self.net_worth.clear()
        self._is_auto_hold = False
        self.done = False
        self.wins = int(0)
        self.losses = int(0)
        self.short = False
        self.tradable = True
        self.market_open = True
        self.qty = int(0)
        self.profit_per = float(0.0)
        self.daily_profit_per.clear()
        self.profits = int(0)
        self.positions.clear()
        self.position_value = int(0)
        self.action_record = ""
        self.position_record = ""
        self.rewards.clear()
        self.sum = 0.0
        self.denominator = np.exp(-1 * self.decay_rate)

        ob = self._transform_obs(self._generate_color_graph())

        for _ in range(self.stack_size):
            self.frames.append(ob)

        return self._get_ob()

    def step(self, action):
        reward = self._take_action(action)

        self.current_step += 1

        obs = self._next_observation()
        self._done()

        return obs, reward, self.done, {}

    def render(self, mode='system'):
        pass

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
