"""
    # This Env version has Dynamically Generated Stock Data.
    This version v2 removes all redundant codes and adds some code optimization.
"""


import cv2
import math
import gym
import talib
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from gym import spaces
from collections import deque
from statistics import mean
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from lib.render.StockTradingGraph import StockTradingGraph
from lib.renko.Renko import get_optimal_box_size
from lib.generator.fbm_generator import FBMExchange

# logging
import logging


def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""
    formatter = logging.Formatter('%(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


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


class StockTradingEnv(gym.Env):
    '''A Stock trading environment for OpenAI gym'''
    metadata = {'render.modes': ['human', 'system', 'none']}
    viewer = None

    def __init__(self, config):
        super(StockTradingEnv, self).__init__()

        self.initial_balance = 10000

        self.exchange = FBMExchange()

        self.enable_logging = config['enable_env_logging']
        if self.enable_logging:
            self.logger = setup_logger('env_logger', 'env.log')
            self.logger.info('Env Logger!')

        # Stuff from REnko
        self.source_prices = []
        self.renko_prices = []
        self.renko_directions = []
        self.brick_size = 10.0

        # Stuff from the Env Before
        self.decay_rate = 1e-2
        self.done = False
        self.current_step = int(375)
        self.t = int(0)
        self.wins = int(0)
        self.losses = int(0)
        self.qty = int(0)
        self.short = False
        self.tradable = True
        self.market_open = True
        self.amt = self.initial_balance
        self.qty = int(0)
        self.profit_per = float(0.0)
        self.daily_profit_per = []
        self.profits = int(0)
        self.positions = []
        self.position_value = int(0)
        self.action_record = ""
        self.position_record = ""
        self.rewards = deque(np.zeros(1, dtype=float))
        self.sum = 0.0
        self.denominator = np.exp(-1 * self.decay_rate)

        self.look_back_window_size = config['look_back_window_size'] or 375*5
        self.obs_window = config['observation_window'] or 84

        # Frame Stack
        self.stack_size = config['frame_stack_size'] or 4
        self.frames = deque([], maxlen=self.stack_size)

        # Actions of the format Buy, Sell , Hold .
        self.action_space = spaces.Discrete(3)

        # Observes the price action, indicators, account action, price forecasts
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.obs_window, self.obs_window, self.stack_size),
                                            dtype=np.uint8)

    # Setting brick size. Auto mode is preferred, it uses history
    def set_brick_size(self, HLC_history=None, auto=True, brick_size=10.0):
        if auto:
            self.brick_size = self.__get_optimal_brick_size(HLC_history.iloc[:, [0, 1, 2]])
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
    def __get_optimal_brick_size(self, HLC_history, atr_timeperiod=14):
        brick_size = 0.0

        # If we have enough of data
        if HLC_history.shape[0] > atr_timeperiod:
            brick_size = np.median(talib.ATR(high=np.double(HLC_history.iloc[:, 0]),
                                             low=np.double(HLC_history.iloc[:, 1]),
                                             close=np.double(HLC_history.iloc[:, 2]),
                                             timeperiod=atr_timeperiod)[atr_timeperiod:])

        return brick_size

    def get_renko_prices(self):
        return self.renko_prices

    def get_renko_directions(self):
        return self.renko_directions

    def resetReward(self):
        self.rewards = deque(np.zeros(1, dtype=float))
        self.sum = 0.0

    def getReward(self, reward):
        stale_reward = self.rewards.popleft()
        self.sum = self.sum - np.exp(-1 * self.decay_rate) * stale_reward
        self.sum = self.sum * np.exp(-1 * self.decay_rate)
        self.sum = self.sum + reward
        self.rewards.append(reward)
        return self.sum / self.denominator

    def set_qty(self, price):
        self.qty = int(self.balance / price)

        if self.qty == 0:
            self.done = True

    def _generate_color_graph(self):

        renko_graph_directions = [float(i) for i in self.renko_directions]

        renko_graph_directions = renko_graph_directions[-self.obs_window:]

        color_graph = np.zeros([self.obs_window, self.obs_window, 3], dtype=np.uint8)

        fill_color = [[255, 0, 0], [0, 255, 0], [255, 255, 255]]

        i = init_i = math.ceil((color_graph.shape[0] / 2))

        values_of_i = []

        for j in range(1, len(renko_graph_directions)):
            i = i - 1 if renko_graph_directions[j] == 1 else i + 1
            values_of_i.append(i)

        spread_of_i = max(values_of_i) - min(values_of_i)

        dist_btw_min_i = init_i - min(values_of_i)

        i = int((color_graph.shape[0] - spread_of_i) / 2) + dist_btw_min_i
        color_graph[i, 0] = fill_color[1] if renko_graph_directions[0] == 1 else fill_color[0]

        for j in range(1, len(renko_graph_directions)):
            i = i - 1 if renko_graph_directions[j] == 1 else i + 1
            color_graph[i, j] = fill_color[1] if renko_graph_directions[j] == 1 else fill_color[0]

        return color_graph

    def _get_ob(self):
        assert len(self.frames) == self.stack_size
        obs = np.ones([self.obs_window, self.obs_window, self.stack_size], dtype=np.uint8)
        for i in range(self.stack_size):
            obs[:, :, i] = self.frames[i]
        return obs

    def _next_observation(self):
        while True:
            observations = self.exchange.data_frame.iloc[self.current_step]
            new_renko_bars = self.do_next(pd.Series([observations['close']]))
            if new_renko_bars == 0:
                if self.current_step + 1 <= len(self.exchange.data_frame) - 1:
                    self.current_step += 1
                    # Automatically perform hold
                    self._take_action(action=0)
                else:
                    self.done = True
                    break
            else:
                break

        self.frames.append(self._transform_obs(self._generate_color_graph()))
        return self._get_ob()

    def _transform_obs(self, obs, resize=False, width=84, height=84, binary=False):
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

    def _current_index(self):
        return self.current_step

    def _take_action(self, action):
        current_price = self._current_price()

        action_type = int(action)  # [0, 1, 2]

        reward = 0
        profit_percent = float(0)
        self.position_record = ""
        self.action_record = ""
        # set next time
        self.t = self._current_index()
        if (self.t + 10) % 375 == 0:
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
                    reward = 1
                    self.short = False
                    self.balance -= self.positions[-1] * self.qty
                    message = "{}: Action: {} ; Reward: {}".format(self._current_timestamp(), "Long",
                                                                   round(reward, 3))
                    self.trades.append({'step': self.current_step,
                                        'amount': self.qty,
                                        'total': (self.qty * float(current_price)),
                                        'type': 'buy'})
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
                profits = 0
                profits += cal_profit_w_brokerage(float(current_price), mean(self.positions),
                                                  self.qty)

                avg_profit = profits / self.qty
                profit_percent = (avg_profit / mean(self.positions)) * 100
                self.amt += self.amt * (profit_percent / 100)
                self.balance = self.amt
                if profit_percent > 0.0:
                    self.wins += 1
                else:
                    self.losses += 1
                self.profit_per += round(profit_percent, 3)
                if profit_percent > 0.3 or profit_percent < -0.3:
                    reward += self.getReward(profit_percent) * 300
                else:
                    reward += self.getReward(profit_percent) * 30
                # Save the record of exit
                self.position_record = "{}: Qty : {} ; Avg: {} ; Ltp: {} ; P&L: {} ; %Chg: {}".format(
                    self._current_timestamp(),
                    self.qty * -1,
                    round(mean(self.positions), 2),
                    round(float(current_price), 2),
                    round(profits, 2),
                    round(profit_percent, 3))
                self.profits += profits
                self.positions = []
                self.short = False
                self.resetReward()
                self.qty = int(0)
                message = "{}: Action: {} ; Reward: {} ; Profit_Per: {}".format(self._current_timestamp(),
                                                                                "Exit Short",
                                                                                round(reward, 3),
                                                                                round(profit_percent, 2))
                self.trades.append({'step': self.current_step,
                                    'amount': self.qty,
                                    'total': (self.qty * float(current_price)),
                                    'type': 'buy'})
                self.action_record = message
                if self.enable_logging:
                    self.logger.info(self.position_record)
                    self.logger.info(message)

        elif action_type == 0:  # hold
            if len(self.positions) > 0:
                profits = 0
                for p in self.positions:
                    if self.short:
                        profits += (p - float(current_price))
                    else:
                        profits += (float(current_price) - p)

                avg_profit = profits / len(self.positions)
                profit_percent = (avg_profit / mean(self.positions)) * 100
                reward += self.getReward(profit_percent)
                message = "{}: Action: {} ; Reward: {}".format(self._current_timestamp(), "Hold",
                                                               round(reward, 3))
                self.action_record = message
                if self.enable_logging:
                    self.logger.info(message)

            else:
                self.action_record = "Thinking for next move!" if self.market_open else "##-------------##"
                message = "{}: {}".format(self._current_timestamp(), self.action_record)
                if self.enable_logging:
                    self.logger.info(message)

        elif action_type == 2 and self.market_open:  # sell
            if len(self.positions) == 0:
                # Going Short
                if self.tradable:
                    self.positions.append(float(current_price))
                    self.set_qty(float(current_price))
                    reward = 1
                    self.short = True
                    self.balance -= self.positions[-1] * self.qty
                    message = "{}: Action: {} ; Reward: {}".format(self._current_timestamp(), "Short",
                                                                   round(reward, 3))
                    self.trades.append({'step': self.current_step,
                                        'amount': self.qty,
                                        'total': (self.qty * float(current_price)),
                                        'type': 'sell'})
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
                profits = 0
                profits += cal_profit_w_brokerage(mean(self.positions), float(current_price),
                                                  self.qty)

                avg_profit = profits / self.qty
                profit_percent = (avg_profit / mean(self.positions)) * 100
                self.amt += self.amt * (profit_percent / 100)
                self.balance = self.amt
                if profit_percent > 0.0:
                    self.wins += 1
                else:
                    self.losses += 1
                self.profit_per += round(profit_percent, 3)
                if profit_percent > 0.3 or profit_percent < -0.3:
                    reward += self.getReward(profit_percent) * 300
                else:
                    reward += self.getReward(profit_percent) * 30
                # Save the record of exit
                self.position_record = "{}: Qty : {} ; Avg: {} ; Ltp: {} ; P&L: {} ; %Chg: {}".format(
                    self._current_timestamp(),
                    self.qty,
                    round(mean(self.positions), 2),
                    round(float(current_price), 2),
                    round(profits, 2),
                    round(profit_percent, 3))
                self.profits += profits
                self.positions = []
                self.short = False
                self.resetReward()
                self.qty = int(0)
                message = "{}: Action: {} ; Reward: {} ; Profit_Per: {}".format(self._current_timestamp(),
                                                                                "Exit Long",
                                                                                round(reward, 3),
                                                                                round(profit_percent, 2))
                self.trades.append({'step': self.current_step,
                                    'amount': self.qty,
                                    'total': (self.qty * float(current_price)),
                                    'type': 'sell'})
                self.action_record = message
                if self.enable_logging:
                    self.logger.info(self.position_record)
                    self.logger.info(message)

        if (self.t + 10) % 375 == 0:
            # close Market at 3:20 pm and skip to next day
            if self.enable_logging:
                self.logger.info("{} Market Closed".format(self._current_timestamp()))
            self.market_open = False

        if (self.t + 1) % 375 == 0:
            self.market_open = True
            self.tradable = True
            reward += self.profit_per * 1000  # Bonus for making a profit at the end of the day
            self.daily_profit_per.append(round(self.profit_per, 3))
            self.profit_per = 0.0
            # Reset Profits for the day
            self.profits = 0.0
            # Set Optimal Box Size
            self._set_optimal_box_size()

        self.position_value = 0
        for p in self.positions:
            self.position_value += (float(current_price)) * self.qty
        self.net_worths.append(self.balance + self.position_value)

        self.account_history = self.account_history.append({
            'timestep': ((self.t + 1) % 375),
            'net_worth': round(self.net_worths[-1], 2),
            'stock_qty': self.qty,
            'position_value': round(self.position_value, 2),
            'profits': round(self.profits, 2),
            'profits_per': round(self.profit_per, 3),
        }, ignore_index=True)
        if self.market_open:
            if self.enable_logging:
                self.logger.info(
                    "{}: {} Net_worth: {} Stk_Qty: {} Pos_Val: {} Profits: {} Profit_Per: {}".format(
                        self._current_timestamp(),
                        ((self.t + 1) % 375),
                        round(self.net_worths[-1], 2),
                        round(self.qty),
                        round(self.position_value, 2),
                        round(self.profits, 2),
                        round(self.profit_per, 3), ))

        # clip reward
        reward = round(reward, 3)

        return reward

    def _reward(self):
        length = self.current_step
        returns = np.diff(self.net_worths[-length:])

        if np.count_nonzero(returns) < 1:
            return 0
        reward = 0

        return reward if np.isfinite(reward) else 0

    def _done(self):
        self.done = self.net_worths[-1] < self.initial_balance / 2 or self.current_step == len(self.exchange.data_frame) - 1
        return self.done

    def _set_history(self):
        current_idx = self.current_step
        past_data = self.exchange.data_frame[-self.look_back_window_size + current_idx:current_idx]

        self.build_history(past_data['close'])

    def _set_optimal_box_size(self):
        current_idx = self.current_step
        past_data = self.exchange.data_frame[-self.look_back_window_size + current_idx:current_idx]

        self.set_brick_size(auto=False, brick_size=get_optimal_box_size(past_data))

    def reset(self):
        self.balance = self.initial_balance
        self.stock_held = 0
        self.frames = deque([], maxlen=self.stack_size)

        if int(self.look_back_window_size / 375) > 1:
            self.current_step = int(375) * int(self.look_back_window_size / 375)
        else:
            self.current_step = int(375)

        self.exchange.reset()

        self._set_optimal_box_size()
        self._set_history()

        self.net_worths = [self.initial_balance] * (self.current_step + 1)
        self.initial_step = self.current_step
        self.done = False
        self.t = int(0)
        self.wins = int(0)
        self.losses = int(0)
        self.short = False
        self.tradable = True
        self.market_open = True
        self.amt = self.initial_balance
        self.qty = int(0)
        self.profit_per = float(0.0)
        self.daily_profit_per = []
        self.profits = int(0)
        self.positions = []
        self.position_value = int(0)
        self.action_record = ""
        self.position_record = ""
        self.rewards = deque(np.zeros(1, dtype=float))
        self.sum = 0.0
        self.denominator = np.exp(-1 * self.decay_rate)

        self.account_history = pd.DataFrame([{
            'timestep': 0,
            'net_worth': self.balance,
            'stock_qty': 0,
            'position_value': 0,
            'profits': 0,
            'profits_per': 0,
        }])
        self.trades = []

        ob = self._transform_obs(self._generate_color_graph())

        for _ in range(self.stack_size):
            self.frames.append(ob)

        return self._get_ob()

    def step(self, action):
        reward = self._take_action(action)

        self.current_step += 1

        obs = self._next_observation()
        # reward = self._reward()
        self._done()

        return obs, reward, self.done, {}

    def render(self, mode='human'):
        if mode == 'system':
            print('Price: ' + str(self._current_price()))
            print('\t\t' + '=' * 20 + ' History ' + '=' * 20)
            print(str(self.account_history[-1:]))
            print('Net worth: ' + str(self.net_worths[-1]))

        elif mode == 'human':
            if self.viewer is None:
                self.viewer = StockTradingGraph(self.exchange.data_frame)

            self.viewer.render(self.current_step, self.net_worths, self.initial_step, self.trades, window_size=20)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
