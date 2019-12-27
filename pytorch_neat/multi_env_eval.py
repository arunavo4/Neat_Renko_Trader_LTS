# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import numpy as np
from statistics import mean

# setup logging
import logging

formatter = logging.Formatter('%(message)s')


def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


logger = setup_logger('trainer_logger', 'training.log')
logger.info('Training Logger!')


class MultiEnvEvaluator:
    def __init__(self, make_net, activate_net, batch_size=1, max_env_steps=None, make_env=None, env_parms=None,
                 envs=None):
        if envs is None:
            self.envs = [make_env(env_parms) for _ in range(batch_size)]
        else:
            self.envs = envs
        self.make_net = make_net
        self.activate_net = activate_net
        self.batch_size = batch_size
        self.max_env_steps = max_env_steps

    def eval_genome(self, genome_id, genome, config, generation, debug=False):
        net = self.make_net(genome, config, self.batch_size)

        fitnesses = np.zeros(self.batch_size)
        states = [env.reset(generation) for env in self.envs]
        dones = [False] * self.batch_size

        step_num = self.envs[0].current_step
        while True:
            step_num = self.envs[0].current_step + 1
            if self.max_env_steps is not None and step_num == self.max_env_steps:
                break
            if debug:
                actions = self.activate_net(
                    net, states, debug=True, step_num=step_num)
            else:
                actions = self.activate_net(net, states)
            assert len(actions) == len(self.envs)
            for i, (env, action, done) in enumerate(zip(self.envs, actions, dones)):
                if not done:
                    state, reward, done, _ = env.step(action)
                    # fitnesses[i] += reward    //TODO: Basically trying to use ultimate P&L as fitness for each genome
                    states[i] = state
                    dones[i] = done
                if done:
                    fitnesses[i] = round(mean(env.daily_profit_per)*100, 3)
                    # fitnesses[i] = (env.net_worth[0] - env.initial_balance)   Doesn't work initial stages.

            if all(dones):
                for i, (env, done) in enumerate(zip(self.envs, dones)):
                    if len(env.daily_profit_per) == 0:
                        avg_profit = 0.0
                    else:
                        avg_profit = round(mean(env.daily_profit_per), 3)
                    message = "Gen#: {} Env #:{} Stock#:{} Genome_id # :{} Fitness :{} Final Amt :{} Days :{} Avg Daily Profit :{} %".format(
                        generation, i, env.stock_name, genome_id, round(fitnesses[i], 2), round(env.net_worth[0], 2),
                        len(env.daily_profit_per), avg_profit)
                    print(message)
                    logger.info(message)
                break

        return sum(fitnesses) / len(fitnesses)
