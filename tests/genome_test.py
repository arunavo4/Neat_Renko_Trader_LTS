import glob
import multiprocessing
from os import path

from visualization import genome_plots
from statistics import mean
import numpy as np
import neat
import pickle
from utils.data_utils import load_csv, split_data
from lib.features.indicators import add_indicators
from pytorch_neat.recurrent_net import RecurrentNet
from lib.env.TraderRenkoEnv_v3_lite import StockTradingEnv


# logging
# importing module
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


def eval_genome(genome, config, env_data, env_params, pre_obs=None):
    if env_params['pre_computed_observation']:
        env = StockTradingEnv(env_data, pre_obs, **env_params)
    else:
        env = StockTradingEnv(env_data, **env_params)

    max_env_steps = len(env_data) - 1

    ob = env.reset()

    net = RecurrentNet.create(genome, config)

    current_max_fitness = 0
    fitness_current = 0
    counter = 0
    step = 0
    step_max = max_env_steps
    actions = []
    for _ in range(env.t):
        actions.append(0)
    done = False

    while not done:
        env.render()
        states = [ob]
        outputs = net.activate(states).numpy()
        actions.append(np.argmax(outputs))

        ob, rew, done, _ = env.step(np.argmax(outputs))
        # print("id",genome_id,"Step:",step,"act:",np.argmax(nnOutput),"reward:",rew)

        fitness_current += rew
        step += 1

        if fitness_current > current_max_fitness:
            current_max_fitness = fitness_current
            counter = 0
        else:
            counter += 1

        if step >= step_max:
            done = True

        if done or env.amt <= 0:
            done = True
            print("Genome id#: ", genome.key)
            message = "Fitness :{} Max Fitness :{} Avg Daily Profit :{} %".format(fitness_current,
                                                                                  current_max_fitness,
                                                                                  round(mean(env.daily_profit_per), 3))
            print(message)

        genome.fitness = fitness_current

    return env, actions


def run_tests(genome, train_data, test_data, env_params):

    train_env, train_acts = eval_genome(genome, config, train_data, env_params)

    test_env, test_acts = eval_genome(genome, config, test_data, env_params)

    reward_filename = './/genome_plots//' + str(genome.key) + '_reward.png'
    genome_plots.plot_train_test_reward(train_env.daily_profit_per, test_env.daily_profit_per, reward_filename)

    actions_filename = './/genome_plots//' + str(genome.key) + '_actions.png'
    date_split = '2019-01-18'
    message = genome_plots.plot_train_test_actions(genome.key, train_env, test_env, train_acts, test_acts, date_split, actions_filename)

    logger.info(message)


def run_files(files_set, train_data, test_data, env_params):
    for genomeFile in files_set:
        genome = pickle.load(open(genomeFile, 'rb'))
        run_tests(genome,train_data, test_data, env_params)


def chunks(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


# if __name__ == "__main__":
#     threads = []
#
#     # Data Prep
#     input_data_path = path.join('data', 'dataset', 'ADANIPORTS-EQ.csv')
#     feature_df = load_csv(input_data_path)
#     feature_df = add_indicators(feature_df.reset_index())
#
#     train_data, test_data = split_data(feature_df)
#
#     config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
#                          '../data/config/config.cfg')
#
#     params = {
#         'look_back_window_size': 375 * 7,
#         'enable_stationarization': True,
#         'n_processes': multiprocessing.cpu_count(),
#         'pre_computed_observation': True,
#         'enable_env_logging': False
#     }
#
#     # Load all the genomes
#     files = glob.glob(".\\data\\genomes\\*.pkl")
#     n_processes = 3
#
#     logger = setup_logger('genome_logger', 'genome_test.log')
#     # divide the file-list
#     chunks_list = chunks(files, n_processes)
#
#     for i in range(n_processes):
#         threads.append(multiprocessing.Process(target=run_files, args=(chunks_list[i],train_data,test_data,params,)))
#
#     # start all threads
#     for t in threads:
#         t.start()
#
#     # Join all threads
#     for t in threads:
#         t.join()


# Single genome
if __name__ == "__main__":
    logger = setup_logger('genome_logger', 'genome_test.log')

    input_data_path = path.join('..','data', 'dataset', 'ADANIPORTS-EQ.csv')
    feature_df = load_csv(input_data_path)
    feature_df = add_indicators(feature_df.reset_index())

    train_data, test_data = split_data(feature_df)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         '../data/config/config.cfg')

    params = {
        'look_back_window_size': 375 * 7,
        'enable_stationarization': True,
        'n_processes': multiprocessing.cpu_count(),
        'pre_computed_observation': False,
        'enable_env_logging': True
    }

    genomeFile = '../winner.pkl'
    genome = pickle.load(open(genomeFile, 'rb'))

    run_tests(genome,train_data, test_data, params)

