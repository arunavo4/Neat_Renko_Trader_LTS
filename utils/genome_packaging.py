import glob
import neat
import gzip
import random
from neat.six_util import iteritems, itervalues
import pickle

restore_file = "neat-checkpoint-0"

config_default = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             'config.cfg')

stats = neat.StatisticsReporter()
reporter = neat.StdOutReporter(stats)
pop = neat.Population(config_default)
print(len(pop.species.species))
# reporter.end_generation(config_default,pop.population,pop.species)
# generation = 0
#
# files = glob.glob(".\\genomes\\*.pkl")
#
# selected_population = {}
#
# for genomeFile in files:
#     genome = pickle.load(open(genomeFile, 'rb'))
#     selected_population[genome.key] = genome
#
# # species_set = neat.DefaultSpeciesSet()
#
# #
# with gzip.open(restore_file) as f:
#     generation, config, population, species_set, rndstate = pickle.load(f)
#     print(type(generation), type(config), type(population), type(species_set), type(rndstate))
#
#     generation = 0
#     population = selected_population
#
#     species_set.speciate(config_default, population, generation)
#
#     filename = "neat-checkpoint-0"
#     print("Saving checkpoint to {0}".format(filename))
#
#     with gzip.open(filename, 'w', compresslevel=5) as file:
#         data = (generation, config, population, species_set, rndstate)
#         pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
