
import neat
from neat.six_util import iteritems, itervalues
import pickle

# selected_genomes = [698]
#
# for i in range(133, 248):
#     restore_file = "neat-checkpoint-" + str(i)
#     pop = neat.Checkpointer.restore_checkpoint(restore_file)
#
#     for g in itervalues(pop.population):
#         # print(str(g.key))
#         if g.key in selected_genomes:
#             print(str(g.key))
#
#             save_path = './genomes/' + str(g.key) + '.pkl'
#             with open(save_path, 'wb') as output:
#                 pickle.dump(g, output, 1)
#

restore_file = "neat-checkpoint-0"
pop = neat.Checkpointer.restore_checkpoint(restore_file)

for g in itervalues(pop.population):
    print(str(g.key))

    save_path = './genomes/' + str(g.key) + '.pkl'
    with open(save_path, 'wb') as output:
        pickle.dump(g, output, 1)
