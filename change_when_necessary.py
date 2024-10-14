################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt,ceil,floor
import shutil
import glob, os
import sys, os

run_mode = 'train' # train or test
start_new_generation = False # Delete the current directory of saves, to start training from the beginning
experiment_name = 'change_when_necessary'

# Enemy and fitness function
enemies = [4,7,8]
enemy_life_weight = 0.9
player_life_weight = 0.1

# Parameters
upperbound = 1
lowerbound = -1
population_size = 100
generations = 30
mutation_strength = 1 # Gaussian sigma
p_mutation = 0.02
p_cross = 0.4
elite_fraction = 0.05

tournament_k = 3
crossover_k = 20 # k-point crossover
# parentsn = 2

class environm(Environment):
    def multiple(self,pcont,econt):
        vfitness, vplayerlife, venemylife, vtime = [],[],[],[]
        for e in self.enemies:
            fitness, playerlife, enemylife, time  = self.run_single(e,pcont,econt)
            vfitness.append(fitness)
            vplayerlife.append(playerlife)
            venemylife.append(enemylife)
            vtime.append(time)

        return vfitness, vplayerlife, venemylife, vtime


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


# Remove the saved states to start training over again
if start_new_generation:
    if os.path.exists(experiment_name):
        shutil.rmtree(experiment_name)

# Create save directory
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


n_hidden_neurons = 10

env = environm(experiment_name=experiment_name,
              enemies=enemies,
              multiplemode="yes",
              playermode="ai",
              player_controller=player_controller(n_hidden_neurons),
              enemymode="static",
              level=2,
              speed="fastest",
              visuals=False)

env.state_to_log()

begin_time = time.time()  # sets time marker

number_of_weights = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5


def evaluate(env, individual, weights):
    fit, plife, elife, runtime = env.play(pcont=individual)
    total_fitness = fitness_value(fit, weights)
    fit.append(total_fitness)
    return fit

weights = [ 1 for enemy in enemies ]
def fitness_value(vfitness, weights):
    return np.average(vfitness, weights=weights) - np.std(vfitness)


def initialize_population():
    population = np.random.uniform(lowerbound, upperbound, (population_size, number_of_weights))
    population_fitness = np.array([evaluate(env, individual, weights) for individual in population])
    return population, population_fitness


def tournament(k, population, population_fitness):
    population_size = len(population)
    curr_best = np.random.randint(0, population_size)

    for _ in range(k-1):
        contender = np.random.randint(0, population_size)
        if population_fitness[contender][-1] > population_fitness[curr_best][-1]:
            curr_best = contender

    return population[curr_best], population_fitness[curr_best]


def elitism(population, population_fitness, elite_fraction):
    fittest_amount = int(len(population) * elite_fraction)
    elitest_indices = np.argsort(population_fitness[:,-1], axis=-1)[-fittest_amount:]
    elites = population[elitest_indices]
    elites_fitness = population_fitness[elitest_indices]
    return elites, elites_fitness


def k_point_crossover(p_cross, k, parent1, parent2, parent1fit, parent2fit):
    parent_size = len(parent1)


    # k = np.random.randint(0, max_k)

    if np.random.uniform() > p_cross:
        return parent1, parent2
    else:
        cuts = sorted(np.random.choice(range(1, parent_size - 1), k, replace=False))
        cuts = np.append(cuts, parent_size)
        cuts = cuts.astype(int)

        # print(parent1fit, parent2fit)
        cuts_assigned = np.random.choice([1,2], size=k+1)

        child1 = np.array([])
        child2 = np.array([])

        current = 0
        for i, parent_number in enumerate(cuts_assigned):
            endpoint = cuts[i]
            if parent_number == 1:
                child1 = np.append(child1, parent1[current:endpoint])
                child2 = np.append(child2, parent2[current:endpoint])
            elif parent_number == 2:
                child1 = np.append(child1, parent2[current:endpoint])
                child2 = np.append(child2, parent1[current:endpoint])

            current = endpoint

    return child1, child2

def multi_parent_crossover(p_cross: float, parents: list, parents_fit):
    if np.random.uniform() > p_cross:
        return parents

    parent_size = len(parents[0])
    num_parents = len(parents)

    cuts = np.sort(np.random.choice(range(1, parent_size-1), num_parents, False))
    cuts = np.append(cuts, parent_size)

    children = [[] for _ in range(num_parents)]

    for i, cut in enumerate(cuts):
        for j, child in enumerate(children):
            current_parent_i = (j+i)%num_parents
            child.extend(parents[current_parent_i][len(child):cut])

    children = [np.array(child) for child in children]

    return children



# def discrete_recombination(p_cross, parent1, parent2, parent1fit, parent2fit, parent_number):
#     parent_size = len(parent1)

#     # k = np.random.randint(0, max_k)

#     if np.random.uniform() > p_cross:
#         if parent_number == '1':
#             return parent1
#         else:
#             return parent2
#     else:
#         print(parent1fit, parent2fit)

#         child = np.array([])

#         for i in range(parent_size):
#             if np.random.uniform() > 0.5:
#                 child = np.append(child, parent1[i])
#             else:
#                 child = np.append(child, parent2[i])

#     return child

def mutate(p_mutation, individual, sigma):
    for i in range(len(individual)):
        if np.random.uniform() < p_mutation:
            individual[i] += np.random.normal(0, sigma)

    return individual



# loads file with the best solution for testing
if run_mode =='test':
    bsol = np.loadtxt(experiment_name + '/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    env.update_parameter('visuals', True)
    evaluate(env, bsol, weights)

    sys.exit(0)


if not os.path.exists(experiment_name+'/evoman_solstate'):
    print( '\nNEW EVOLUTION\n')

    starting_generation = 0
    population, population_fitness = initialize_population()
    best_index = np.argmax(population_fitness[:,-1])
    best_individual = population[best_index]
    best_fitness = population_fitness[best_index]

    mean = np.mean(population_fitness)
    std = np.std(population_fitness)

    solutions = [population, population_fitness]
    env.update_solutions(solutions)

else:
    print( '\nCONTINUING EVOLUTION\n')

    env.load_state()
    population = env.solutions[0]
    population_fitness = env.solutions[1]

    best_index = np.argmax(population_fitness[:,-1])
    best_individual = population[best_index]
    best_fitness = population_fitness[best_index]

    mean = np.mean(population_fitness[:,-1])
    std = np.std(population_fitness[:,-1])

    # finds last generation number
    file_aux = open(experiment_name + '/gen.txt','r')
    starting_generation = int(file_aux.readline())
    file_aux.close()


# saves results for first pop
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(starting_generation)+' '+str(round(best_fitness[-1], 6))+' '+str(round(mean, 6))+' '+str(round(std, 6)))
file_aux.write('\n'+str(starting_generation)+' '+str(round(best_fitness[-1], 6))+' '+str(round(mean, 6))+' '+str(round(std, 6))   )
file_aux.close()



for gen in range(starting_generation + 1, generations+1):
    new_generation = np.zeros((population_size, number_of_weights))

    elitest, elitest_fitness = elitism(population, population_fitness, elite_fraction)

    # Linearly decrease from 8 to 2 parents over time
    # parentsn = floor(gen / -5 + 8.2)
    parentsn = 32 - gen
    # parentsn = 1 + gen

    for i in range(ceil(len(population) / parentsn)):
        parents = []
        fitnesses = []
        for j in range(parentsn):
            parent, fitness = tournament(tournament_k, population, population_fitness)
            parents.append(parent)
            fitnesses.append(fitness)

        children = multi_parent_crossover(p_cross, parents, fitnesses)

        # child1 = discrete_recombination(p_cross, parent1, parent2, fitness1, fitness2, '1')
        # child2 = discrete_recombination(p_cross, parent1, parent2, fitness1, fitness2, '2')

        #mutate
        children = [mutate(p_mutation, child, mutation_strength) for child in children]

        if (i+1)*parentsn > len(population):
            # We might be generating too many children; simply cut off the last ones.
            children = children[:len(population)-i*parentsn]

        for j, child in enumerate(children):
            new_generation[i*parentsn + j] = np.clip(child, lowerbound, upperbound)

    new_fitness = np.array([evaluate(env, individual, weights) for individual in new_generation])

    #Replace weakest with elitest
    weakest_indices = np.argsort(new_fitness[:,-1])[:len(elitest)]
    new_generation[weakest_indices] = elitest
    new_fitness[weakest_indices] = elitest_fitness


    population = new_generation
    population_fitness = new_fitness

    mean = np.mean(population_fitness[:,-1])
    std = np.std(population_fitness[:,-1])

    std_between_enemies = np.std(np.mean(population_fitness, axis=1))
    print("std deviation between enemies: " + str(std_between_enemies))
    if std_between_enemies > 30-gen:
        averages = population_fitness[:,:-1].mean(axis=0)
        weights = 1 / averages
        print("updating weights to " + str(weights))
        p_cross = .9
    else:
        weights = [ 1 for enemy in enemies ]
        p_cross = .4

    best_index = np.argmax(population_fitness[:,-1])
    if population_fitness[best_index][-1] > best_fitness[-1]:
        best_individual = population[best_index]
        best_fitness = population_fitness[best_index]

        # saves file with the best solution
        np.savetxt(experiment_name+'/best.txt', best_individual)

    # saves results
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(gen)+' '+str(round(best_fitness[-1], 6))+' '+str(round(mean, 6))+' '+str(round(std, 6)))
    file_aux.write('\n'+str(gen)+' '+str(round(best_fitness[-1], 6))+' '+str(round(mean, 6))+' '+str(round(std, 6))   )
    file_aux.close()

    # saves generation number
    file_aux  = open(experiment_name+'/gen.txt','w')
    file_aux.write(str(gen))
    file_aux.close()

    # saves simulation state
    solutions = [population, population_fitness]
    env.update_solutions(solutions)
    env.save_state()


end_time = time.time() # prints total execution time for experiment
print( '\nExecution time: ' + str(round((end_time - begin_time)/60)) + ' minutes \n')
print( '\nExecution time: ' + str(round((end_time - begin_time))) + ' seconds \n')

print(population[best_index])

file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()

env.state_to_log() # checks environment state
