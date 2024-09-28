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
from math import fabs,sqrt
import shutil
import glob, os
import sys, os

run_mode = 'train' # train or test
start_new_generation = False # Delete the current directory of saves, to start training from the beginning
experiment_name = 'EvolutionaryAlgorithm1'

# Enemy and fitness function
enemy = 8
enemy_life_weight = 0.9
player_life_weight = 0.1

# Parameters
upperbound = 1
lowerbound = -1
population_size = 100
generations = 30
mutation_strength = 1 # Gaussian sigma
p_mutation = 0.4
p_cross = 0.3
elite_fraction = 0.05

tournament_k = 3
crossover_k = 20 # k-point crossover

class environm(Environment):
    # implements fitness function
    def fitness_single(self):
        return enemy_life_weight*(100 - self.get_enemylife()) + player_life_weight*self.get_playerlife() - np.log(self.get_time())


# choose this for not using visuals and thus making experiments faster
headless = False
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

env = Environment(experiment_name=experiment_name,
              enemies=[enemy],
              playermode="ai",
              player_controller=player_controller(n_hidden_neurons),
              enemymode="static",
              level=2,
              speed="fastest",
              visuals=False)

env.state_to_log()

begin_time = time.time()  # sets time marker

number_of_weights = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5



def evaluate(env, individual):
    fit, plife, elife, runtime = env.play(pcont=individual)
    return fit


def initialize_population():
    population = np.random.uniform(lowerbound, upperbound, (population_size, number_of_weights))
    population_fitness = np.array([evaluate(env, individual) for individual in population])
    return population, population_fitness


def tournament(k, population, population_fitness):
    population_size = len(population)
    curr_best = np.random.randint(0, population_size)

    for _ in range(k-1):
        contender = np.random.randint(0, population_size)
        if population_fitness[contender] > population_fitness[curr_best]:
            curr_best = contender

    return population[curr_best], population_fitness[curr_best]


def elitism(population, population_fitness, elite_fraction):
    fittest_amount = int(len(population) * elite_fraction)
    elitest_indices = np.argsort(population_fitness)[-fittest_amount:]
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
    evaluate(env, bsol)

    sys.exit(0)


if not os.path.exists(experiment_name+'/evoman_solstate'):
    print( '\nNEW EVOLUTION\n')

    starting_generation = 0
    population, population_fitness = initialize_population()
    best_index = np.argmax(population_fitness)
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

    best_index = np.argmax(population_fitness)
    best_individual = population[best_index]
    best_fitness = population_fitness[best_index]

    mean = np.mean(population_fitness)
    std = np.std(population_fitness)

    # finds last generation number
    file_aux = open(experiment_name + '/gen.txt','r')
    starting_generation = int(file_aux.readline())
    file_aux.close()


# saves results for first pop
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(starting_generation)+' '+str(round(best_fitness, 6))+' '+str(round(mean, 6))+' '+str(round(std, 6)))
file_aux.write('\n'+str(starting_generation)+' '+str(round(best_fitness, 6))+' '+str(round(mean, 6))+' '+str(round(std, 6))   )
file_aux.close()



for gen in range(starting_generation + 1, generations):
    new_generation = np.zeros((population_size, number_of_weights))
    new_fitness = np.zeros(population_size)

    elitest, elitest_fitness = elitism(population, population_fitness, elite_fraction)

    for i in range(len(population) // 2): # 2 children
        parent1, fitness1 = tournament(tournament_k, population, population_fitness)
        parent2, fitness2 = tournament(tournament_k, population, population_fitness)

        child1, child2 = k_point_crossover(p_cross, crossover_k, parent1, parent2, fitness1, fitness2)
        # child1 = discrete_recombination(p_cross, parent1, parent2, fitness1, fitness2, '1')
        # child2 = discrete_recombination(p_cross, parent1, parent2, fitness1, fitness2, '2')

        #mutate
        child1 = mutate(p_mutation, child1, mutation_strength)
        child2 = mutate(p_mutation, child2, mutation_strength)

        new_generation[i*2] = np.clip(child1, lowerbound, upperbound)
        new_generation[i*2 + 1] = np.clip(child2, lowerbound, upperbound)

        new_fitness[i*2] = evaluate(env, new_generation[i*2])
        new_fitness[i*2 + 1] = evaluate(env, new_generation[i*2 + 1])


    #Replace weakest with elitest
    weakest_indices = np.argsort(new_fitness)[:len(elitest)]
    new_generation[weakest_indices] = elitest
    new_fitness[weakest_indices] = elitest_fitness


    population = new_generation
    population_fitness = new_fitness

    mean = np.mean(population_fitness)
    std = np.std(population_fitness)

    best_index = np.argmax(population_fitness)
    if population_fitness[best_index] > best_fitness:
        best_individual = population[best_index]
        best_fitness = population_fitness[best_index]

        # saves file with the best solution
        np.savetxt(experiment_name+'/best.txt', best_individual)

    # saves results
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(gen)+' '+str(round(best_fitness, 6))+' '+str(round(mean, 6))+' '+str(round(std, 6)))
    file_aux.write('\n'+str(gen)+' '+str(round(best_fitness, 6))+' '+str(round(mean, 6))+' '+str(round(std, 6))   )
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

file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()

env.state_to_log() # checks environment state
