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
from deap import creator, base, tools, algorithms




run_mode = 'train' # train or test
start_new_generation = False # Delete the current directory of saves, to start training from the beginning
experiment_name = 'EA2'

# Enemy and fitness function
fitness_implementation = "weighted"
enemies = [3, 4, 7]
# enemies = [1, 2, 3, 4, 5, 6, 7, 8]

enemy_life_weight = 0.9
player_life_weight = 0.1
weight_priority = 0.05

# Parameters
upperbound = 1
lowerbound = -1
population_size = 100
generations = 60

frac_exploitation_group = 0.5
frac_exploration_group = 1 - frac_exploitation_group

# Exploit parameters
exploit_mutation_strength = 0.1 # Gaussian sigma
exploit_p_mutation = 0.3
exploit_p_cross = 0.3
exploit_tournament_k = 5
elite_fraction = 0.05
exploit_crossover_eta = 30

# Explore parameters
explore_mutation_strength = 0.6 # Gaussian sigma
explore_p_mutation = 0.3
explore_p_cross = 0.3
explore_tournament_k = 2
explore_crossover_eta = 1


elite_amount = int((population_size // 2) * elite_fraction)

class Environment(Environment):
    def cons_multi_fitness(self,values):
        """
        Function used by Environment.multiple().
        Prioritizes the least competitive fitness value by a set amount (weight_priority).
        Deprioritizes other fitness values by the same amount downscaled equally betweewn them.

        param values: fitness values reflecting the algorithm's quality against different enemies
        """
        num_enemies = len(values)
        default_weight = 1 / num_enemies
        weights = [default_weight] * num_enemies

        weight_depriority = weight_priority / num_enemies

        # weight_priority = 0.05 # relative change 0.38 vs 0.25, 0.25
        fitnesses = [(enemy, fitness) for enemy, fitness in zip(enemies, values)]
        min_enemy, min_fitness = min(fitnesses, key=lambda x: x[1])
        # self.min_enemy = min_enemy
        
        for i, (e, fitness) in enumerate(fitnesses):
            if e == min_enemy:
                weights[i] = default_weight + weight_depriority*(num_enemies - 1)
            else:
                weights[i] = default_weight - weight_depriority

        aggregate_fitness = sum([(w * v) for w, v in zip(weights, values)])

        return aggregate_fitness

    # repeats run for every enemy in list
    def multiple(self,pcont,econt):
        """
        Original Environment.multiple() with cons_multi() for vfitness substituted with cons_multi_fitness().
        """

        vfitness, vplayerlife, venemylife, vtime = [],[],[],[]
        for e in self.enemies:

            fitness, playerlife, enemylife, time  = self.run_single(e,pcont,econt)
            vfitness.append(fitness)
            vplayerlife.append(playerlife)
            venemylife.append(enemylife)
            vtime.append(time)

        if fitness_implementation == "classic":
            vfitness = self.cons_multi(np.array(vfitness))
        elif fitness_implementation == "weighted":
            vfitness = self.cons_multi_fitness(np.array(vfitness))

        vplayerlife = self.cons_multi(np.array(vplayerlife))
        venemylife = self.cons_multi(np.array(venemylife))
        vtime = self.cons_multi(np.array(vtime))

        return  vfitness, vplayerlife, venemylife, vtime

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
              enemies=enemies,
              multiplemode='yes',
              playermode="ai",
              player_controller=player_controller(n_hidden_neurons),
              enemymode="static",
              level=2,
              speed="fastest",
              randomini="yes",
              visuals=False)

env.state_to_log()

begin_time = time.time()  # sets time marker

number_of_weights = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5



# DEAP framework initialization
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, lowerbound, upperbound)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, number_of_weights)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    fit, plife, elife, runtime = env.play(pcont=individual)
    return (fit,)


def evaluate_multi(individual):
    fitness_values = []
    for enemy in enemies:
        fitness, plife, elife, runtime = env.run_single(enemy, pcont=individual, econt="None")
        fitness_values.append(fitness)

    return fitness_values


toolbox.register("evaluate", evaluate)
# toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=exploit_crossover_eta, low=lowerbound, up=upperbound)
toolbox.register("mate", tools.cxSimulatedBinary, eta=exploit_crossover_eta)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=exploit_mutation_strength, indpb=exploit_p_mutation)
toolbox.register("select", tools.selTournament, tournsize=exploit_tournament_k)


def initialize_population():
    population = toolbox.population(n=population_size)
    return population


def group_division(population):
    all_fitnesses = []
    for individual in population:
        fitnesses = evaluate_multi(individual)
        all_fitnesses.append(fitnesses)

    all_fitnesses = np.array(all_fitnesses)
    mean_fitnesses = np.mean(all_fitnesses, axis=0)

    enemies_beat = []
    for i, individual in enumerate(population):
        enemies_beat.append(np.sum(all_fitnesses[i] > mean_fitnesses))

    # Sort the amount of enemies beaten in descending order
    indices = np.argsort(np.array(enemies_beat))[::-1]
    sorted_population = [population[indice] for indice in indices]
    half = len(population) // 2

    exploitation_group = sorted_population[:half]
    exploration_group = sorted_population[half:]

    return exploitation_group, exploration_group


def exploit(exploitation_group, p_cross, crossover_eta, p_mutation, mutation_strength, tournament_k, elite_amount):
    elites = tools.selBest(exploitation_group, elite_amount)

    # New k for tournament selection
    toolbox.unregister("select")
    toolbox.register("select", tools.selTournament, tournsize=tournament_k)

    offspring = toolbox.select(exploitation_group, len(exploitation_group) - len(elites))
    offspring = map(toolbox.clone, offspring)

    # Change mutation probability and strength for exploitation
    toolbox.unregister("mutate")
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=mutation_strength, indpb=p_mutation)
    toolbox.unregister("mate")
    toolbox.register("mate", tools.cxSimulatedBinary, eta=exploit_crossover_eta)


    offspring = algorithms.varAnd(offspring, toolbox, p_cross, p_mutation)

    for individual in offspring:
        individual[:] = np.clip(individual, lowerbound, upperbound)

    return offspring + elites


def explore(exploration_group, p_cross, crossover_eta, p_mutation, mutation_strength, tournament_k):
    # New k for tournament selection
    toolbox.unregister("select")
    toolbox.register("select", tools.selTournament, tournsize=tournament_k)

    offspring = toolbox.select(exploration_group, len(exploration_group))
    offspring = map(toolbox.clone, offspring)

    # Change mutation probability and strength for exploration
    toolbox.unregister("mutate")
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=mutation_strength, indpb=p_mutation)
    toolbox.unregister("mate")
    toolbox.register("mate", tools.cxSimulatedBinary, eta=exploit_crossover_eta)


    offspring = algorithms.varAnd(offspring, toolbox, p_cross, p_mutation)

    for individual in offspring:
        individual[:] = np.clip(individual, lowerbound, upperbound)

    return offspring



# loads file with the best solution for testing
if run_mode == 'test':
    bsol = np.loadtxt(experiment_name + '/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    env.update_parameter('visuals', True)
    fit, plife, elife, runtime = env.play(pcont=bsol)
    print("Gain:", plife - elife)

    sys.exit(0)


if not os.path.exists(experiment_name+'/evoman_solstate'):
    print( '\nNEW EVOLUTION\n')

    starting_generation = 0
    population = initialize_population()

    solutions = [population]
    env.update_solutions(solutions)

else:
    print( '\nCONTINUING EVOLUTION\n')

    env.load_state()
    population = env.solutions[0]

    # finds last generation number
    file_aux = open(experiment_name + '/gen.txt','r')
    starting_generation = int(file_aux.readline())
    file_aux.close()


fitnesses = toolbox.map(toolbox.evaluate, population)
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit

best_individual = tools.selBest(population, 1)[0]
best_solution = np.array(best_individual)
best_fitness = best_individual.fitness.values[0]


# Collect all non weighted fitness values for reporting statistics
fitness_implementation = "classic"
reporting_fitnesses = [evaluate(individual) for individual in population]
mean_fitness = np.mean(reporting_fitnesses)
std_fitness = np.std(reporting_fitnesses)
fitness_implementation = "weighted"

# saves results for first pop
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(starting_generation)+' '+str(round(best_fitness, 6))+' '+str(round(mean_fitness, 6))+' '+str(round(std_fitness, 6)))
file_aux.write('\n'+str(starting_generation)+' '+str(round(best_fitness, 6))+' '+str(round(mean_fitness, 6))+' '+str(round(std_fitness, 6))   )
file_aux.close()


for gen in range(starting_generation + 1, generations):
    exploitation_group, exploration_group = group_division(population)

    exploited_offspring = exploit(exploitation_group, exploit_p_cross, exploit_crossover_eta, exploit_p_mutation, exploit_mutation_strength, exploit_tournament_k, elite_amount)
    explored_offspring = explore(exploration_group, explore_p_cross, explore_crossover_eta, explore_p_mutation, explore_mutation_strength, explore_tournament_k)

    offspring = exploited_offspring + explored_offspring

    fitnesses = toolbox.map(toolbox.evaluate, offspring)
    for ind, fit in zip(offspring, fitnesses):
        ind.fitness.values = fit

    population[:] = offspring

    # update best solution
    best = tools.selBest(population, 1)[0]
    if best.fitness.values[0] > best_fitness:
        # saves file with the best solution
        best_solution = np.array(best)
        np.savetxt(experiment_name+'/best.txt', best_solution)
        best_fitness = best.fitness.values[0]


    # Collect all non weighted fitness values for reporting statistics
    fitness_implementation = "classic"
    reporting_fitnesses = [evaluate(individual) for individual in population]
    mean_fitness = np.mean(reporting_fitnesses)
    std_fitness = np.std(reporting_fitnesses)
    fitness_implementation = "weighted"


    # saves results
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(gen)+' '+str(round(best_fitness, 6))+' '+str(round(mean_fitness, 6))+' '+str(round(std_fitness, 6)))
    file_aux.write('\n'+str(gen)+' '+str(round(best_fitness, 6))+' '+str(round(mean_fitness, 6))+' '+str(round(std_fitness, 6))   )
    file_aux.close()

    # saves generation number
    file_aux  = open(experiment_name+'/gen.txt','w')
    file_aux.write(str(gen))
    file_aux.close()

    # saves simulation state
    solutions = [population]
    env.update_solutions(solutions)
    env.save_state()

end_time = time.time() # prints total execution time for experiment
print( '\nExecution time: ' + str(round((end_time - begin_time)/60)) + ' minutes \n')
print( '\nExecution time: ' + str(round((end_time - begin_time))) + ' seconds \n')

file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()

env.state_to_log() # checks environment state
