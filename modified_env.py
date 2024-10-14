
class modified_environment(Environment):


    def cons_multi_fitness(self,values):
        """
        Function used by Environment.multiple(). 
        Prioritizes the least competitive fitness value by a set amount (weight_priority).
        Deprioritizes other fitness values by the same amount downscaled equally betweewn them.

        param values: fitness values reflecting the algorithm's quality against different enemies 
        """
        num_enemies = len(values)
        default_weight = 0.3333333
        weights = [default_weight] * num_enemies 

        weight_priority = 0.05 # relative change 0.38 vs 0.25, 0.25 
        weight_depriority = weight_priority / num_enemies

        values = [(i + 1, fitness) for i, fitness in enumerate(values)]       
        min_enemy, min_fitness = min(values, key=lambda x: x[1])
        self.min_enemy = min_enemy

        for i in range(values): 
            if values[i] == min_enemy: 
                weights[i] = default_weight + weight_priority 
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

        vfitness = self.cons_multi_fitness(np.array(vfitness))
        vplayerlife = self.cons_multi(np.array(vplayerlife))
        venemylife = self.cons_multi(np.array(venemylife))
        vtime = self.cons_multi(np.array(vtime))
        self.min_enemy 

        return  vfitness, vplayerlife, venemylife, vtime
