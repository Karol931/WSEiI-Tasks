import numpy as np
from datetime import datetime, timedelta
import random

class GeneticAlgorithm:

    def __init__(self, n, tol):
        self.n = n
        self.tol = tol
        self.error_counter = 0
        self.range = [0, 2*np.pi]  
        self.population = [[random.uniform(0, 2*np.pi), random.uniform(0, 2*np.pi)]  for _ in range(n)]
        self.old_max_value = 0

    def get_value(self, x, y):
        return np.abs(np.sin(x) + np.sin(2*x) + np.sin(4*x) + np.cos(y) + np.cos(2*y) + np.cos(4*y))


    def select_top_individuals(self):
        values = [(individual, self.get_value(individual[0], individual[1])) for individual in self.population]
        
        # Sortuj według wartości malejąco
        ranked_population = sorted(values, key=lambda x: x[1], reverse=True)
        
        # Wybierz n najlepszych
        top_individuals = [x[0] for x in ranked_population[:self.n]]

        self.population = top_individuals
        return self.get_value(top_individuals[0][0], top_individuals[0][1])
        

    def mutate(self):
        i = random.randint(0, len(self.population) - 1)
        j = random.randint(0,1)
        flag = random.randint(0,1)
        if flag == 1:
            self.population[i][j] = self.population[i][j] * 1.001
        else:
            self.population[i][j] = self.population[i][j] * 0.999
        
        if self.population[i][j] > self.range[1] or self.population[i][j] < self.range[0]:
                del self.population[i]

    def check_max_value(self, new_max_value):
        if self.old_max_value >= new_max_value:
            self.error_counter += 1
        else:
            self.error_counter = 0
            self.old_max_value = new_max_value
        return self.old_max_value, new_max_value
        
    def crossing(self):
        children = [None] * 100
        for i in range(0, 100, 2):
            children[i] = [self.population[i][0], self.population[i+1][1]]
            children[i+1] = [self.population[i][1], self.population[i+1][0]]
        self.population.extend(children)

    def run(self):
        start_time = datetime.now()
        
        while True:  
            self.crossing()
            self.mutate()
            new_max_value = self.select_top_individuals()
            self.check_max_value(new_max_value)

            if self.error_counter == self.tol:
                break
        
        end_time = datetime.now()
        self.solution_time = end_time-start_time


def print_result(tol, val, solution_time):
        print(f'Maximal value for {tol} errors: {val}')
        print(f'Solution time for {tol} errors: {solution_time}')


N = 100
POPULATION_NUMBER = 100
TOL = [1000, 10000, 100000]

solution_times = [None] * 100
for tol in TOL:
    genetic_algorithm = GeneticAlgorithm(
    n=POPULATION_NUMBER,
    tol=TOL[0])


    genetic_algorithm.run()
    genetic_algorithm.solution_time
    genetic_algorithm.select_top_individuals()
            
    print_result(tol, genetic_algorithm.select_top_individuals(), genetic_algorithm.solution_time)
        

    

