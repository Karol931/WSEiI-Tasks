import numpy as np
from datetime import datetime, timedelta

class GeneticAlgorithm:

    def __init__(self, n, tol, lengths_min, lengths_max):
        self.n = n
        self.tol = tol
        self.error_counter = 0
        self.cities_l = np.random.randint(lengths_min, lengths_max+1, size=(self.n, self.n))
        self.cities_l = [[0 if i == j else self.cities_l[i][j] for j in range(self.n)] for i in range(self.n)]
        self.parent = np.arange(0,100,1)
        np.random.shuffle(self.parent)


    def get_length(self, child_or_parent):
        lengths = [self.cities_l[child_or_parent[i]][child_or_parent[i+1]] for i in range(self.n - 1)]
        return np.sum(lengths)


    def is_child_better(self, child):
        if self.get_length(child) < self.get_length(self.parent):
            return True
        
        return False


    def mutate(self):
        child = self.parent.copy()
        i, j = [np.random.randint(0,self.n)] * 2
        tmp = child[i]
        child[i] = child[j]
        child[j] = child[i]
        
        if self.is_child_better(child):
            self.parent = child
            self.error_counter = 0
        else:
            self.error_counter += 1
        

    def run(self):
        start_time = datetime.now()
        
        while True:  
            
            self.mutate()

            if self.error_counter == self.tol:
                break
        
        end_time = datetime.now()
        self.solution_time = end_time-start_time


    def print_result(self):
        print(f'Final min length: {self.get_length(self.parent)}')
        print(f'Solution took: {self.solution_time}')


def calculate_average_min_length(lengths):

    return np.average(lengths)


def calculate_average_solution_time(solution_times):

    return sum(solution_times, timedelta(0)) / len(solution_times)


def print_result(tol, lengths, solution_times):
        print(f'Average minimal length for {tol} errors: {calculate_average_min_length(lengths)}')
        print(f'Average solution time {tol} errors: {calculate_average_solution_time(solution_times)}')


N = 100
CITIES_NUMBER = 100
TOL = [1000, 10000, 100000]
LENGTHS_RANGE = [10,100]
min_lengths = [None] * 100
solution_times = [None] * 100
for tol in TOL:
    for i in range(N):
        genetic_algorithm = GeneticAlgorithm(
        n=CITIES_NUMBER,
        tol=TOL[0], 
        lengths_min=LENGTHS_RANGE[0], 
        lengths_max=LENGTHS_RANGE[1])


        genetic_algorithm.run()
        solution_times[i] = genetic_algorithm.solution_time
        min_lengths[i] = genetic_algorithm.get_length(genetic_algorithm.parent)
            
    print_result(tol, min_lengths, solution_times)
        

    

