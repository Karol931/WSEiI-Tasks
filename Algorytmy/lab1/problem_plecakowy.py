import numpy as np
from datetime import datetime, timedelta
class GeneticAlgorithm:

    def __init__(self, n, backpack_v, tol, items_weight_min, items_weight_max):
        self.n = n
        self.backpack_v = backpack_v
        self.tol = tol
        self.error_counter = 0
        while True:
            self.items_v = np.random.randint(items_weight_min, items_weight_max+1, size=self.n)

            if np.sum(self.items_v) > self.backpack_v:
                break
        
        self.parent = np.random.randint(0,2, size=self.n)
        self.solution_time = 0


    def get_epsilon(self, child_or_parent):

        return self.backpack_v - np.dot(child_or_parent, self.items_v)


    def is_child_better(self, child):
        parent_epsilon = self.get_epsilon(self.parent)
        child_epsilon = self.get_epsilon(child)

        if parent_epsilon < 0 and child_epsilon > 0:
            return True 
        if parent_epsilon < 0 and child_epsilon < 0 and parent_epsilon < child_epsilon:
            return True
        if parent_epsilon > 0 and child_epsilon > 0 and parent_epsilon > child_epsilon:
            return True
        
        return False


    def mutate(self):
        child = self.parent.copy()
        i = np.random.randint(0,self.n)
        child[i] = abs(1 - child[i])
        if self.is_child_better(child):
            self.parent = child
            self.error_counter = 0
        else:
            self.error_counter += 1
        

    def run(self):
        start_time = datetime.now()
        
        while True:  
            
            self.mutate()

            if self.error_counter == self.tol or self.get_epsilon(self.parent) == 0:
                break
        
        end_time = datetime.now()
        self.solution_time = end_time-start_time


    def print_result(self):
        print(f'Final Epsilon: {self.get_epsilon(self.parent)}')
        print(f'Solution took: {self.solution_time}')


def calculate_average_epsilon(epsilon_solutions):

    return np.average(epsilon_solutions)


def calculate_average_time(solution_times):

    return sum(solution_times, timedelta(0)) / len(solution_times)


def print_result(epsilon_solutions, solution_times, tol):
        print(f'Average Epsilon for {tol} errors: {calculate_average_epsilon(epsilon_solutions)}')
        print(f'Average Solution time for {tol} errors: {calculate_average_time(solution_times)}')

N = 100
ITEMS_NUMBER = 100
TOL = [1000, 10000, 100000]
BACKPACK_V = 2500
ITEMS_WEIGHTS_RANGE = [10,90]
epsilon_solutions = [None] * 100
solution_times = [None] * 100
for tol in TOL:
    for i in range(N):
        genetic_algorithm = GeneticAlgorithm(
        n=ITEMS_NUMBER,
        backpack_v=BACKPACK_V, 
        tol=tol, 
        items_weight_min=ITEMS_WEIGHTS_RANGE[0], 
        items_weight_max=ITEMS_WEIGHTS_RANGE[1])

        genetic_algorithm.run()
        epsilon_solutions[i] = genetic_algorithm.get_epsilon(genetic_algorithm.parent)
        solution_times[i] = genetic_algorithm.solution_time

    print_result(epsilon_solutions, solution_times, tol)
        

    

