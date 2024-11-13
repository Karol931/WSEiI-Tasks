import numpy as np
from datetime import datetime, timedelta
class GeneticAlgorithm:

    def __init__(self, n, procesor_t, tol, items_time_min, items_time_max):
        self.n = n
        self.procesor_t = procesor_t
        self.tol = tol
        self.error_counter = 0
        self.items_t = np.random.randint(items_time_min, items_time_max+1, size=self.n)
        self.parent = np.random.randint(0,4, size=self.n)


    def get_max_time(self, child_or_parent):
        times = [0,0,0,0]
        for i, val in enumerate(child_or_parent):
            times[val] += self.items_t[i] * self.procesor_t[val]

        return np.max(times)


    def is_child_better(self, child):
        if self.get_max_time(child) < self.get_max_time(self.parent):
            return True
        
        return False


    def mutate(self):
        child = self.parent.copy()
        i = np.random.randint(0,self.n)
        rand = np.random.randint(0,4)
        child[i] = rand
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
        print(f'Final max time: {self.get_max_time(self.parent)}')
        print(f'Solution took: {self.solution_time}')


def calculate_average_max_time(max_proces_time):

    return np.average(max_proces_time)


def calculate_average_solution_time(solution_times):

    return sum(solution_times, timedelta(0)) / len(solution_times)


def print_result(tol, max_procesor_solution_times, solution_times):
        print(f'Average maximal time for {tol} errors: {calculate_average_max_time(max_procesor_solution_times)}')
        print(f'Average solution time {tol} errors: {calculate_average_solution_time(solution_times)}')


N = 100
ITEMS_NUMBER = 100
TOL = [1000, 10000, 100000]
PROCESOR_TIMES = [1,1.25,1.5,1.75]
ITEMS_TIMES_RANGE = [10,90]
max_procesor_solution_times = [None] * 100
solution_times = [None] * 100
for tol in TOL:
    for i in range(N):
        genetic_algorithm = GeneticAlgorithm(
        n=ITEMS_NUMBER,
        procesor_t=PROCESOR_TIMES, 
        tol=TOL[0], 
        items_time_min=ITEMS_TIMES_RANGE[0], 
        items_time_max=ITEMS_TIMES_RANGE[1])

        genetic_algorithm.run()
        # genetic_algorithm.print_result()
        solution_times[i] = genetic_algorithm.solution_time
        max_procesor_solution_times[i] = genetic_algorithm.get_max_time(genetic_algorithm.parent)
            
    print_result(tol, max_procesor_solution_times, solution_times)
        

    

