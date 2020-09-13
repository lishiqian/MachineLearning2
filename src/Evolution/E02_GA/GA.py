import numpy as np

TARGET_STR = "You get it!lalalalaala"

DNA_SIZE = len(TARGET_STR)  # DNA length DNA长度
POP_SIZE = 3000  # population size 种群人员数量
CROSS_RATE = 0.4  # mating probability (DNA crossover) DNA交叉配对概率
MUTATION_RATE = 0.001  # mutation probability 基因变异概率
N_GENERATIONS = 100
ASCII_BOUND = [32, 126]  # x upper and lower bounds X轴区间范围


TARGET_ASCII = np.fromstring(TARGET_STR, dtype=np.uint8)  # convert string to number


class GA(object):

    def __init__(self, DNA_size, pop_size, cross_rate, mutation_rate, DNA_bound):
        self.DNA_size = DNA_size
        self.pop_size = pop_size
        self.DNA_bound = DNA_bound
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate

        self.pop = np.random.randint(*DNA_bound, size=(POP_SIZE, DNA_SIZE)).astype(np.uint8)

    def get_fitness(self):
        match_count = (self.pop == TARGET_ASCII).sum(axis=1)
        return match_count

    def translate_DNA(self, DNA):
        return DNA.tostring().decode('ascii')

    def select(self):
        fitness = self.get_fitness() + 1e-4  # add a small amount to avoid all zero fitness
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)  # select another individual from pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)  # choose crossover points
            parent[cross_points] = pop[i_, cross_points]  # mating and produce one child
        return parent


    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                child[point] = np.random.randint(*self.DNA_bound)  # choose a random ASCII index
        return child

    def evolve(self):
        pop = self.select()
        pop_copy = pop.copy()
        for parent in pop:
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop




if __name__ == '__main__':
    ga = GA(DNA_size=DNA_SIZE, DNA_bound=ASCII_BOUND, cross_rate=CROSS_RATE,
            mutation_rate=MUTATION_RATE, pop_size=POP_SIZE)

    for generation in range(N_GENERATIONS):
        fitness = ga.get_fitness()
        best_DNA = ga.pop[np.argmax(fitness)]
        best_phrase = ga.translate_DNA(best_DNA)
        print('Gen', generation, ': ', best_phrase)
        if best_phrase == TARGET_STR:
            break
        ga.evolve()