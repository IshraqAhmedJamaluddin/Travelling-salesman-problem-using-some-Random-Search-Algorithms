import random
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

class Tour:     # Chromosome
    def __init__(self, genes, fitness=0):
        self.cities = genes
        self.fitness = fitness
    
    def __repr__(self):
        return f'<Tour:\tfitness: {self.fitness},\tcities list:\n{self.cities}>'

class City:     # Gene
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f'<City:\tid: {self.id},\tx: {self.x},\ty: {self.y}>'

def model(filename):
    df = pd.read_csv(filename)
    size = df.shape[0]
    cities = []
    for i in range(size):
        city = City(df['City'][i], df['x'][i], df['y'][i])
        cities.append(city)
    return cities

def distance(city1, city2):
    return sqrt((city1.x - city2.x) ** 2 + (city1.y - city2.y) ** 2)
 
def calculate_fitness(cities):
    cost = 0
    for i in range(len(cities)):
        city1 = cities[i]
        city2 = cities[(i+1)%len(cities)]
        cost += distance(city1, city2)
    return 1/cost

def generate_random_tour(cities):
    genes = np.array(random.sample(cities, len(cities)))
    fitness = calculate_fitness(genes)
    tour = Tour(genes, fitness)
    return tour
 
def generate_initial_population(pop_size, cities):
    population = []
    for i in range(pop_size):
        population.append(generate_random_tour(cities))
    return population

def elitism(elitism_percentage, pop):
    elite = sorted(pop, key=lambda tour: tour.fitness, reverse=True)
    return elite[:int(elitism_percentage*len(elite))]

def k_tournament(pop, k):
    tours = random.sample(pop, k)
    fittest = sorted(tours, key=lambda tour: tour.fitness, reverse=True)
    return fittest[0]

def crossover(parent1, parent2):
    o1_cities, o2_cities = PMC(np.array(parent1.cities), np.array(parent2.cities))

    fitness1 = calculate_fitness(o1_cities)
    tour1 = Tour(o1_cities, fitness1)
    fitness2 = calculate_fitness(o2_cities)
    tour2 = Tour(o2_cities, fitness2)

    if tour1.fitness + tour2.fitness >= parent1.fitness + parent2.fitness:
        return tour1, tour2
    return parent1, parent2

def PMC(parent1, parent2):
    cutoff_1, cutoff_2 = np.sort(np.random.choice(np.arange(len(parent1)+1), size=2, replace=False))

    def PMX_one_offspring(p1, p2):
        offspring = np.zeros(len(p1), dtype=p1.dtype)

        # Copy the mapping section (middle) from parent1
        offspring[cutoff_1:cutoff_2] = p1[cutoff_1:cutoff_2]

        # copy the rest from parent2 (provided it's not already there
        for i in np.concatenate([np.arange(0,cutoff_1), np.arange(cutoff_2,len(p1))]):
            candidate = p2[i]
            while candidate in p1[cutoff_1:cutoff_2]: # allows for several successive mappings
                # print(f"Candidate {candidate} not valid in position {i}") # DEBUGONLY
                candidate = p2[np.where(p1 == candidate)[0][0]]
            offspring[i] = candidate
        return offspring

    offspring1 = PMX_one_offspring(parent1, parent2)
    offspring2 = PMX_one_offspring(parent2, parent1)

    return offspring1, offspring2

def mutation(tour):
    old_genes = tour.cities
    genes = old_genes.copy()
    cutoff_1, cutoff_2 = np.sort(np.random.choice(np.arange(len(genes)), size=2, replace=False))
    genes[cutoff_1], genes[cutoff_2] = genes[cutoff_2], genes[cutoff_1]
    fitness2 = calculate_fitness(genes)
    tour2 = Tour(genes, fitness2)
    return tour2

def search(population_size=50, generations_Count=100, elitism_percentage=.2, crossover_probability=0.6, mutation_probability=0.1):
    filename = '15-Points.csv' # 'dj38.tsp'
    nodes = model(filename)
    k = 5

    pop = generate_initial_population(population_size, nodes)
    
    newP = elitism(elitism_percentage, pop)

    for gen in range(generations_Count):

        for i in range((population_size-len(newP))//2):
            crossover_luck = np.random.random()
            mutation_luck = np.random.random()
            parent_1 = k_tournament(pop, k)
            parent_2 = k_tournament(pop, k)
            if crossover_luck <= crossover_probability:
                parent_1, parent_2 = crossover(parent_1, parent_2)
            if mutation_luck <= mutation_probability:
                parent_1 = mutation(parent_1)
                parent_2 = mutation(parent_2)
            newP.append(parent_1)
            newP.append(parent_2)

        pop = newP
        newP = elitism(elitism_percentage, pop)

    return newP[0].cities, newP[0].fitness



def search_with_new_points(nodes, population_size=50, generations_Count=100, elitism_percentage=.2, crossover_probability=0.6, mutation_probability=0.1):
    k = 5

    pop = generate_initial_population(population_size, nodes)
    
    newP = elitism(elitism_percentage, pop)

    for gen in range(generations_Count):

        for i in range((population_size-len(newP))//2):
            crossover_luck = np.random.random()
            mutation_luck = np.random.random()
            parent_1 = k_tournament(pop, k)
            parent_2 = k_tournament(pop, k)
            if crossover_luck <= crossover_probability:
                parent_1, parent_2 = crossover(parent_1, parent_2)
            if mutation_luck <= mutation_probability:
                parent_1 = mutation(parent_1)
                parent_2 = mutation(parent_2)
            newP.append(parent_1)
            newP.append(parent_2)

        pop = newP
        newP = elitism(elitism_percentage, pop)

    return newP[0].cities, newP[0].fitness


def on_press(event):
    sys.stdout.flush()
    if event.key == 'enter':
        fig.canvas.mpl_disconnect(cid)
        finished()

def onclick(event):
    global i
    city = City(i, event.xdata, event.ydata)
    i += 1
    cities.append(city)

    plt.plot(event.xdata, event.ydata, 'ro')
    fig.canvas.draw()

def finished():
    # population_size = 50
    # generations_Count = 100
    # elitism_percentage = .2 # 2% of population (mean two chromosomes)
    # crossover_probability = 0.6
    # mutation_probability = 0.1

    citiesList, fitness = search_with_new_points(cities)
    print('The list of cities in order:', *citiesList, sep='\n')
    print()
    print('The cost of the trip', round(1/fitness, 2))

    x = list(map(lambda city: city.x, citiesList))
    y = list(map(lambda city: city.y, citiesList))

    for i in range(len(x)):
        plt.arrow(x[i], y[i], x[(i+1)%len(x)]-x[i], y[(i+1)%len(y)]-y[i], width=.7, head_width=5*.7, head_length= 5*.7, length_includes_head=True, edgecolor='pink', facecolor='pink', zorder=3)
        plt.text(x[i]+5, y[i], str(i+1), color="purple", fontsize=12, zorder=4)
        fig.canvas.draw()
        plt.pause(.5)
    plt.text(3, 90, f'The cost is {round(1/fitness, 2)}', fontsize = 15)
    fig.canvas.draw()


i = 0
cities = []
fig = plt.figure()
fig.canvas.set_window_title('Random Search Optimization - Genetic Algorithm')
plt.title('Genetic Algorithm')
ax = fig.add_subplot(111)
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])

cid = fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', on_press)
plt.show()
