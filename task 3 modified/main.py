import matplotlib.pyplot as plt
import sys
import math
import random


class City:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f'<City:\tid: {self.id},\tx: {self.x},\ty: {self.y}>'

class Graph(object):
    def __init__(self, cost_matrix: list, rank: int):
        self.matrix = cost_matrix
        self.rank = rank
        self.pheromone = [[1 / (rank * rank) for j in range(rank)] for i in range(rank)]


class ACO(object):
    def __init__(self, ant_count: int, generations: int, alpha: float, beta: float, rho: float, q: int,
                 strategy: int):
        self.Q = q
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.ant_count = ant_count
        self.generations = generations
        self.update_strategy = strategy

    def _update_pheromone(self, graph: Graph, ants: list):
        for i, row in enumerate(graph.pheromone):
            for j, col in enumerate(row):
                graph.pheromone[i][j] *= self.rho
                for ant in ants:
                    graph.pheromone[i][j] += ant.pheromone_delta[i][j]

    def solve(self, graph: Graph):
        best_cost = float('inf')
        best_solution = []
        for _ in range(self.generations):
            ants = [_Ant(self, graph) for i in range(self.ant_count)]
            for ant in ants:
                for i in range(graph.rank - 1):
                    ant._select_next()
                ant.total_cost += graph.matrix[ant.tabu[-1]][ant.tabu[0]]
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_solution = [] + ant.tabu
                ant._update_pheromone_delta()
            self._update_pheromone(graph, ants)
        return best_solution, best_cost


class _Ant(object):
    def __init__(self, aco: ACO, graph: Graph):
        self.colony = aco
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []
        self.pheromone_delta = []
        self.allowed = [i for i in range(graph.rank)]
        self.eta = [[0 if i == j else 1 / graph.matrix[i][j] for j in range(graph.rank)] for i in
                    range(graph.rank)]
        start = random.randint(0, graph.rank - 1)
        self.tabu.append(start)
        self.current = start
        self.allowed.remove(start)

    def _select_next(self):
        denominator = 0
        for i in self.allowed:
            denominator += self.graph.pheromone[self.current][i] ** self.colony.alpha * self.eta[self.current][i] ** self.colony.beta
        probabilities = [0 for i in range(self.graph.rank)]
        for i in range(self.graph.rank):
            try:
                self.allowed.index(i)
                probabilities[i] = self.graph.pheromone[self.current][i] ** self.colony.alpha * \
                    self.eta[self.current][i] ** self.colony.beta / denominator
            except ValueError:
                pass
        selected = 0
        rand = random.random()
        for i, probability in enumerate(probabilities):
            rand -= probability
            if rand <= 0:
                selected = i
                break
        self.allowed.remove(selected)
        self.tabu.append(selected)
        self.total_cost += self.graph.matrix[self.current][selected]
        self.current = selected

    def _update_pheromone_delta(self):
        self.pheromone_delta = [[0 for j in range(self.graph.rank)] for i in range(self.graph.rank)]
        for _ in range(1, len(self.tabu)):
            i = self.tabu[_ - 1]
            j = self.tabu[_]
            if self.colony.update_strategy == 1:
                self.pheromone_delta[i][j] = self.colony.Q
            elif self.colony.update_strategy == 2:
                self.pheromone_delta[i][j] = self.colony.Q / self.graph.matrix[i][j]
            else:
                self.pheromone_delta[i][j] = self.colony.Q / self.total_cost


def on_press(event):
    sys.stdout.flush()
    if event.key == 'enter':
        fig.canvas.mpl_disconnect(cid)
        finished()

def onclick(event):
    global i
    cities.append(dict(index=i, x=event.xdata, y=event.ydata))
    points.append((event.xdata, event.ydata))
    i += 1

    plt.plot(event.xdata, event.ydata, 'ro')
    fig.canvas.draw()

def finished():

    cost_matrix = []
    rank = len(cities)
    for i in range(rank):
        row = []
        for j in range(rank):
            row.append(distance(cities[i], cities[j]))
        cost_matrix.append(row)

    aco = ACO(10, 100, 1.0, 10.0, 0.5, 10, 2)
    graph = Graph(cost_matrix, rank)

    oldCitiesList, cost = aco.solve(graph)

    citiesList = []
    for city in oldCitiesList:
        newCity = City(city, points[city][0], points[city][1])
        citiesList.append(newCity)

    print('The list of cities in order:', *citiesList, sep='\n')
    print()
    print('The cost of the trip', round(cost, 2))

    x = list(map(lambda city: city.x, citiesList))
    y = list(map(lambda city: city.y, citiesList))

    for i in range(len(x)):
        plt.arrow(x[i], y[i], x[(i+1)%len(x)]-x[i], y[(i+1)%len(y)]-y[i], width=.7, head_width=5*.7, head_length= 5*.7, length_includes_head=True, edgecolor='pink', facecolor='pink', zorder=3)
        plt.text(x[i]+5, y[i], str(i+1), color="purple", fontsize=12, zorder=4)
        fig.canvas.draw()
        plt.pause(.5)
    plt.text(3, 90, f'The cost is {round(cost, 2)}', fontsize = 15)
    fig.canvas.draw()



def distance(city1: dict, city2: dict):
    return math.sqrt((city1['x'] - city2['x']) ** 2 + (city1['y'] - city2['y']) ** 2)


i = 0
cities = []
points = []
fig = plt.figure()
fig.canvas.set_window_title('Random Search Optimization - ACO Algorithm')
plt.title('ACO Algorithm')
ax = fig.add_subplot(111)
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])

cid = fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', on_press)
plt.show()
