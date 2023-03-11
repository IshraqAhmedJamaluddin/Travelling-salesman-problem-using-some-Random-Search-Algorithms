import math, pandas as pd
import sys
import matplotlib.pyplot as plt

class City:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.visited = False
    
    def __repr__(self):
        return f'<City:\tid: {self.id},\tx: {self.x},\ty: {self.y},\tvisited: {self.visited}>'


def model(filename):
    df = pd.read_csv(filename)
    cities = []
    for i in range(df.shape[0]):
        city = City(df['City'][i], df['x'][i], df['y'][i])
        cities.append(city)
    return cities


def euclidean_distance(A, B):
    return math.sqrt((A.x-B.x)**2 + (A.y-B.y)**2)


def generate_distance_matrix(nodes):
    matrix = {}
    for i in range(len(nodes)):
        cityA = nodes[i]
        matrix[cityA] = {}
        for j in range(len(nodes)):
            cityB = nodes[j]
            if cityA is cityB:
                # dist = float('inf')
                pass
            else:
                dist = euclidean_distance(cityA, cityB)
                matrix[cityA][cityB] = dist     # indented
    return matrix


def choose_nearest_point(city, matrix):
    row = matrix[city]
    closestCity = min(row.keys(), key=lambda x: row[x])
    while closestCity.visited == True:
        row[closestCity] = float('inf')
        closestCity = min(row.keys(), key=lambda x: row[x])
    closestCity.visited = True
    return closestCity, row[closestCity]


def should_terminate(cities):
    return len(list(filter(lambda x: not x.visited, cities))) == 0


def select_starting_point(cities):
    return cities[0]
    # return random.choice(cities)


def search():
    filename = '15-Points.csv' # 'dj38.tsp'
    nodes = model(filename)
    matrix = generate_distance_matrix(nodes)

    city = select_starting_point(nodes)
    city.visited = True
    citiesList = [city]
    cost = 0

    while not should_terminate(nodes):
        city, currentCost = choose_nearest_point(city, matrix)
        city.visited = True
        citiesList.append(city)
        cost += currentCost
    
    cost += matrix[citiesList[-1]][citiesList[0]]

    return citiesList, cost


def search_with_new_points(nodes):
    matrix = generate_distance_matrix(nodes)

    city = select_starting_point(nodes)
    city.visited = True
    citiesList = [city]
    cost = 0

    while not should_terminate(nodes):
        city, currentCost = choose_nearest_point(city, matrix)
        city.visited = True
        citiesList.append(city)
        cost += currentCost
    
    cost += matrix[citiesList[-1]][citiesList[0]]

    return citiesList, cost


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
    citiesList, cost = search_with_new_points(cities)
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
    # plt.scatter(x, y, c='red', s=100)
    # fig.canvas.draw()


i = 0
cities = []
fig = plt.figure()
fig.canvas.set_window_title('Random Search Optimization - Nearest Neighbour Algorithm')
plt.title('Nearest Neighbour Algorithm')
ax = fig.add_subplot(111)
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])

cid = fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', on_press)
plt.show()
