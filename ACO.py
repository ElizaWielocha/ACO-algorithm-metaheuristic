from datetime import timedelta, datetime
import math
import numpy as np
import sys
import random

# Początkowe parametry -------------------------------------------------------------------------------------------

print("METAHEURISTIC - ANT COLONY ALGORITHM")
print("Select parameters")
print("if you don't enter some parameters, they will have default values\n")

algorithm_time = input("Enter the running time of the algorithm (in seconds): ")
if algorithm_time == '':
  algorithm_time = 300
else:
  algorithm_time = int(algorithm_time)

ant_number = input("Enter the number of ants: ")
if ant_number == '':
  ant_number = 100
else:
  ant_number = int(ant_number)

alfa = input("Enter the value of alfa parameter: ")
if alfa == '':
  alfa = 1
else:
  alfa = int(alfa)

beta = input("Enter the value of beta parameter: ")
if beta == '':
  beta = 0
else:
  beta = int(beta)

x = input("Enter the x parameter that is needed to calculate the cost for path (more than 5): ")
if x == '':
  x = 5
else:
  x = int(x)

v_number = 100
min_edge_number = 6
max_edge_number = 30
min_weight = 1
max_weight = 100
solution_number = 1
d_pheromone_vaping = 0.1
pheromone_use_chance = 0.01
increase_pheromone = 0.001
smooth_limit = 35
smooth = 20

# generowanie instancji -----------------------------------------------------------------------------------------

def Instance_generate():
  # generowanie losowego grafu, który jest wypełniany wagami za krawedzie
  graph = np.zeros((v_number, v_number), dtype=int)
  for i in range(1, v_number):
    weight = random.randint(min_weight, max_weight)
    graph[i-1][i] = weight
    graph[i][i-1] = weight

  # sprwdzenie czy minminum i maksimum krawedzi w grafie sie zgadza - jesli nie, to wartosci w grafie sa poprawiane
  for i in range(v_number):
    filling = min_edge_number - len(graph[i].nonzero()[0])
    if filling > 0:
      for _ in range(filling):
        other = set(range(v_number)) - set(graph[i].nonzero()[0]) - {i}
        while other:
          j = random.choice(tuple(other))
          if len(graph[:,j].nonzero()[0]) < max_edge_number:
            weight = random.randint(min_weight, max_weight)
            graph[i][j] = weight
            graph[j][i] = weight
            break
          other.remove(j)
        else:
          sys.exit("Cannot create graph")

  # Tworzenie grafów feromonów i prawdopodobieństwa
  pheromone_matrix = np.zeros((v_number, v_number), dtype=float)
  pheromone_matrix[graph.nonzero()] = 1

  probability_matrix = np.copy(pheromone_matrix)

  return graph, pheromone_matrix, probability_matrix

# Wyliczanie kosztu za ścieżke -----------------------------------------------------------------------------------
# suma wszystkich wag odwiedzonych do tej pory krawędzi
# co x odwiedzonych łuków licząc od startu do aktualnej sumy S dodawana jest suma ostatnich x/2 odwiedzonych wag 
# pomnożona przez podwójny stopień wierzchołka, w którym znajduje się algorytm przeszukiwania po przejściu x krawędzi. 

def cost_for_path(graph, path):
  weights = np.empty(path.size - 1, dtype=int)
  for i in range(0, path.size - 1):
    weights[i] = graph[path[i], path[i + 1]]
    if (i + 1) % x == 0:
      undo = math.ceil(x/2)
      weights[i] = weights[i] + 2 * weights[i+1-undo:i+1].sum() * len(graph[path[i + 1]].nonzero()[0])
  return weights.sum()


# Puszczanie mrówki ----------------------------------------------------------------------------------------------

def ant_go(graph, pheromones, probabilities):
  path = np.zeros(v_number, dtype=int)

  # Dla każdej mrówki generowane jest losowo miejsce, z którego ma rozpocząć wędrówkę.
  index = 0
  max_index = v_number - 1
  path[0] = random.randrange(0, v_number)

  pheromone_use = random.random() < pheromone_use_chance # mrowka decyduje czy skorzystac z feromonow, czy nie

  missed = set(range(v_number)) - {path[0]} #nieodwiedzone wierzcholki mrowki
  while missed: 
    neighbors = graph[path[index]].nonzero()[0]

    missed_neighbors = tuple(missed & set(neighbors)) 
    if missed_neighbors: # Jeżeli istnieją nieodwiedzeni sąsiedzi to zawsze wybieramy któryś z nich
      if pheromone_use: #jeśli istnieja nieodwiedzeni neighbors i jest jakies uzycie feromonow
        missed_prob = np.zeros(v_number)
        missed_prob[missed_neighbors, ] = probabilities[path[index], missed_neighbors] # Kopiujemy prawdopodobieństwa tylko nieodwiedzonych wierzchołków

        next = random.choices(list(range(0, v_number)), weights=missed_prob)[0] # nieodwiedzony wierzchołek na podstawie ich prawdopodobieństw
      else:
        next = random.choice(missed_neighbors) # nieodwiedzony wierzchołek z równymi szansami
    else:
      if pheromone_use: 
        next = random.choices(list(range(0, v_number)), weights=probabilities[path[index]])[0] #jakikolwiek sąsiedni wierzchołek na podstawie prawdopodobieństw
      else:
        next = random.choice(neighbors) #jakikolwiek sąsiedni wierzchołek z równymi szansami

    if next in missed: # usuwany juz odwiedzony wierzcholek z nieodwiedzonych
      missed.remove(next)
    index += 1
    if index > max_index:
      path.resize(max_index + 11)
      max_index += 10
    path[index] = next

  path.resize(index + 1)
  return cost_for_path(graph, path), path



# Puszczanie algorytmu -----------------------------------------------------------------------------------------------

if __name__ == "__main__":
  # generujemy trzy macierze: grafu, feromonow i prawdopodobienstwa
  graph_matrix, pheromone_matrix, probability_matrix = Instance_generate()

  best_solution = None
  stop = datetime.now() + timedelta(seconds=algorithm_time)
  while datetime.now() < stop:
    paths = []
    # puszczamy mrówki - każda mrówka robi jedna sciezke i dodajemy je do listy 'paths'
    for index in range(ant_number):
      paths.append(ant_go(graph_matrix, pheromone_matrix, probability_matrix))

    paths = sorted(paths, key=lambda x: x[0]) # Sortujemy paths po koszcie
    best_paths = paths[:solution_number] 


    # Aktualizacja feromonów, dodajemy wartości pomiędzy 0.1-1
    difference = best_paths[-1][0] - best_paths[0][0]
    for cost, path in best_paths:
      if difference:
        pheromone_power = (cost - best_paths[0][0]) / difference * 0.9
      else:
        pheromone_power = 1

      for j in range(0, len(path) - 1):
        k, l = path[j:j+2]
        pheromone_matrix[k, l] += 1 - pheromone_power


    # Parowanie feromonów -
    pheromone_matrix *= 1 - d_pheromone_vaping


    # Aktualizacja prawdopodobieństw
    probability_matrix = (pheromone_matrix ** alfa)
    probability_matrix *= (graph_matrix / 1) ** beta


    # Wygładzanie
    for line in pheromone_matrix:
      if np.where(line > smooth_limit)[0].size:
        minimum = line[line.nonzero()].min()
        for i, x in enumerate(line):
          if x > 0:
            line[i] = minimum * (1 + math.log(x / minimum, smooth))


    pheromone_use_chance += increase_pheromone

    if not best_solution or paths[0][0] < best_solution:
      best_solution = paths[0][0]
      print(f'Path with the lowest cost: {best_solution} at time: {datetime.now().hour}:{datetime.now().minute}:{datetime.now().second}')


print('Parameters choosen by user:')
print(f'Running time of the algorithm: {algorithm_time} | number of ants: {ant_number} | alfa: {alfa} | beta: {beta} | x parametr: {x}')
print(f'Cost for the best path is {best_solution}.')