#BFS
from collections import deque

def bfs(graph, start, goal):
    visited = []
    queue = deque([start])
    visited.append(start)
    came_from = {start: None}

    while queue:
        node = queue.popleft()
        print(f"Visiting: {node}")

        if node == goal:
            # Reconstruct path
            path = []
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            print(f"Path found: {path}")
            return path

        for neighbour in graph[node]:
            if neighbour not in visited:
                visited.append(neighbour)
                came_from[neighbour] = node
                queue.append(neighbour)

    print("Goal not found")
    return None

# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
bfs(graph, 'A', 'F')

#DLS

def dls(graph, start, goal, depth_limit):
    def dfs(node, depth, path):
        print(f"Visiting: {node} at depth {depth}")
        if node == goal:
            print(f"Goal found! Path: {path}")
            return path
        if depth >= depth_limit:
            return None
        for neighbor in graph.get(node, []):
            if neighbor not in path:
                result = dfs(neighbor, depth + 1, path + [neighbor])
                if result:
                    return result
        return None

    result = dfs(start, 0, [start])
    if not result:
        print("Goal not found within depth limit")
    return result

# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
dls(graph, 'A', 'F', depth_limit=3)

#A*

def a_star(graph, heuristic, start, goal):
    # frontier stores (node, f_cost)
    frontier = [(start, heuristic[start])]
    visited = set()
    g_costs = {start: 0}
    came_from = {start: None}

    print(f"Open List: {frontier}")

    while frontier:
        # Sort by f(n) and pick lowest
        frontier.sort(key=lambda x: x[1])
        current_node, current_f = frontier.pop(0)

        if current_node in visited:
            continue

        visited.add(current_node)
        print(f"Closed List: {visited}")

        if current_node == goal:
            # Reconstruct optimal path
            path = []
            node = current_node
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            print(f"Optimal Path: {path}")
            return path

        for neighbor, cost in graph[current_node].items():
            new_g = g_costs[current_node] + cost
            f_cost = new_g + heuristic[neighbor]

            if neighbor not in g_costs or new_g < g_costs[neighbor]:
                g_costs[neighbor] = new_g
                came_from[neighbor] = current_node
                frontier.append((neighbor, f_cost))
                print(f"Open List: {frontier}")

    print("Goal not found")
    return None

# Example usage
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'C': 2, 'D': 5},
    'C': {'D': 1},
    'D': {}
}
heuristic = {'A': 7, 'B': 6, 'C': 2, 'D': 0}
a_star(graph, heuristic, 'A', 'D')

#Goal Based Agent

class Environment:
    def __init__(self, graph):
        self.graph = graph

    def get_percept(self, node):
        return node

class GoalBasedAgent:
    def __init__(self, goal):
        self.goal = goal

    def act(self, percept, graph):
        print(f"Agent at: {percept}, Goal: {self.goal}")
        result = bfs(graph, percept, self.goal)  # or dls / a_star
        return result

def run_agent(agent, environment, start_node):
    percept = environment.get_percept(start_node)
    action = agent.act(percept, environment.graph)
    print(f"Action taken: {action}")

# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
env = Environment(graph)
agent = GoalBasedAgent('F')
run_agent(agent, env, 'A')

#roulete

import random

def roulette_wheel_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    pick = random.uniform(0, total_fitness)
    current = 0
    for individual, fitness in zip(population, fitness_scores):
        current += fitness
        if current >= pick:
            return individual

# Example usage
population = ['A', 'B', 'C', 'D']
fitness_scores = [10, 20, 30, 40]
selected = roulette_wheel_selection(population, fitness_scores)
print(f"Selected: {selected}")

#Tournament 

import random

def tournament_selection(population, fitness_scores, k=3):
    selected = random.sample(list(zip(population, fitness_scores)), k)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]

# Example usage
population = ['A', 'B', 'C', 'D', 'E']
fitness_scores = [10, 50, 30, 80, 20]
selected = tournament_selection(population, fitness_scores, k=3)
print(f"Selected: {selected}")

#Single point Crossover
import random

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 2)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    print(f"Crossover point: {point}")
    print(f"Child 1: {child1}")
    print(f"Child 2: {child2}")
    return child1, child2

# Example usage
parent1 = [1, 0, 1, 1, 0]
parent2 = [0, 1, 0, 0, 1]
crossover(parent1, parent2)

#Swap Mutation

import random

def mutate(individual, mutation_rate=0.1):
    individual = individual[:]  # copy to avoid modifying original
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        print(f"Mutated at indices {idx1} and {idx2}")
    return individual

# Example usage
individual = [1, 0, 1, 1, 0]
result = mutate(individual, mutation_rate=1.0)  # rate=1.0 forces mutation
print(f"Mutated individual: {result}")

#Fitness Function

def calculate_fitness(individual):
    penalty = 0
    # Example: penalize consecutive 1s
    for i in range(len(individual) - 1):
        if individual[i] == 1 and individual[i+1] == 1:
            penalty += 1
    fitness = 1 / (1 + penalty)  # higher fitness = fewer penalties
    return fitness

# Example usage
individual = [1, 0, 1, 1, 0]
print(f"Fitness: {calculate_fitness(individual)}")
