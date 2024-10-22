import matplotlib.pyplot as plt
import numpy as np

def plot_solution(result_list: list, coordinates: np.ndarray, weights: np.ndarray, title: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.title(title)
    for i1, i2 in zip(result_list, result_list[1:] + result_list[:1]):
        point1 = coordinates[i1]
        point2 = coordinates[i2]
        plt.plot(
            [point1[0], point2[0]],
            [point1[1], point2[1]],
            c = 'black',
            zorder=1
        )

    scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=weights, cmap='inferno', s=50)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Weights')
   
    plt.show()

def score(path: list, distance: np.ndarray, costs: np.ndarray) -> int:
    score=0
    for x in range(1,len(path)):
        score+=distance[path[x-1]][path[x]]+costs[path[x]]
   
    score+=distance[path[-1]][path[0]]+costs[path[0]]
    return int(score[0])

def greedy_2_regret_weighted(starting_point: int, distance_matrix: np.ndarray, costs: np.ndarray, weight_regret=0.5) -> list:
    cost_matrix = (distance_matrix + costs).T 
    n = len(distance_matrix)
    num_nodes_to_use = n // 2


    cycle = np.array([starting_point])  
    unvisited = np.ones(n, dtype=bool) 
    unvisited[starting_point] = False

 
    unvisited_indices = np.where(unvisited)[0]  
    nearest = unvisited_indices[np.argmin(cost_matrix[starting_point, unvisited_indices])] 
    cycle = np.append(cycle, nearest) 
    unvisited[nearest] = False  
    unvisited_indices = np.where(unvisited)[0]
    increases = (
        cost_matrix[cycle[0], unvisited_indices[:, None]] +  
        cost_matrix[unvisited_indices[:, None], cycle[1]]    
    )
    min_idx = np.argmin(increases)
    best_insertion = (unvisited_indices[min_idx], 0)  
    i, j = best_insertion
    cycle = np.insert(cycle, j + 2, i)  
    unvisited[i] = False  
    while len(cycle) < num_nodes_to_use:
        cycle_len = len(cycle)
        unvisited_indices = np.where(unvisited)[0]
        j_indices = np.arange(cycle_len)
        k_indices = (j_indices + 1) % cycle_len  
        cost_increases = (
            cost_matrix[cycle[j_indices], unvisited_indices[:, None]] +  
            cost_matrix[unvisited_indices[:, None], cycle[k_indices]] - 
            cost_matrix[cycle[j_indices], cycle[k_indices]]
        )
        best_increases = np.min(cost_increases, axis=1)  
        second_best_increases = np.partition(cost_increases, 1, axis=1)[:, 1] 
        
        weighted= best_increases - weight_regret*second_best_increases

        max_weighted_idx = np.argmin(weighted)
        i = unvisited_indices[max_weighted_idx]
        best_position = np.argmin(cost_increases[max_weighted_idx])
        cycle = np.insert(cycle, (best_position + 1) % cycle_len, i) 
        unvisited[i] = False  
    return cycle.tolist()