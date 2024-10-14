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