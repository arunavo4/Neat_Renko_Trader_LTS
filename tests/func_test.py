from PIL import Image
import numpy as np
from math import ceil


def generate_color_graph(renko_directions, obs_window, gap_window=0):
    renko_graph_directions = [float(i) for i in renko_directions]

    renko_graph_directions = renko_graph_directions[-(obs_window - gap_window):]

    color_graph = np.zeros([obs_window, obs_window, 3], dtype=np.uint8)

    fill_color = [[255, 0, 0], [0, 255, 0], [255, 255, 255]]

    i = init_i = ceil((color_graph.shape[0] / 2))

    values_of_i = []

    for j in range(1, len(renko_graph_directions)):
        i = i - 1 if renko_graph_directions[j] == 1 else i + 1
        values_of_i.append(i)

    spread_of_i = max(values_of_i) - min(values_of_i)

    # To compensate for zero {init_i - (min(values_of_i) + 1)}
    dist_btw_min_i = init_i - min(values_of_i) if min(values_of_i) > 0 else init_i - (min(values_of_i) + 1)

    i = int((color_graph.shape[0] - spread_of_i) / 2) + dist_btw_min_i
    color_graph[i, 0] = fill_color[1] if renko_graph_directions[0] == 1 else fill_color[0]

    for j in range(1, len(renko_graph_directions)):
        i = i - 1 if renko_graph_directions[j] == 1 else i + 1
        color_graph[i, j] = fill_color[1] if renko_graph_directions[j] == 1 else fill_color[0]

    return color_graph


if __name__ == "__main__":
    renko_dirs = ['1']*100
    array = generate_color_graph(renko_dirs, obs_window=31)
    img = Image.fromarray(array)
    img.save('testrgb.png')