from PIL import Image
import random as rand
import numpy as np

array = np.zeros([100, 100, 3], dtype=np.uint8)

# fill_color = [[255, 0, 0], [0, 255, 0], [255, 255, 255]]
fill_color = [[255, 0, 0], [0, 255, 0], [0, 0, 0]]

# array.fill(255)
for i in range(100):
    for j in range(100):
        array[i,j] =  fill_color[rand.randint(0,2)]


img = Image.fromarray(array)
img.save('testrgb.png')