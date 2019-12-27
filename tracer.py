from PIL import Image
import numpy as np

WIDTH, HEIGHT = 300, 200

foo = np.zeros((HEIGHT, WIDTH, 3))
foo[:, :, 0] = np.linspace(0, 255, WIDTH)
foo[:, :, 1] = np.linspace(255, 0, HEIGHT).reshape(-1, 1)
foo[:, :, 2] = 0.2

im = Image.fromarray(foo.astype(np.uint8))
im.save("output.png")
