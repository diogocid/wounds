import numpy as np
from PIL import Image

img = Image.open(r'C:\Users\diogo\Documents\Isec\projeto\MedT Model\github\wounds\valid\labelcol\0671_png_jpg.rf.ac7982e30eeb6c1f7f9325d9187ffc1f.png').convert('L')
arr = np.array(img)

# record the original shape
shape = arr.shape

# make a 1-dimensional view of arr
flat_arr = arr.ravel()

# convert it to a matrix
vector = np.matrix(flat_arr)

# do something to the vector
vector[:,::10] = 128

# reform a numpy array of the original shape
arr2 = np.asarray(vector).reshape(shape)

# make a PIL image
img2 = Image.fromarray(arr2, 'RGBA')
img2.show()