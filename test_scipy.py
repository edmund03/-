# Python script using Scipy
# for image manipulation

from imageio import imwrite,imread
import numpy as np
from PIL import Image

# Read a JPEG image into a numpy array
img = imread('cat.jpg')  # path of the image
print(img.dtype, img.shape)

# Tinting the image
img_tint = img * [1, 0.45, 0.3]

# Saving the tinted image
imwrite('cat_tinted.jpg', img_tint)

# Resizing the tinted image to be 300 x 300 pixels
img_tint_resize = np.array(Image.fromarray(np.uint8(img_tint)).resize((300,300)))

# Saving the resized tinted image
imwrite('cat_tinted_resized.jpg', img_tint_resize)