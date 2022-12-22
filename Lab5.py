#!/usr/bin/env python
# coding: utf-8

# In[38]:


# Low Pass SPatial Domain Filtering
# to observe the blurring effect


import cv2
import numpy as np


# Read the image
img = cv2.imread('filter.png', 0)

# Obtain number of rows and columns
# of the image
m,n = img.shape

# Develop Averaging filter(3, 3) mask
mask = np.ones([3, 3], dtype = int)
mask = mask / 9

# Convolve the 3X3 mask over the image
img_new = np.zeros([m, n])

for i in range(1, m-1):
    for j in range(1, n-1):
        temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i-1, j + 1]*mask[0, 2]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]+img[i, j + 1]*mask[1, 2]+img[i + 1, j-1]*mask[2, 0]+img[i + 1, j]*mask[2, 1]+img[i + 1, j + 1]*mask[2, 2]

        img_new[i, j]= temp

img_new = img_new.astype(np.uint8)
cv2.imshow('blurred', img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[50]:


# Importing Image module from PIL package
from PIL import Image
import PIL

# creating a image object (main image)
im1 = Image.open("dog.jpg")

# quantize a image
im1 = im1.quantize(15)

# to show specified image
im1.show()


# In[49]:


import cv2
import numpy as np

#nearest neighbor interpolation
img = cv2.imread('flower 1.jpg')
near_img = cv2.resize(img,None, fx = 25, fy = 25, interpolation = cv2.INTER_NEAREST)
cv2.imshow('Near',near_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#bi-linear interpolation
import cv2
import numpy as np
img = cv2.imread('flower 1.jpg')
bilinear_img = cv2.resize(img,None, fx = 25, fy = 25, interpolation = cv2.INTER_LINEAR)
cv2.imshow('Bilinear',bilinear_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#bi-cubic interpolation
import cv2
import numpy as np
img = cv2.imread('flower 1.jpg')
bicubic_img = cv2.resize(img,None, fx = 25, fy = 25, interpolation = cv2.INTER_CUBIC)
cv2.imshow('Bicubic',bicubic_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




