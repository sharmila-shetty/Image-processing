#!/usr/bin/env python
# coding: utf-8

# In[1]:


#contouring shapes
import matplotlib.pyplot as plt
def show_image_contour(image,contours):
    plt.figure()
    for n,contour in enumerate(contours):
        plt.plot(contour[:,1],contour[:,0],linewidth=3)
    plt.imshow(image,interpolation='nearest',cmap='gray_r')
    plt.title('contours')
    plt.axis('off')


# In[2]:


from skimage import measure,data

horse_image=data.horse()

contours=measure.find_contours(horse_image,level=0.8)

show_image_contour(horse_image,contours)


# In[20]:


#Find contours of an image that is not binary
from skimage import color
from skimage.io import imread
from skimage.filters import threshold_otsu

image_dices = imread('dice.jpg')

# Make the image grayscale
image_dices = color.rgb2gray(image_dices)

# Obtain the optimal thresh value
thresh = threshold_otsu(image_dices)

# Apply thresholding
binary = image_dices > thresh

# Find contours at a constant value of 0.8
contours = measure.find_contours(binary, level=0.8)

# Show the image
show_image_contour(image_dices, contours)


# In[22]:


#Count the dots in a dice's image
import numpy as np
shape_contours = [cnt.shape[0] for cnt in contours]

# Set 50 as the maximum size of the dots shape
max_dots_shape = 200

# Count dots in contours excluding bigger than dots size
dots_contours = [cnt for cnt in contours if np.shape(cnt)[0] < max_dots_shape]

# Shows all contours found
show_image_contour(binary, contours)

# Print the dice's number
print('Dice`s dots number: {}.'.format(len(dots_contours)))


# In[ ]:




