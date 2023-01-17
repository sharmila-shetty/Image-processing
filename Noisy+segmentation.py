#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Adding Noisy
from skimage.util import random_noise
import matplotlib.pyplot as plt
image=plt.imread("blue.jpeg")

noisy_image = random_noise(image)

plt.title("original Image")
plt.imshow(image)
plt.show()

plt.title("Noisy Image")
plt.imshow(noisy_image)
plt.show()


# In[3]:


#Removing Noisy
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt
image=plt.imread("filter.png")

denoised_image = denoise_tv_chambolle(image,multichannel=True)

plt.title("original Image")
plt.imshow(image)
plt.show()

plt.title("Denoised Image")
plt.imshow(denoised_image)
plt.show()


# In[5]:


#Reducing noise while preserving edges
from skimage.restoration import denoise_bilateral

landscape_image=plt.imread('filter.png')

denoised_image = denoise_bilateral(landscape_image,multichannel=True)

plt.title("original Image")
plt.imshow(landscape_image)
plt.show()

plt.title("Denoised Image")
plt.imshow(denoised_image)
plt.show()


# In[2]:


#segmentation
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import label2rgb
face_image=plt.imread('blue.jpeg')

segments=slic(face_image,n_segments=400)

segmented_image=label2rgb(segments,face_image,kind='avg')

plt.title("original Image")
plt.imshow(face_image.astype('uint8'))
plt.show()

plt.title("Segmented image")
plt.imshow(segmented_image.astype('uint8'))
plt.show()


# In[ ]:




