#!/usr/bin/env python
# coding: utf-8

# In[5]:


# import the modules
import os
from os import listdir

# get the path/directory
folder_dir = "C:/Users/User/Desktop/images"
for images in os.listdir(folder_dir):

	# check if the image ends with png
	if (images.endswith(".png")):
		print(images)


# In[9]:


# import the modules
import os
from os import listdir

# get the path/directory
folder_dir = "C:/Users/User/Desktop/images"
for images in os.listdir(folder_dir):
		print(images)


# In[38]:


#import necessary packages 
import cv2 
import os 
import glob 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
 
#Set the path where images are stored 
img_dir = "C:/Users/User/Desktop/images" # Enter Directory of all images  
data_path = os.path.join(img_dir,'*g') 
files = glob.glob(data_path) 
data = [] 
for f1 in files: 
    img = cv2.imread(f1) 
    data.append(img) 
    plt.figure() 
    plt.imshow(img) 


# In[ ]:





# In[ ]:




