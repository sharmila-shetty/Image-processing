#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Image sharpness

from PIL import Image
from PIL import ImageFilter 
import matplotlib.pyplot as plt

my_image=Image.open("p6.jpg")
sharp=my_image.filter(ImageFilter.SHARPEN)
sharp.save("D:/image_sharpen.jpg")
sharp.show()
plt.imshow(sharp)
plt.show()


# In[3]:


#Image flipping

import matplotlib.pyplot as plt

img=Image.open("p6.jpg")
plt.imshow(img)
plt.show()

flip=img.transpose(Image.FLIP_LEFT_RIGHT)
sharp.save("D:/image_flip.jpg")
plt.imshow(flip)
plt.show()


# In[5]:


#Cropping image

from PIL import Image
from PIL import ImageFilter 

img=Image.open("p6.jpg")
width,height=img.size
im1=img.crop((600,200,2200,2000))

im1.show()
plt.imshow(im1)
plt.show()


# In[ ]:




