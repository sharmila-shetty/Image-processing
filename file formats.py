#!/usr/bin/env python
# coding: utf-8

# In[3]:


#converting one file format to another
from PIL import Image
im=Image.open("dog.jpg")
print(im.mode)
im.save("D:/dog.png")


# In[55]:


#image manipulation with numpy array slicing
import cv2
from PIL import Image
import numpy as np

im = np.array(Image.open('rabbit.jpg').resize((256, 256)))

im1 = im[:, :100]
im2 = im[:, 100:]

cv2.imshow('To display image',im1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('To display image',im2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[10]:


from PIL import Image


# Take two images for blending them together   
image1 = Image.open("dog.jpg")
image2 = Image.open("rabbit.jpg")

# alpha-blend the images with varying values of alpha
alphaBlended1 = Image.blend(image5, image6, alpha=.2)
alphaBlended2 = Image.blend(image5, image6, alpha=.4)

# Display the alpha-blended images
alphaBlended1.show()
alphaBlended2.show()


# In[12]:


#croping an image
from PIL import Image
img = Image.open('rabbit.jpg')
img.show()
box = (250, 250, 750, 750)
img2 = img.crop(box)
img2.show()


# In[14]:


#Negating an image
import cv2
import numpy as np
# Load the image
img = cv2.imread('blue flower.jpg')
# Check the datatype of the image
print(img.dtype)
# Subtract the img from max value(calculated from dtype)
img_neg = 255 - img
# Show the image
cv2.imshow('negative',img_neg)
cv2.waitKey(0)


# In[53]:


import cv2
image = cv2.imread('rabbit.jpg')

new_img=cv2.putText(
img=image,
text="Image Processing",
org=(200,200),
fontFace=cv2.FONT_HERSHEY_DUPLEX,
fontScale=4.0,
color=(155,246,55),
thickness=3
)

cv2.imshow('New Image',new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[39]:


# Importing Image and ImageDraw from PIL
from PIL import Image, ImageDraw

# Opening the image to be used
img = Image.open('rabbit.jpg')

# Creating a Draw object
draw = ImageDraw.Draw(img)

# Drawing a green rectangle
# in the middle of the image
draw.rectangle(xy = (150, 150, 450, 450),
fill = (0, 127, 0))

# Method to display the modified image
img.show()


# In[54]:


#Import required libraries
import sys
from PIL import Image, ImageDraw

#Create Image object
im = Image.open("rabbit.jpg")

#Draw line
draw = ImageDraw.Draw(im)
draw.line((0, 0) + im.size, fill=128)
draw.line((0, im.size[1], im.size[0], 0), fill=128)

#Show image
im.show()


# In[44]:


img = cv2.imread('rabbit.jpg')
pl=im.histogram()
plt.bar(range(256),pl[:256],color='r',alpha=0.5)
plt.bar(range(256),pl[256:2*256],color='g',alpha=0.4)
plt.bar(range(256),pl[2*256:],color='b',alpha=0.3)
plt.show()


# In[46]:


from PIL import Image,ImageStat
im = Image.open("rabbit.jpg")
stat=ImageStat.Stat(im)
print(stat.mean)


# In[47]:


from PIL import Image,ImageStat
im = Image.open("rabbit.jpg")
stat=ImageStat.Stat(im)
print(stat.median)


# In[48]:


from PIL import Image,ImageStat
im = Image.open("rabbit.jpg")
stat=ImageStat.Stat(im)
print(stat.stddev)


# In[56]:


img = np.array(Image.open('dog.jpg'))
img_R, img_G, img_B = img.copy(), img.copy(), img.copy()
img_R[:, :, (1, 2)] = 0
img_G[:, :, (0, 2)] = 0
img_B[:, :, (0, 1)] = 0
img_rgb = np.concatenate((img_R,img_G,img_B), axis=1)
plt.figure(figsize=(15, 15))
plt.imshow(img_rgb)


# In[ ]:





# In[ ]:




