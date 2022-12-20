#!/usr/bin/env python
# coding: utf-8

# In[40]:


from PIL import Image,ImageDraw,ImageFilter
im1=Image.open('ant.jpg')
im2=Image.open('earth.jpg')
mask_im=Image.new("L",im2.size,0)
draw=ImageDraw.Draw(mask_im)
draw.ellipse((690,240,1370,900),fill=225)
mask_im_blur=mask_im.filter(ImageFilter.GaussianBlur(10))
back_im=im1.copy()
back_im.paste(im2,(0,0),mask_im_blur)
back_im.show()


# In[46]:


#Upsampling

import cv2

image = cv2.imread('red.jpg')
cv2.imshow('image before pyrUp:',image)
image = cv2.pyrUp(image)
cv2.imshow('image after pyrUp:',image)
cv2.imshow('UpSample', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[47]:


#Downsampling

import cv2
image = cv2.imread('red.jpg')
cv2.imshow('image before pyrDown:',image)
image = cv2.pyrDown(image)
cv2.imshow('image after pyrDown:',image)
cv2.imshow('UpSample', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




