#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
img=cv2.imread('red-roses.jpg')
cv2.imshow('To display image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


cv2.imwrite('C:/Users/User/Desktop/img.jpg',img)


# In[4]:


import cv2
img=cv2.imread('red-roses.jpg',0)
cv2.imshow('To display image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[5]:


print(img.shape)


# In[6]:


print(img.size)


# In[2]:


import cv2
img=cv2.imread("D:/blue flower.jpg")
cv2.imshow('To display image from drive',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[10]:


from PIL import Image

#read the image
im = Image.open("blue flower.jpg")

#image size
width = im.size[0]
height = im.size[1]

print('Width  of the image is:', width)
print('Height of the image is:', height)


# In[6]:


#No. of channels from color image
import numpy

img=cv2.imread("blue flower.jpg")
print('No of Channels is:'+str(img.ndim))
print('No of Channels is:',img.shape)
cv2.imshow("Channel",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[7]:


#No. of channels from grey-scale image
import numpy

img=cv2.imread("blue flower.jpg",0)
print('No of Channels is:'+str(img.ndim))
print('No of Channels is:',img.shape)

cv2.imshow("Channel",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


#Resize images
from PIL import Image
filepath="blue flower.jpg"
im=Image.open(filepath)
new_im = im.resize((200, 200))
new_im


# In[4]:


#matrix representation of image

import matplotlib.image as image
img=image.imread('blue flower.jpg')
print('The shape of the image is:',img.shape)
print('The image as array is:')
print(img)


# In[10]:


#binary image
import cv2
img=cv2.imread('flower 1.jpg',0)
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)

#converting to its binary form
bw=cv2.threshold(img,127,255,cv2.THRESH_BINARY)

cv2.imshow("Binary",bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[11]:


import cv2
img=cv2.imread('flower 1.jpg')
B,G,R=cv2.split(img)
print(B)
print(G)
print(R)


# In[19]:


#blue channel
cv2.imshow("Blue",B)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[20]:


#green channel
cv2.imshow("Green",G)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[22]:


# Red channel
cv2.imshow("Red",R)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[36]:


im = cv2.imread("flower 1.jpg")
new_im = im.resize((400, 300))
ar = 1.0 * (im.shape[1]/im.shape[0])
print("aspect ratio: ")
print(ar)


# In[39]:


#mirror an image
from PIL import Image
filepath="blue flower.jpg"
im=Image.open(filepath)
hori_flippedImage=im.transpose(Image.FLIP_LEFT_RIGHT)
im.show()
hori_flippedImage.show()

#vertical
vert_flippedImage=im.transpose(Image.FLIP_TOP_BOTTOM)
vert_flippedImage.show()


# In[2]:


#diffrent color space

import cv2
im=cv2.imread("red-roses.jpg")
gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB)
hls=cv2.cvtColor(im,cv2.COLOR_BGR2HLS)
yuv=cv2.cvtColor(im,cv2.COLOR_BGR2YUV)
rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

cv2.imshow('converted Gray Color:',gray)
cv2.imshow('converted hsv Color:',hsv)
cv2.imshow('converted lab Color:',lab)
cv2.imshow('converted hls Color:',hls)
cv2.imshow('converted yuv Color:',yuv)
cv2.imshow('converted rgb Color:',rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[4]:


import cv2
img1=cv2.imread("dog.jpg")
img2=cv2.imread("rabbit.jpg")
sum=img1+img2
sub=img1-img2
mul=img1*img2
div=img1/img2



cv2.imshow('Sum:',sum)
cv2.imshow('Sub:',sub)
cv2.imshow('Mul:',mul)
cv2.imshow('Div:',div)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




