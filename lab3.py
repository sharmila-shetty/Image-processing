#!/usr/bin/env python
# coding: utf-8

# In[1]:


#bitwise operation
import cv2
img1=cv2.imread("dog.jpg")
img2=cv2.imread("rabbit.jpg")

bit_and = cv2.bitwise_and(img1,img2)
cv2.imshow('AND',bit_and)

bit_or = cv2.bitwise_or(img1,img2)
cv2.imshow('OR',bit_or)

bit_not = cv2.bitwise_not(img1)
cv2.imshow('NOT',bit_not)

bit_xor = cv2.bitwise_xor(img1,img2)
cv2.imshow('XOR',bit_xor)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[2]:


from PIL import Image, ImageDraw, ImageFilter

im1 = Image.open('red.jpg')
im2 = Image.open('p4.png')

back_im = im1.copy()
back_im.paste(im2, (100,50))

back_im.show()


# In[3]:


#MEDIAN FILTERING
import cv2
import numpy as np
img_noisy1=cv2.imread("filter.png",0)
m,n=img_noisy1.shape
img_new1=np.zeros([m,n])
for i in range(1,m-1):
    for j in range(1,n-1):
        temp=[img_noisy1[i-1,j-1],img_noisy1[i-1,j],img_noisy1[i-1,j-1],img_noisy1[i-1,j+1],img_noisy1[i,j-1],img_noisy1[i,j],img_noisy1[i,j+1],img_noisy1[i+1,j-1],img_noisy1[i+1,j],img_noisy1[i+1,j+1]]
        temp=sorted(temp)
        img_new1[i,j]=temp[4]
        img_new1=img_new1.astype(np.uint8)
cv2.imshow("MEDIAN FILTERED IMAGE",img_new1)
cv2.waitKey(0)  
cv2.destroyAllWindows()    


# In[ ]:




