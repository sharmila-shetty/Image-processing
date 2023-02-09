#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("fp.png",0)
epsilon=0.001
diff_threshold=100

mean_data=np.mean(img)
img.shape


# In[2]:


plt.title('original Image')
plt.imshow(img,cmap='gray')
plt.show()


# In[3]:


r=270
c=186
while diff_threshold>epsilon:
    data_one=[]
    data_two=[]
    for i in range(r):
        for j in range(c):
            if img[i,j]<mean_data:
                data_one.append(img[i,j])
            else:
                data_two.append(img[i,j])
    print(data_one)
    print(data_two)
    mu_one=np.mean(data_one)
    mu_two=np.mean(data_two)
    avg_mean=(mu_one+mu_two)/2
    diff_threshold=abs(mean_data-avg_mean)
    mean_data=avg_mean
   


# In[4]:


print(mean_data)


# In[5]:


import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('noisy_leaf.jpg', cv2.IMREAD_GRAYSCALE)  
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.title('Original Image')
plt.imshow(img,cmap='gray')
plt.axis('off')

blur = cv2.GaussianBlur(img,(7,7),0)
plt.figure(figsize=(10,10))
plt.subplot(2,2,2)
plt.title('Gaussian Blur')
plt.imshow(blur,cmap='gray')
plt.axis('off')

x,threshold = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY) 
plt.figure(figsize=(10,10))
plt.subplot(2,2,3)
plt.title('Binary Threshold')
plt.imshow(threshold ,cmap='gray')
plt.axis('off')

ret2,th2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.title('Otsus Thresholding')
plt.imshow(th2,cmap='gray')
plt.axis('off')

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




