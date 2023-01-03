#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Image restoration
import numpy as np
import cv2

# Open the image.
img = cv2.imread('cat_damaged.png')

# Load the mask.
mask = cv2.imread('cat_mask.png', 0)

# Inpaint.
dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

cv2.imshow("cat",img)
cv2.imshow("cat1",mask)
cv2.imshow("cat2",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




