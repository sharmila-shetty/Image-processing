#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import os


# In[2]:


os.getcwd()


# In[3]:


# lists all the files and folders in the current working directory
os.listdir()


# In[4]:


for f in os.listdir("."):
    if f.endswith(".jpg"):
        i = Image.open(f)
        fn, fext = os.path.splitext(f)
        print(fn, "&", fext)


# In[5]:


# Creating new Directory using OS library
os.mkdir('NewExtnsn')
# Note: If you already have a directory with this name, you will get error.


# In[6]:


for f in os.listdir("."):
    if f.endswith(".jpg"):
        i = Image.open(f)
        fn, fext = os.path.splitext(f)
        i.save("NewExtnsn/{}.pdf".format(fn))


# In[7]:


# Creating new multiple Directories using OS library
os.makedirs('resize//small')
os.makedirs('resize//tiny')
# Note: If you already have a directory with this name, you will get error.


# In[8]:


size_small = (600,600) # small images of 600 X 600 pixels
size_tiny = (200,200)  # tiny images of 200 X 200 pixels
for f in os.listdir("."):
    if f.endswith(".jpg"):
        i = Image.open(f)
        fn, fext = os.path.splitext(f)
        i.thumbnail(size_small)
        i.save("resize/small/{}_small{}".format(fn, fext))
        i.thumbnail(size_tiny)
        i.save("resize/tiny/{}_tiny{}".format(fn, fext))


# In[9]:


# Creating new Directory using OS library
os.mkdir('rotate')


# In[10]:


for f in os.listdir("."):
    if f.endswith(".jpg"):
        i = Image.open(f)
        fn, fext = os.path.splitext(f)
        im = i.rotate(90)
        im.save("rotate/{}_rot.{}".format(fn, fext))


# In[ ]:




