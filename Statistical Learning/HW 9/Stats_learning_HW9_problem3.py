#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import math
pre_v = np.zeros((4,4))
actions = [np.array([1,0]),np.array([-1,0]),np.array([0,1]),np.array([0,-1])]
delta = 0.0001
prob = 0.25
err = delta+1
while err>=delta:
    err = 0
    new_v = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            if (i==0 and j==0) or (i==3 and j==3):
                continue
            old_v_s = pre_v[i,j]
            for action in actions:
                row_index = i+action[0]
                column_index = j+action[1]
                if row_index<0:
                    row_index = 0
                if row_index>3:
                    row_index = 3
                if column_index<0:
                    column_index = 0
                if column_index>3:
                    column_index = 3
                new_v[i,j]+= prob*(-1+pre_v[row_index,column_index])
            err = max([err,abs(old_v_s-new_v[i,j])])
    pre_v= new_v
for i in range(4):
    for j in range(4):
        pre_v[i,j] = math.floor(pre_v[i,j])


# In[25]:


pre_v
print (new_v)

# In[ ]:




