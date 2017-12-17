
# coding: utf-8

# In[1]:


import os
import re
from glob import glob
import pandas as pd
import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt


# In[2]:


listValid = []
def read_audio(path):
    amount = int(len(os.listdir(path)))
    amount = amount / 10
    if amount < 1:
        amount = 1
    root = os.path.split(path)[-1]
    
    for file in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file))
        
        if os.path.isdir(abs_path): 
            temp = os.path.split(abs_path)[-1]
            #if temp != "_background_noise_":
            read_audio(abs_path)
        elif os.path.isfile(abs_path) and file.endswith('.wav') and amount != 0: 
            amount -= 1
            temp = os.path.split(abs_path)[-1]
            listValid.append(root + "/" + temp + " ")
            
    return listValid
    
def creat_valid_list():
    listValid = read_audio('./data/train/audio')  
    #np.savetxt('validation_list.txt', listValid, delimiter=' ', fmt="%s,%s,%s")
    np.savetxt('./data/validation_list.txt', listValid, fmt="%s")


# In[3]:


creat_valid_list()

