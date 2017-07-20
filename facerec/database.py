
# coding: utf-8

# In[42]:

# Imports
import pickle
import numpy as np


# In[43]:

# Constants
DATABASE_FR = "data/facial_features.txt"

# Variables
database = {}


# In[44]:

# Retrieve DB
def retrieve_database(filepath=DATABASE_FR):
    with open(filepath, "rb") as f:
        db = pickle.load(f)
    return db


# In[45]:

# Write DB
def write_database(filepath=DATABASE_FR):
    with open(filepath, "wb") as f:
        pickle.dump(database, f)


# In[46]:

# Add image to database
def add_image(descriptor,name=None):
    
    if name != None:
        
        
        old_descriptor_list = list(database.get(name))[0]
        
        old_descriptor_list.append(descriptor)
        
        new_list = old_descriptor_list
        
        num_descriptors = len(new_list)
        
        temp_arr = np.array(new_list)
        
        new_mean = np.sum(temp_arr)/num_descriptors
        
    
        database[name] = [new_list,new_mean]
    

    if name == None:
        the_name = input("Please enter your name: ")
        
        the_descriptors = []
        the_descriptors.append(descriptor)
        
        database[the_name] = [the_descriptors]
        
        mean_val = descriptor
        
        database[the_name].append(mean_val)
        


# In[47]:

def add_multiple_images(num_detections, descriptors, names):
    
    
    for i in range(num_detections):
        
        add_image(descriptors[i],names[i])
    


# In[48]:

def clear_database(password):
    if password.lower() == "Yes I am sure":
        if input("Are you very sure?").lower() == "y":
            global database
            database = {}


# In[51]:

# Start
def initialize():
    global database
    database = retrieve_database()

initialize()


# In[ ]:



