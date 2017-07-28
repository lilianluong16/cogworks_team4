# coding: utf-8

# In[109]:

import nltk, pickle, os, numpy as np, news_loader
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict
from os import path, makedirs

DATABASE_FR = "data/entities_database.txt"

# In[110]:

def extract_single(tokens, by_type=False):
    """
    Gets the named entities in the list of tokens
    
    Parameters:
    -----------
    tokens: list(str)
        a tokenized document containing named entities
    by_type: bool, optional
        whether or not differentiate between types of named entities
        ***True currently not supported***
        
    Returns:
    --------
    out: list(str)
        a list of the named entities in the document
    """
    entities = []
    pos = nltk.pos_tag(tokens) # label parts of speech
    named_entities = nltk.ne_chunk(pos, binary=not by_type) # identify named entities
    for i in range(0, len(named_entities)):
        ents = named_entities.pop()
        if getattr(ents, 'label', None) != None and ents.label() == "NE": 
            entities.append(([ne for ne in ents]))
    extracted = np.array(entities)
    if extracted.ndim == 3:
        final = extracted[:,:,0].tolist()
    else:
        final = []
        for entity in extracted:
            final.append(np.array(entity)[:,0].tolist())
    out = []
    for entity in final:
        entity = " ".join(entity)
        out.append(entity)
    return out


# In[111]:

def create_database(filepath=DATABASE_FR):
    """
    Initializes the global variable db with the database stored in the specfied filepath
    
    Parameters:
    -----------
    filepath: str (optional)
        the directory and filename of the location of the database pickled txt file
        
    Returns:
    --------
        db: dict (str --> Counter)
            the database mapping a named entity to the Counter 
            of co-occurences with other entities in the database
    """
    global db
    db = retrieve_database(filepath=filepath)
    return db


# In[112]:

def update_single(extracted):
    """
    Updates the database with the provided named entities in a single document
    
    Parameters:
    -----------
    extracted: list(str)
        a list of the named entities in the document
    
    Returns:
    --------
    db: dict (str --> Counter)
            the database mapping a named entity to the Counter 
            of co-occurences with other entities in the database
    """
    global db
    counts = Counter(extracted)
    for entity in extracted:
        if entity in db:
            db[entity.lower()].update(counts)
            for key, value in counts.items():
                if not key == entity:
                    db[entity.lower()][key] = value * db[entity.lower()][entity]
        else:
            db[entity.lower()] = Counter(extracted)
            for key, value in counts.items():
                if not key == entity:
                    db[entity.lower()][key] = value * db[entity.lower()][entity]
        del db[entity.lower()][entity]
    return db


# In[114]:

def new_database(filepath=DATABASE_FR):
    """
    Creates a new text file and folder in the filepath 

    Parameters:
    -----------
    filepath: str (optional)
        the directory and filename of the location of the database pickled txt file 
    """
    if not os.path.exists(filepath):
        os.makedirs(str.partition(filepath, "/")[0])


# In[115]:

def write_database(filepath=DATABASE_FR):
    """
    Saves the database to the text file in the specified file path

    Parameters:
    -----------
    filepath: str (optional)
        the directory and filename of the location of the database pickled txt file 
    """
    with open(filepath, "wb") as f:
        pickle.dump(db, f)


# In[116]:

def retrieve_database(filepath=DATABASE_FR):
    """
    Retrieves the database from the text file in the specified file path

    Parameters:
    -----------
    filepath: str (optional)
        the directory and filename of the location of the database pickled txt file
        
    Returns:
    --------
    db: dict (str --> Counter)
            the database mapping a named entity to the Counter 
            of co-occurences with other entities in the database
    """
    with open(filepath, "rb") as f:
        db = pickle.load(f)
    return db


# In[117]:

def clear_database(filepath=DATABASE_FR):
    """
    Removes all entries in the database in the text file in the specified file path

    Parameters:
    -----------
    filepath: str (optional)
        the directory and filename of the location of the database pickled txt file
    """
    global db
    db = defaultdict(Counter)
    with open(filepath, "wb") as f:
        pickle.dump(db, f)


# In[118]:

def top_related(entity, k=None):
    """
    Finds the k entities most closely related to the query (an entity itself)
    
    Parameters:
    -----------
    entity: str
        the entity which the results will be related to 
    k: int (optional)
        the number of entities to be returned (default all matches)
        
    Returns:
    --------
    out: list(tuple(str, int))
        the k entities most closely related to the query and their count
    """
    global db
    if entity not in db:
        return None
    out = db[entity].most_common(k)
    return out


# In[119]:

def update(tokenlists):
    """
    Updates the database with the provided named entities in multiple documents
    
    Parameters:
    -----------
    tokenlists: list(list(str))
        a list of the named entities in the documents
    
    Returns:
    --------
     db: dict (str --> Counter)
            the database mapping a named entity to the Counter 
            of co-occurences with other entities in the database
    """
    for doc in tokenlists:
        update_single(extract_single(doc))
    return db


# In[120]:

def common_entities(tokenlists, k=None):
    """
    Given a list of token-lists, return the k most common entities among them
    
    Parameters:
    -----------
    tokenlists: list(list(str))
        a list of the named entities in the documents
    k: int (optional)
        the number of entities to be returned (default all matches)
        
    Returns:
    --------
    out: list(tuple(str, int))
        the k entities most common in the documents and their counts
    """
    matches = Counter()
    for tokens in tokenlists:
        extraction = extract_single(tokens)
        matches.update(Counter(extraction))
    out = matches.most_common(k)
    return out

db = retrieve_database()
