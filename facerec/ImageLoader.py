
# coding: utf-8

# In[20]:

get_ipython().magic('matplotlib notebook')
from camera import take_picture
import skimage.io as io 
import matplotlib.pyplot as plt
import numpy as np
#uncomment for the first time running on a machine
"""from dlib_models import download_model, download_predictor
download_model()
download_predictor()"""

import dlib_models
from dlib_models import load_dlib_models
load_dlib_models()
from dlib_models import models


# In[2]:

def get_img_from_camera():
    """
    Gets an image numpy array from the default camera
    Parameters:
    -----------
    None
    
    Returns:
    --------
    img (numpy array):
    the (H,W,3) rgb values of the image
    """
    img_array = take_picture()
    return img_array


# In[3]:

def get_img_from_file(filepath):
    """
    Gets an image numpy array from the default camera
    Parameters:
    -----------
    the string file path of the picture
    
    Returns:
    --------
    img (numpy array):
    the (H,W,3) rgb values of the image
    """
    img_array = io.imread(filepath)
    return img_array


# In[6]:

def display_img(img_array):
    """
    For testing. Shows the image based on it's numpy array
    Parameters:
    -----------
    None
    
    Returns:
    --------
    None; shows the image
    """
    fig,ax = plt.subplots()
    ax.imshow(img_array)


# In[11]:

def find_faces(img_array):
    """
    Finds all faces in an image
    Parameters:
    -----------
    img_array (numpy array):
        the array (H,W,3) of rgb values for the image
    
    Returns:
    --------
    detections (list):
        each element has the corners of the bounding box for that detected face
    """
    face_detect = models["face detect"]

    # Number of times to upscale image before detecting faces.
    # When would you want to increase this number?
    upscale = 1 

    detections = face_detect(img_array, upscale)  # returns sequence of face-detections
    detections = list(detections)

    det = detections[0] # first detected face in image

    # bounding box dimensions for detection
    l, r, t, b = det.left(), det.right(), det.top(), det.bottom()
    return detections


# In[22]:

def find_descriptors(img_array, detections):
    """
    Provides descriptors of the faces bounded by the detection boxes in the img array
    Parameters:
    -----------
    img_array (numpy array):
        the array (H,W,3) of rgb values for the image
    detections (list):
        each element has the corners of the bounding box for that detected face
    
    Returns:
    --------
    descriptors (list of numpy arrays):
        a list of descriptors for each face in the image (has shape (128,))
    """
    descriptors = []
    for det in detections:
        shape_predictor = models["shape predict"]
        shape = shape_predictor(img_array, det)
        face_rec_model = models["face rec"]
        descriptor = np.array(face_rec_model.compute_face_descriptor(img_array, shape))
        descriptors.append(descriptor)
    return descriptors

def describe():
    """
    Takes a picture and finds the descriptors of each face in it
    Parameters:
    -----------
    None; will use configured camera
    
    Returns:
    --------
    descriptors (list of numpy arrays):
        a list of descriptors for each face in the image (has shape (128,))
    """
    img = get_img_from_camera()
    rects = find_faces(img)
    descriptors = find_descriptors(img, rects)
    return descriptors