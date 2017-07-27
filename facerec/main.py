
# coding: utf-8

# In[2]:

import ImageLoader
import ImageCompare
import database
import matplotlib.pyplot as plt
import matplotlib.patches as patches
get_ipython().magic('matplotlib notebook')

from camera import save_camera_config
save_camera_config(port=1, exposure=0.7)


# In[3]:

def identify(save=True, force_input=False, from_file=False):
    """
    Takes a picture with configured camera and identifies all of the faces in the picture
    Parameters:
    -----------
    save (boolean):
        whether or not to add the captured image to the database
    from_file(boolean):
        whether or not expect a filename instead of taking a picture
    
    Returns:
    --------
    names (list)
        the list of the name of each person in the picture
    """
    if not from_file:
        img = ImageLoader.get_img_from_camera()
        dets = ImageLoader.find_faces(img)
        descs = ImageLoader.find_descriptors(img, dets)
    else:
        filepath = input('Please enter the location (filepath) of the image: ')
        img = ImageLoader.get_img_from_file(filepath)
        dets = ImageLoader.find_faces(img)
        descs = ImageLoader.find_descriptors(img, dets)
    names = ImageCompare.compare_faces(descs, database.database, threshold=0.4)
    if save:
        if len(descs) > 1:
            print("Cannot add multiple people at once.")
        elif len(descs) < 1:
            print("There's no one there!")
        else:
            if force_input:
                database.add_image(descs[0])
            else:
                database.add_image(descs[0], name=names[0])
    draw_faces(dets, names, img)
    return names


# In[4]:

def draw_faces(detections, people, img):
    """
    Draws bounding boxes over image, and labels them with people.
    
    Parameters
    ----------
    detections: list of rectangles
        List of bounding box rectangles corresponding to the position of each detected face.
    people: list of strings
        List of the keys/names of people as found by compare_faces(), or None if no match is found.
    img: numpy array, shape (480, 640, 3)
        The array representing the image.
    
    Returns:
    --------
    None
    """
    fig, ax = plt.subplots()
    ax.imshow(img)
    for i in range(len(detections)):
        d = detections[i]
        rect = patches.Rectangle((d.left(), d.top()), d.width(), d.height(), fill=False, linewidth=1.2, color='#57FF36')
        ax.add_patch(rect)
        if people[i] is not None:
            ax.text(d.left() + 8, d.top() + d.height() + 15, people[i], backgroundcolor='#57FF36', fontsize='5', color='black', weight='bold')
    plt.show()


# In[5]:

def go():
    """
    Takes a picture from the configured camera and displays the image with recognized faces and labels
    Parameters:
    -----------
    None
    
    Returns:
    --------
    None; shows the image with captioned faces
    """
    img = ImageLoader.get_img_from_camera()
    dets = ImageLoader.find_faces(img)
    descs = ImageLoader.find_descriptors(img, dets)
    compared = ImageCompare.compare_faces(descs, database.database)
    draw_faces(dets, compared, img)


# In[8]:

identify(from_file=True)


# In[6]:

go()


# In[7]:

def add_file(filepath):
    """
    Adds a person to the database given a picture of their face
    Will ask for their name
    
    Parameters
    ----------
    filepath (string):
        the location of the file that is the picture of the person's face
    Returns:
    --------
    None
    """
    img = ImageLoader.get_img_from_file(filepath)
    det = ImageLoader.find_faces(img)
    descriptor = ImageLoader.find_descriptors(img, det)
    database.add_image(descriptor)


# In[8]:

print(len(database.retrieve_database().get('Ameer Syedibrahim')[0]))


# In[ ]:

print(database.database.keys())


# In[ ]:

database.database

