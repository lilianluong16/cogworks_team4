from os import path, makedirs
from pathlib import Path
from camera import take_picture
import os
import pickle
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import skimage.io as io
import dlib_models
from dlib_models import load_dlib_models
from dlib_models import models
import numpy as np
from camera import save_camera_config
import cloudinary
import cloudinary.uploader
import cloudinary.api


load_dlib_models()
save_camera_config(port=1, exposure=0.7)

_path = Path(path.dirname(path.abspath(__file__)))

__all__ = ['get_img_from_camera', 'get_img_from_file', 'display_img', 'find_faces', 'find_descriptors', 
'describe', 'calc_dist', 'is_face', 'identify_face', 'compare_faces', 'new_database', 'retrieve_database',
'write_database', 'add_image', 'initialize', 'clear_database' , 'del_person', 'identify', 'draw_faces', 'go', 'add_file']

# uncomment for the first time running on a new machine
"""from dlib_models import download_model, download_predictor
download_model()
download_predictor()"""

# TO CHANGE DEFAULT DATA FILE, CHANGE STRING BELOW
DATABASE_FR = "data/facial_features.txt"


# creates data file if it doesn't exist
if not os.path.exists(DATABASE_FR):
        os.makedirs('/'.join(str.partition(DATABASE_FR, "/")[:-1]))
        with open(DATABASE_FR, "w+"):
            pass

db = {}

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
    fig, ax = plt.subplots()
    ax.imshow(img_array)

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
    if len(detections) > 0:
        det = detections[0]  # first detected face in image

        # bounding box dimensions for detection
        l, r, t, b = det.left(), det.right(), det.top(), det.bottom()
    return detections

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

def calc_dist(test, data):
    """
    Calculates the L2 distance between two feature vectors.

    Parameters
    ----------
    test: numpy array, shape (N,)
    data: numpy array, shape (N,)

    Returns
    -------
    float
    """
    return np.sqrt(np.sum((test - data) ** 2))

def is_face(test_desc, profile_mean, threshold):
    """
    Determines whether or not a descriptor is close enough to a face,
    returning False if the L2 distance is greater than the threshold.

    Parameters
    ----------
    test_desc: numpy array, shape (N,)
        The descriptor of the unknown face being tested.
    profile_mean: numpy array, shape (N,)
        The mean of descriptors for the profile being tested.
    threshold: numerical value (int, float)
        The maximum L2 distance accepted as a match.

    Returns
    -------
    float, if L2 distance is less than the threshold
    None, otherwise
    """
    l2d = calc_dist(test_desc, profile_mean)
    if l2d < threshold:
        return l2d
    return None

def identify_face(desc, database, threshold=0.5, face_thres=0):
    """
    Compares a test descriptor to all faces in a database and determines the best match, if any.

    Parameters
    ----------
    desc: numpy array, shape (N,)
        The descriptor of the unknown face being tested.
    database: dictionary
        The database containing name keys and a list of descriptor vectors as well as the mean.
    threshold: numerical value (int, float)
        The maximum L2 distance accepted as a face match.
    face_thres: numerical value (int, float)
        The minimum distance between the top two matches to count a match.

    Returns
    -------
    string, representing the name/key if a match is found
    None, otherwise
    """
    matches = []
    for key, data in db.items():
        i_f = is_face(desc, data[1], threshold)
        if i_f is not None:
            matches.append((key, i_f))
    if len(matches) == 0:
        return None
    if len(matches) == 1:
        return matches[0][0]
    matches = sorted(matches, key=lambda x: x[1])
    if matches[1][1] - matches[0][1] > face_thres:
        return matches[0][0]
    return None

def compare_faces(descriptors, database, threshold=0.45):
    """
    Compares each face with the database and returns a list of detected people.

    Parameters
    ----------
    descriptors: list of numpy arrays
        List of descriptor vectors corresponding to the features of each face.
    database: dictionary
        The database containing name keys and a list of descriptor vectors as well as the mean.

    Returns
    -------
    list of strings, or None if match not found for that unit
    """
    people = []
    for d in descriptors:
        result = identify_face(d, database, threshold=threshold)
        people.append(result)
    return people

def new_database(filepath=DATABASE_FR):
    """
    Creates a new text file and folder in the filepath; uses 

    If creating additional filepaths, specify it in the filepath variable 
    in all functions with the filepath kwarg
    """
    if not os.path.exists(filepath):
        os.makedirs(str.partition(filepath, "/")[0])
        with open(filepath, "w+"):
            pass


def retrieve_database():
    global db
    with open(DATABASE_FR, "rb") as f:
        db = pickle.load(f)
    return db


def write_database(self, filepath=DATABASE_FR):

    """
    Simple function that writes to the Database

    """
    with open(filepath, "wb") as f:
        global db
        pickle.dump(db, f)

# Add image to database
def add_image(descriptor, name=None):

    """
    Assigns a descritpor to a name depending on whether the name is already in the Database or not.

    Parameters
    ----------
    descriptor: numpy.array, shape (128,)
        The descriptor of the face whose image is to be added to the Database
    name= string
        If available, the name of the face is passed to added the corresponding descriptor to the Database

    Returns
    -------
    Nothing. The purpose of this function is to associate the incoming descriptor with the right name (if present)
    or to ask the user to input a new name and associate it with the incoming descriptor
    """
    global db
    if name != None:
        old_descriptor_list = list(db.get(name))[0]

        old_descriptor_list.append(descriptor)

        new_list = old_descriptor_list

        num_descriptors = len(new_list)

        temp_arr = np.array(new_list)

        new_mean = np.sum(temp_arr) / num_descriptors

        db[name] = [new_list, new_mean]

    if name == None:
        the_name = input("Please enter your name: ")

        if the_name in db:
            add_image(descriptor, name=the_name)
        else:
            
            db[the_name] = [[descriptor], descriptor]

def clear_database(password):

    """
    Clears everything in the database given the incoming parameter 'password'
    """

    if password.lower() == "yes i am sure":
        if input("Are you very sure?").lower() == "y":
            global db
            db = {}

# Start
def initialize():

    """
    Initializes the Database
    """
    cloudinary.config(
        cloud_name="luong44976",
        api_key="165891819185365",
        api_secret="p2ib0QA6Rl2nK8CNxlBFQeJmoaM"
    )
    global db
    db = retrieve_database()


def del_person(name):

    """
    Deletes a person and their descriptors and mean from the Database.

    Parameters
    ----------

    name= string
        The name of the individual whose descriptors are to be deleted from the Database

    Returns
    -------
    Nothing. The incoming name parameter is simply deleted, along with its accompanying descriptor(s) and mean
    """

    del db[name]

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
        img = get_img_from_camera()
        dets = find_faces(img)
        descs = find_descriptors(img, dets)
    else:
        filepath = input('Please enter the location (filepath) of the image: ')
        img = get_img_from_file(filepath)
        dets = find_faces(img)
        descs = find_descriptors(img, dets)
        names = compare_faces(descs, db, threshold=0.4)
    if save:
        if len(descs) > 1:
            print("Cannot add multiple people at once.")
        elif len(descs) < 1:
            print("There's no one there!")
        else:
            if force_input:
                add_image(descs[0])
            else:
                add_image(descs[0], name=names[0])
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
    # plt.show()
    plt.savefig('static/img.png')
    return cloudinary.uploader.upload('static/img.png')['secure_url']

def go():
    """
    Takes a picture from the configured camera and displays the image with recognized faces and labels
    Parameters:
    -----------
    None

    Returns:
    --------
    compared: list of strings
        Names of everyone found in photo.
    img: numpy array
        The image itself.
    url: string
        URL of location for img file
    descs: list of numpy arrays
        Face descriptors.
    """
    img = get_img_from_camera()
    dets = find_faces(img)
    descs = find_descriptors(img, dets)
    compared = compare_faces(descs, db)
    url = draw_faces(dets, compared, img)
    return compared, img, url, descs

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
    img = get_img_from_file(filepath)
    det = find_faces(img)
    descriptor = find_descriptors(img, det)
    add_image(descriptor)
