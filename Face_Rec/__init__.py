from os import path
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_path = Path(path.dirname(path.abspath(__file__)))

__all__ = ["image_loader", "image_compare", "database"]


class image_loader():
    """
    Allows access to functions to take pictures using built-in cameras,
    load images from files, and get descriptors and locations
    of faces in images using Davis King's Dlib
    """
    from camera import take_picture
    import skimage.io as io
    import matplotlib.pyplot as plt
    import numpy as np
    # uncomment for the first time running on a machine
    """from dlib_models import download_model, download_predictor
    download_model()
    download_predictor()"""

    import dlib_models
    from dlib_models import load_dlib_models
    load_dlib_models()
    from dlib_models import models

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


class image_compare():
    """
    Contains functions to compare faces found in images; use in conjunction with image_loader
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

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
        for key, data in database.items():
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

    def compare_faces(descriptors, database):
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
            result = identify_face(d, database)
            people.append(result)
        return people


class database():
    """
    Adds and removes people's facial features from a dictionary where the key is thier name
    the dictionary be saved to a text file
    """
    import pickle
    import numpy as np

    DATABASE_FR = "data/facial_features.txt"

    database = {}

    def new_database(filepath=DATABASE_FR):
        """
        Creates a new text file and folder in the filepath; uses 

        If creating additional filepaths, specify it in the filepath variable 
        in all functions with the filepath kwarg
        """
        f = open(filepath,"w+")

    new_database(filepath=DATABASE_FR)

    def retrieve_database(filepath=DATABASE_FR):
        with open(filepath, "rb") as f:
            db = pickle.load(f)
        return db

    def write_database(filepath=DATABASE_FR):

        """
        Simple function that writes to the Database

        """
        with open(filepath, "wb") as f:
            pickle.dump(database, f)

    write_database(DATABASE_FR)

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

        if name != None:
            old_descriptor_list = list(database.get(name))[0]

            old_descriptor_list.append(descriptor)

            new_list = old_descriptor_list

            num_descriptors = len(new_list)

            temp_arr = np.array(new_list)

            new_mean = np.sum(temp_arr) / num_descriptors

            database[name] = [new_list, new_mean]

        if name == None:
            the_name = input("Please enter your name: ")

            the_descriptors = []
            the_descriptors.append(descriptor)

            database[the_name] = [the_descriptors]

            mean_val = descriptor

            database[the_name].append(mean_val)

            # def add_multiple_images(num_detections, descriptors, names):


            # for i in range(num_detections):

            # add_image(descriptors[i],names[i])

    def clear_database(password):

        """
        Clears everything in the database given the incoming parameter 'password'
        """

        if password.lower() == "yes i am sure":
            if input("Are you very sure?").lower() == "y":
                global database
                database = {}

    # Start
    def initialize():

        """
        Initializes the Database
        """

        global database
        database = retrieve_database()

    initialize()

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

        del database[name]




