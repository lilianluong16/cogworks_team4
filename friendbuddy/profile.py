import Face_Rec
import numpy as np
import pickle
from collections import defaultdict

database = defaultdict(dict)


def find_profile():
    """
    Takes a photo and returns detected names and descriptors.

    Returns
    -------
    Case: no faces --> False
    Case: one unknown face --> numpy array (desc)
    Case: one known face --> string (name)
    Case: multiple unknown faces --> -1
    Case: multiple known faces --> list of strings (names)
    """
    names, descs = Face_Rec.go_friend()
    print(names)
    if len(names) == 0:
        # No faces
        return False
    elif len(names) == 1 and names[0] is None:
        # One unknown face
        return descs[0]
    elif len(names) == 1:
        # One known face
        return names[0]
    else:
        names = np.array(names)
        identified_names = names[names != np.array(None)]  # Filters Nones out of list
        print(identified_names)
        if len(identified_names) == 0:
            # Multiple unknown faces
            return -1
        if len(identified_names) == 1:
            # One known face and other unknown ones
            return identified_names[0]
        # Multiple known faces
        return identified_names.tolist()


def write_database(filepath="data/profiles.txt"):
    """
    Writes the profiles database to text file.

    Parameters
    ----------
    filepath: String
        The filepath pointing to the text file.
    """
    global database
    with open(filepath, "wb") as f:
        pickle.dump(database, f)


def read_database(filepath="data/profiles.txt"):
    """
    Reads the profiles database from text file and sets global variable.

    Parameters
    ----------
    filepath: String
        The filepath pointing to the text file.
    """
    global database
    with open(filepath, "rb") as f:
        database = pickle.load(f)


def initialize():
    """
    Called at start of program to retrieve database and call Face_Rec initialization.
    """
    read_database()
    Face_Rec.initialize()


def get_property(profile_name, ppty):
    """
    Gets an associated property from profile.

    Parameters
    ----------
    profile_name: String
        Name associated with profile.
    ppty: String
        Name of property.

    Returns
    -------
    Case: ppty property exists --> string (value)
    Else: None
    """
    global database
    if ppty in database[profile_name]:
        return database[profile_name][ppty]
    return None


def change_property(profile_name, ppty, new_value):
    """
    Updates an associated property of profile.

    Parameters
    ----------
    profile_name: String
        Name associated with profile.
    ppty: String
        Name of property.
    new_value: String
        The new value to which the property should be set.

    Returns
    -------
    True
    """
    global database
    database[profile_name][ppty] = new_value
    return True


def add_to_list(profile_name, list_name, items):
    """
    Adds items to a list.

    Parameters
    ----------
    profile_name: String
        Name associated with profile.
    list_name: String
        Name of list.
    items: List
        List of strings to be added to list.

    Returns
    -------
    Case: list_name variable is not a list --> False
    Else: True
    """
    global database
    if list_name in database[profile_name]:
        if isinstance(database[profile_name][list_name], list):
            database[profile_name][list_name] += items
        else:
            return False
    else:
        database[profile_name][list_name] = items
    return True


def remove_from_list(profile_name, list_name, items):
    """
    Removes items from a profile's list, if present.

    Parameters
    ----------
    profile_name: String
        Name associated with profile.
    list_name: String
        Name of list.
    items: Union(List, Tuple, np.ndarray)
        Iterable of strings to be removed to list.

    Returns
    -------
    Case: list_name variable is not a list or does not exist --> False
    Else: True
    """
    global database
    if list_name not in database[profile_name] or not isinstance(database[profile_name][list_name], list):
        return False
    for i in items:
        database[profile_name][list_name].remove(i)
    return True


def get_list(profile_name, list_name):
    """
    Obtains list from profile, if available.

    Parameters
    ----------
    profile_name: String
        Name associated with profile.
    list_name: String
        Name of list.

    Returns
    -------
    Case: list_name list exists --> List
    Else: None
    """
    global database
    if list_name in database[profile_name]:
        return database[profile_name][list_name]
    return None


def delete_variable(profile_name, item_name):
    """
    Deletes/resets variable of profile.

    Parameters
    ----------
    profile_name: String
        Name associated with profile.
    item_name: String
        Name of variable to be deleted.

    Returns
    -------
    None
    """
    global database
    del database[profile_name][item_name]


def tell_list(profile_name, list_name):
    """
    Returns a string that represents items in a list.

    Parameters
    ----------
    profile_name: String
        Name associated with profile.
    list_name: String
        Name of list.

    Returns
    -------
    String
    """
    global database
    l = get_list(profile_name, list_name)
    if l is None:
        return profile_name + "'s " + list_name + " list is empty."
    if len(l) == 1:
        return "Your list contains " + l[0] + "."
    return "Your list contains " + ", ".join(l[:-1]) + ", and " + l[-1] + "."


def add_person(name, img):
    """
    Adds person to Face_Rec database.

    Parameters
    ----------
    name: String
        Name inputted by user.
    img: np.ndarray
        Face descriptor

    Returns
    -------
    String
    """
    Face_Rec.add_image(img, name)
    return name

initialize()
