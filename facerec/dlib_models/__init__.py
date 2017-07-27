""" Provide absolute paths to trained face-recognition and shape-prediction models. Also provides download utilities
    if you do not have the files.

    All downloads will be sent to this directory."""

import os as _os
from pathlib import Path as _Path

__all__ = ["model_path", "predictor_path", "download_model", "download_predictor", "load_dlib_models"]

_path = _Path(_os.path.dirname(_os.path.abspath(__file__)))

# absolute path to model & shape predictor
model_path = _path / "dlib_face_recognition_resnet_model_v1.dat"
predictor_path = _path / "shape_predictor_68_face_landmarks.dat"

models = {"face rec": None, "shape predict": None, "face detect": None}


def _load():
    """ Load the dlib face detector, shape predictor, and face recognition models, if
        they are not already loaded.

        Call this function explicitly if you want to load these up front; otherwise
        the `face_rec` utility that first requires these will load them automatically.

        NOTE: `models` must be imported *AFTER* they are loaded."""
    global models

    if all(x is not None for x in models.values()):
        return None
    import dlib

    if models["face detect"] is None:
        models["face detect"] = dlib.get_frontal_face_detector()

    if models["shape predict"] is None:
        if not predictor_path.is_file():
            print("dlib's shape-predictor needs to be downloaded. Run:\n\ndlib_models.download_predictor()")
            if not model_path.is_file():
                print("dlib's resnet model needs to be downloaded. Run:\n\ndlib_models.download_model()")
            return None
        models["shape predict"] = dlib.shape_predictor(str(predictor_path))

    if models["face rec"] is None:
        if not model_path.is_file():
            print("dlib's resnet model needs to be downloaded. Run:\n\ndlib_models.download_model()")
            return None
        models["face rec"] = dlib.face_recognition_model_v1(str(model_path))


def load_dlib_models(func=None):
    """ This function can be invoked directly to lazy-load the dlib models,
        or as a decorator, such that the models are lazy-loaded prior to calling
        that function.

        NOTE: `models` must be imported *AFTER* they are loaded.

        See dlib_models.load for more information.

        Parameters
        ----------
        func : Optional[Callable]

        Returns
        -------
        Union[None, Callable] """
    if func is None:
        _load()
        return None

    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        _load()
        return func(*args, **kwargs)

    return wrapper


def _download(url, path):
    """ Generic function for downloading and decompressing a .bz2 file

        Parameters
        ----------
        url : str
            Url-location of download

        path : pathlib.Path
            Path-object for path-destination of download"""
    import urllib.request
    import bz2

    if path.is_file():
        print("File already exists:\n\t{}".format(path))
        return None

    print("Downloading \n\tfrom {}\n\tto: {}".format(url, _path))
    with urllib.request.urlopen(url) as response:
        with path.open(mode="wb") as new, bz2.BZ2File(response, "rb") as uncompressed:
            new.write(uncompressed.read())

    print("Downloaded and decompressed: {}".format(path))


def download_model():
    """ Download trained face-recognition resnet dlib model (bz2) and decompress.

        Returns
        -------
        None"""
    url = 'http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2'
    _download(url, model_path)


def download_predictor():
    """ Download trained facial-shape detector from dlib (bz2) and decompress.

        Returns
        -------
        None"""
    url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    _download(url, predictor_path)

