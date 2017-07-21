from os import path
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_path = Path(path.dirname(path.abspath(__file__)))


__all__ = ["load_cifar10", "load_mnist", "ToyData"]


def load_cifar10(fname='cifar-10-python.npz'):
    """ The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
        There are 50000 training images and 10000 test images.
        https://www.cs.toronto.edu/~kriz/cifar.html
        Returns
        -------
        training-data, training-labels, test-data, test-labels, category-names"""
    url = 'https://www.dropbox.com/s/8ytbmlvy2au72lc/cifar-10-python.npz?dl=0'

    if not (_path / fname).exists():
        msg = '\n\n'.join(["Data not found! Please download the data (cifar-10-python.npz) from here:",
                           url,
                           "and place the file in this directory:\n\t{}".format(_path)])
        raise FileNotFoundError(msg)
    labels = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
              "horse", "ship", "truck")
    with np.load(str(_path / fname)) as data:
        xtr, ytr, xte, yte = tuple(data[key] for key in ['x_train', 'y_train', 'x_test', 'y_test'])
    print("cifar-10 loaded")
    return xtr, ytr, xte, yte, labels


def load_mnist(fname="mnist.npz"):
    """ The MNIST database of handwritten digits, has a training set of 60,000 examples, and a test set of
        10,000 examples. It is a subset of a larger set available from NIST. The digits have been
        size-normalized and centered in a fixed-size image.
        http://yann.lecun.com/exdb/mnist/
        Returns
        -------
        training-data, training-labels, test-data, test-labels"""
    with np.load(str(_path / fname)) as data:
        out = tuple(data[str(key)] for key in ['x_train', 'y_train', 'x_test', 'y_test'])
    print("mnist loaded")
    return out


class ToyData(object):
    """ Curates a randomly-generated dataset of spiraling points, for use in toy problems.
        Each 'class' in this dataset is its own 'tendril' in the spiral."""

    def __init__(self, num_classes=3, points_per_class=120, dimensionality=2, seed_value=None):
        """ Parameters
            ----------
            num_classes : int, optional (default=3)
                Number of tendrils
            points_per_class : int, optional (default=120)
                Number of points generated in each tendril.
            dimensionality : int, optional (default=2)
                Dimensionality of the generated data. 2D is required for plotting.
            seed_value : Optional[int]
                Provide a seed-value to control the generation of the dataset."""
        if seed_value is not None:
            np.random.seed(seed_value)

        N = points_per_class  # number of points per class
        D = dimensionality  # dimensionality
        K = num_classes  # number of classes

        n_train = round(N // 1.2)
        n_val = N - n_train
        self._coords = np.zeros((N * K, D))  # coordinates
        self._labels = np.zeros(N * K, dtype='uint8')  # class labels (used for plotting)

        """ y_labels is a NxK array of truth values used to train model. A value 1 (0) indicates
            that the point is (isn't) a member of a class. For instance, if point 0 is a member of 
            class 0, then y_labels[0] == array([1., 0., 0.])"""
        y_labels = np.zeros((N * K, K), dtype='uint8')
        for j in range(K):
            ix = range(N * j, N * (j + 1))
            r = np.linspace(0.0, 1, N)  # radius
            t = np.linspace(j * (K + 1), (j + 1) * (K + 1), N) + np.random.randn(N) * 0.2  # theta
            self._coords[ix] = np.column_stack((r * np.sin(t), r * np.cos(t)))
            self._labels[ix] = j
            y_labels[ix, j] = 1

        train_ids = np.concatenate([np.random.choice(range(N * i, N * (i + 1)), n_train, replace=False)
                                    for i in range(K)])
        train_ids = np.random.choice(train_ids, K * n_train, replace=False)
        y_ids = np.random.choice(list(set(range(K * N)) - set(train_ids)),
                                 K * n_val, replace=False)

        self.x_train = self._coords[train_ids,].astype('float32')
        self.y_train = y_labels[train_ids,].astype('float32')

        self.x_test = self._coords[y_ids,].astype('float32')
        self.y_test = y_labels[y_ids,].astype('float32')

        # create the mesh spanning the domain
        X = self._coords
        h = 0.02  # spacing
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        self._meshgrid = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        xx, yy = self._meshgrid
        self._domain_data = np.column_stack([xx.flat, yy.flat]).astype('float32')

        self._scatter_config = dict(c=self._labels, s=40,
                                    cmap=plt.cm.Spectral,
                                    edgecolors='black')

    def load_data(self):
        """ Returns
            -------
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
                Training and validation data/labels."""
        return self.x_train, self.y_train, self.x_test, self.y_test

    def _check_plotability(self):
        if self._coords.ndim != 2:
            raise NotImplementedError("plotting is only supported for 2D data")
        if plt is None:
            raise ImportError("No module named matplotlib")

    def plot_spiraldata(self):
        """ Plot the dataset, with a point's color corresponding to its class.
            Returns
            -------
            Tuple[matplotlib.figure.Figure, matplotlib,axes._subplots.AxesSubplot]
            """
        self._check_plotability()

        fig, ax = plt.subplots()
        ax.scatter(self._coords[:, 0], self._coords[:, 1], **self._scatter_config)

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_title("Spiral Dataset (colored by class)")
        ax.set_axis_bgcolor('lightgrey')
        plt.show()
        return fig, ax

    def visualize_model(self, fwd_pass, entropy=False):
        """ Plot a visualization of a model's classification of the problem's state space.
            Parameters
            ----------
            fwd_pass : Callable[[numpy.array] numpy.array]
                Maps input data, shape=(N, 2), to an array of shape (N, K), representing
                N classification-probability distributions over K classes.
            entropy : bool, optional (default=False)
                If True, a visualization of the model's learned entropy is plotted.
            Returns
            -------
            Tuple[matplotlib.figure.Figure, matplotlib,axes._subplots.AxesSubplot]"""

        def _entropy(x):
            return -np.sum(x * np.log(x), axis=1)

        self._check_plotability()
        xx, yy = self._meshgrid
        z = fwd_pass(self._domain_data)  # The classification scores for each point

        if entropy:
            probs = z
            surface = _entropy(probs).reshape(xx.shape)
            cmap = plt.cm.viridis
            descr = "entropy map"
        else:
            surface = np.argmax(z, axis=1).reshape(xx.shape)
            cmap = plt.cm.Spectral
            descr = "classification boundaries"

        # plot the spiral data and a contour plot of the model's classification across the domain
        fig, ax = plt.subplots()
        ax.contourf(xx, yy, surface, cmap=cmap, alpha=0.8)
        ax.scatter(self._coords[:, 0], self._coords[:, 1], **self._scatter_config)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())

        ax.set_title("Visualization of model's " + descr)
        plt.show()
        return fig, ax