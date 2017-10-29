""" I/O, helper functions, visualization
"""
from collections import OrderedDict
import os
import h5py
import pickle
import logging, logging.handlers
import numpy as np
import lasagne as nn
from math import factorial

TIME_FORMAT = "%Y%m%d %H%M%S"
""" str : Default time format used by the logger """

LOG_FORMAT = '%(asctime)-15s %(levelname)-6s %(name)-8s %(message)-60s'
""" str : Default messsage format used by the logger """

INP_PSIZE = 192
OUT_PSIZE = 201


def crop(x):
    """ For comparison, crop output layer to have output shape 208x208
    """
    offset = int((INP_PSIZE - OUT_PSIZE) // 2)
    return x[:, :, offset:offset + OUT_PSIZE, offset:offset + OUT_PSIZE]


def logger_setup(filename=None, fmt_log=LOG_FORMAT, fmt_time=TIME_FORMAT, level=logging.DEBUG):
    ''' Configure the logger

    Set up the logger and configure the save path for the log file as well as formats and logging level.
    When no arguments are given, logging output is not saved on disk and the default values for format
    and log level are used.

    Parameters
    ---------
    filename : Optional[str]
        filename of logfile. The associated directory has to exist, otherwise an error message is thrown
    fmt_log : Optional[str]
        Format used for log messages. See ``ilu_deepml.tools.LOG_FORMAT``
    fmt_time: Optional[str]
        Time format used for log messages. See ``ilu_deepml.tools.TIME_FORMAT``
    level: Optional[logging.level]
        Log level
        
    Example
    -------
    >>> import logging
    >>> from ilu_deepml.tools import logger_setup
    >>> logger_setup()
    >>> log = logging.getLogger(__name__)
    >>> log.info("Test message")

    '''
    formatter = logging.Formatter(fmt_log, fmt_time)
    if filename is not None:
        fh = logging.handlers.RotatingFileHandler(filename, mode='a', maxBytes=10 * 1024 ** 2, backupCount=10,
                                                  encoding='utf-8')
        fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    rootLogger = logging.getLogger('')
    rootLogger.addHandler(ch)
    if filename is not None:
        rootLogger.addHandler(fh)
    rootLogger.setLevel(level=level)

def load_weights(outp_layer, fname):
    with open(fname, "rb") as fp:
        data = pickle.load(fp)
    weights = data["params"]
    nn.layers.set_all_param_values(outp_layer, weights)
    return data

def require_dir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    return dirname


def panelize(img):
    if img.ndim == 1:
        raise ValueError("Invalid dimensions for image data" + str(img.shape))
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return panelize(img[np.newaxis, :, :])

    nb = img.shape[0]
    nb_rows = int(nb ** 0.5)
    psize = img.shape[2]
    nb_channel = img.shape[1]

    w, h = img.shape[-2:]

    img_per_row = nb // nb_rows
    rows = []
    for j in range(nb_rows):
        start = j * img_per_row
        stop = min(start + img_per_row, nb)
        rows.append(
            np.hstack([img[j, :, :, :].reshape(nb_channel, w, h).transpose((1, 2, 0)) for j in range(start, stop)]))
    return np.vstack(rows)


def get_updates(error, layers, optimizer, learning_rates, lfilter=None):
    log = logging.getLogger(__name__)
    updates = OrderedDict()
    params = []
    for k in layers.keys():
        if lfilter(k):
            p = list(param for param in layers[k].params.keys() if "trainable" in layers[k].params[param])
            if not p:
                continue

            if k in learning_rates.keys():
                log.info("Custom lr = %f for %s", learning_rates[k], k)
                lr = np.float32(learning_rates[k])
            else:
                log.info("Default lr = %f for %s", learning_rates["default"], k)
                lr = np.float32(learning_rates["default"])

            dp = nn.updates.nesterov_momentum(error, learning_rate=lr, params=p)

            updates.update(dp)
            params.append(p)
    return updates, params

def get_dataset(fname="data/train_patches_192.hdf5", train_key="H.E.T.", val_key="H.E.T+",
                val_slides=["47453", "74235"]):
    to_list = lambda x : x if isinstance(x, list) else [x]
    train_key = to_list(train_key)
    val_key = to_list(val_key)
    X = []
    Xv = []
    with h5py.File(fname, "r") as ds:
        for key in list(ds.keys()):
            if not key in val_slides:
                for lbl in train_key:
                    X.append(ds[key][lbl][...])
            else:
                for lbl in val_key:
                    Xv.append(ds[key][lbl][...])
    X = np.concatenate(X, axis=0)
    Xv = np.concatenate(Xv, axis=0)
    np.random.shuffle(X)
    return np.float32(X), np.float32(Xv)

def get_dataset_custom():
    """ Implement your own dataset """

    # format: training, validation set
    return X, Xv
 


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688

    Snippet from scipy cookbook (http://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay)
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')
