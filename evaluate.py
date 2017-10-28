""" Evaluation routines

For specification of paths/parameters, use the ``evaluation.json`` script located
in the same directory as this script.

Following options are available:

1. Directories for pandas tables and plots:

```
    "output_directory" : "output_test2",
    "plot_directory": "plots",
```

2. Directories with pre-computed images

Two options are given: The first one is the path, the second one should be either ``false``
or ``true`` and specifies wether multiple protocols are available within the image folder.
In the latter case, the path should contain a placeholder (``{}``).

```
    "method_directories": {
        "unnormalized" :  ["/images/ILLUMINATE/StainNormalization/final", false],
        "marcenko" :      ["../results/results_marcenko/results_marcenko_{}", true],
        "reinhard" :      ["../results/results_reinhard/results_reinhard_{}", true],
        "lstm_reworked" : ["../results/results_lstm_reworked/lstm_reworked_{}", true]
    },
```

3. Protocol specification

```
    "protocols" : ["HoEoTp",
                   "HoEoTm",
                   "HoEoTo",
                   "HoEpTo",
                   "HoEmTo",
                   "HpEoTo",
                   "HmEoTo",
                   "HpEpTo",
                   "HmEmTo"]
```

"""

__author__ = ["Daniel Bug", "Steffen Schneider"]
__email__  = ["steffen.schneider (at) rwth-aachen.de"]

import matplotlib as mpl

mpl.use("agg")

import numpy as np
import pandas as pd
from skimage.io import imread
from glob import glob
import matplotlib.pyplot as plt
import os.path as op
import os
import joblib as jl
import itertools as it
import seaborn as sns
from skimage.measure import compare_ssim
from skimage.color import rgb2lab
from collections import OrderedDict

pd.options.display.float_format = '{:,.4f}'.format

kde_kernel = np.array([1., 6., 15., 20., 15., 6., 1.], dtype=np.float32)

blocks = (47453, 47691, 48631, 74234, 74235,)
channels = ('r', 'g', 'b',)
metrics = (
    'squared_difference',
    'earth_movers_distance',
    'kullback_leibler_divergence',
    'kolmogorow_smirnow_test',
)
patches = tuple(range(1, 6))
protocols = tuple(range(1, 10))
protocol2name = {
    1: 'HoEoTp',
    2: 'HoEoTm',
    3: 'HoEoTo',
    4: 'HoEpTo',
    5: 'HoEmTo',
    6: 'HpEoTo',
    7: 'HmEoTo',
    8: 'HpEpTo',
    9: 'HmEmTo',
}
protoColors = sns.color_palette("hls", 6)


def set_style():
    """
    Applies the default style to the current plot.
    Call just before call to ``plt.savefig`` or ``plt.show``
    """
    sns.set_context(context='paper')
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times"]
    })


###########################################
#           UPDATE to PANDAS              #
###########################################

def parse_filename(fname):
    _, block, patch, protocol = os.path.basename(fname).split('_')[:4]
    protocol = protocol[:2]
    return block, patch, protocol


def crop(img, w, h):
    """
    Extract part of size w x h from center of the given image
    """
    w_, h_ = img.shape[0:2]
    w_off, h_off = (w_ - w) // 2, (h_ - h) // 2
    assert w_ >= w and h_ >= h, "cannot crop from {}x{} to {}x{}".format(w_, h_, w, h)

    return img[w_off:w_off + w, h_off:h_off + h]


def compute_histogram(image):
    """
    Compute the image histogram
    """
    bins = np.arange(0., 257.) - .5

    hist = np.zeros((256, 3), dtype=np.float32)
    hist[:, 0], _ = np.histogram(image[:, :, 0].flatten(), bins=bins, density=True)
    hist[:, 1], _ = np.histogram(image[:, :, 1].flatten(), bins=bins, density=True)
    hist[:, 2], _ = np.histogram(image[:, :, 2].flatten(), bins=bins, density=True)
    return hist


def compute_ssim(src, image):
    """
    Compute SSIM metric
    """
    srcImage = crop(src, *image.shape[0:2])

    sim, _ = compare_ssim(srcImage, image, data_range=image.max() - image.min(), gaussian_weights=True,
                           multichannel=True, full=True)
    return sim


def get_original_path(iFile, srcImageDir):
    """
    Given a filename from the directory of normalized image and the directory of unnormalized images,
    return the path to the matching unnormalized image file
    """
    slidedir = op.basename(op.dirname(iFile))
    basename = op.basename(iFile)
    block, patch, protocol = parse_filename(iFile)
    iSrcFile = list(glob(op.join(srcImageDir, slidedir, "*_{}_{}_*".format(patch, protocol))))[0]

    return iSrcFile


def compute_lab_volume(image):
    """
    Compute the perceivable color volume in LAB space as a measure for the used colors (or color flatness)
    Parameters
    ----------
    image: the image to compute the LAB space volume for

    Returns
    -------
    LABvolume = std(L)*std(a)*std(b)
    """

    lab = rgb2lab(image)
    lab_volume = np.product(np.std(lab, axis=(0, 1)), axis=0)

    return lab_volume


def readData(imageDir, srcDir=None, ext='tif'):
    """
    Utility to prepare a dataset from a folder of images.
    :param imageDir: path-str postpended with '\\*\\*.' + ext and passed into glob()
    :param ext: defines the image extension
    :return: pandas.DataFrame containing the histogram information for analysis
    """
    data = []
    column_names = None
    for iFile in glob(imageDir + '/*/*.' + ext):
        print('Reading', iFile)
        basename = op.basename(iFile)
        block, patch, protocol = parse_filename(iFile)

        block, patch, protocol = int(block), int(patch), int(protocol)
        image = imread(iFile)

        columns = [block, patch, protocol, basename]
        column_names = ['block', 'patch', 'protocol', 'file']

        # metrics computed only on a single image
        hist = compute_histogram(image)
        columns.append(hist)
        column_names.append("histogram")

        lab_volume = compute_lab_volume(image)
        columns.append(lab_volume)
        column_names.append("lab_volume")

        # metrics computed between image and source image
        if srcDir is not None:
            print('Reading', srcDir)
            src_img = imread(get_original_path(iFile, srcDir))
            ssim = compute_ssim(src_img / 255., image / 255.)
            columns.append(ssim)
            column_names.append("ssim")

        # TODO metrics computed between image and target image
        # [...]

        data.append(columns)

    if column_names is None:
        print("Error: no files found in {}".format(imageDir))
    dataframe = pd.DataFrame(data, columns=column_names)
    return dataframe


def computeMetrics(g_, h_, kde_kernel=None):
    """
    Computes the Sum Squared Difference (SSD), Earth Mover's Distance (EMD), Kullberg-Leibler Divergence (KLD) and
    Kolmogorov-Smirnov Test (KST), between the two distributions g_ and h_. A kernel-density estimate can be used
    as preprocessing by specifying a kde_kernel. KLD is renormalized with eps=1e-12 to avoid zero-probabilities.
    :param g_: Distribution 1 (e.g. numpy.ndarray holding a normalized histogram)
    :param h_: Distribution 2 (e.g. numpy.ndarray holding a normalized histogram)
    :param kde_kernel: numpy.ndarray (will be normalized)
    :return: tuple with the metrics (SSD, EMD, KLD, KST)
    """
    assert isinstance(g_, np.ndarray)
    assert isinstance(h_, np.ndarray)
    assert g_.shape == h_.shape

    if kde_kernel is not None:
        assert isinstance(kde_kernel, np.ndarray)
        kde_kernel = kde_kernel / np.sum(kde_kernel)
        g = np.convolve(g_, kde_kernel, mode='same')
        h = np.convolve(h_, kde_kernel, mode='same')
    else:
        g = g_
        h = h_

    def squared_difference(p_, q_):
        return np.sum(np.square(p_ - q_))

    def kullback_leibler_divergence_hm(p_, q_, eps=1e-12):
        p = p_ + eps
        p = p / np.sum(p)
        q = q_ + eps
        q = q / np.sum(q)
        pq = np.sum(np.log(p / q) * p)
        qp = np.sum(np.log(q / p) * q)
        # return (pq * qp) / (pq + qp)
        return 0.5 * (pq + qp)

    def kolmogorow_smirnow_test(p_, q_):
        P = np.cumsum(p_)
        Q = np.cumsum(q_)
        return np.max(np.abs(P - Q))

    def earth_movers_distance(p_, q_):
        # note: emd special case for two 1-D histograms
        P = np.cumsum(p_)
        Q = np.cumsum(q_)
        return np.sum(np.abs(P - Q))

    return (
        squared_difference(g, h),
        earth_movers_distance(g, h),
        kullback_leibler_divergence_hm(g, h),
        kolmogorow_smirnow_test(g, h)
    )


def comparePatchwise(data, kde_kernel=None):
    """
    Comparing the protocol errors patchwise, thus ignoring the different statistics.
    This corresponds to a macro-statistical approach. The per-patch metrics are averaged externally. Furthermore,
    this method computes average histograms per patch. Using compareMeanHistograms(), the metrics can be recomputed pairwise
    between the normalized averaged per-patch histograms.
    :param data: expects a pandas.DataFrame holding the full dataset
    :param kde_kernel: smoothing the histogramms before computing metrics, default: None
    :return: pandas.DataFrame holding the metrics and averaged histograms
    """
    results = []
    for block, patch in it.product(blocks, patches):
        df = data.loc[(data.block == block) & (data.patch == patch)]
        hists = df['histogram'].as_matrix()  # sound like np.ndarray but is a list/tuple (of #protocols)
        combinations = [(x, y) if x > y else (None, None) for y in range(len(hists)) for x in range(len(hists))]
        metric = []
        for i, j in combinations:
            if i is None or j is None:
                continue

            histA = hists[i]
            histB = hists[j]
            assert histA.shape == (256,3)
            assert histB.shape == (256,3)
            mR = computeMetrics(histA[:, 0], histB[:, 0], kde_kernel)
            mG = computeMetrics(histA[:, 1], histB[:, 1], kde_kernel)
            mB = computeMetrics(histA[:, 2], histB[:, 2], kde_kernel)
            metric.append(np.sum([mR, mG, mB], axis=0))

        metric = np.mean(metric, axis=0)
        avgHist = np.mean(hists, axis=0)
        results.append((block, patch, metric[0], metric[1], metric[2], metric[3], avgHist))

    return pd.DataFrame(results, columns=['block', 'patch', 'SSD', 'EMD', 'KLD', 'KST', 'avgHistogram'])


def compareProtocolwise(data, kde_kernel=None):
    """
    Averaging histograms over all patches (from all blocks) and comparing them pairwise.
    In this interpretation, the different patches are used to sample the true color distribution.
    :param data: expects a pandas.DataFrame holding the full dataset
    :param kde_kernel: smoothing the histogramms before computing metrics, default: None
    :return: pandas.DataFrame holding the computed metrics per protocol combination
    """
    avgHists = []
    for p in protocols:
        df = data.loc[data.protocol == p]
        # list of numpy array, contains 25 blocks
        hists = df['histogram'].as_matrix()
        if len(hists) > 0:
            hists = np.stack(hists, axis=0)
            avgHists.append(hists)

    avgHists = np.stack(avgHists, axis=0)
    print("avg hists", avgHists.shape)

    metric = []
    # compare differences between all ordered pairs (i,j) of protocols
    nb_protocols, nb_patches = avgHists.shape[:2]
    assert nb_protocols == 9, nb_patches == 25

    for block in range(nb_patches):
        for i in range(nb_protocols):
            for j in range(i + 1, nb_protocols):
                histA = avgHists[i, block]
                histB = avgHists[j, block]
                assert histA.ndim > 1 and histB.ndim > 1
                mR = computeMetrics(histA[:, 0], histB[:, 0], kde_kernel)
                mG = computeMetrics(histA[:, 1], histB[:, 1], kde_kernel)
                mB = computeMetrics(histA[:, 2], histB[:, 2], kde_kernel)
                metric.append(np.sum([mR, mG, mB], axis=0))
    # metric = np.mean(metric, axis=0, keepdims=True)
    print(len(metric), metric[0].shape)

    return pd.DataFrame(data=metric, columns=['SSD', 'EMD', 'KLD', 'KST'])


def boxplot_similarity(dfs, measures='ssim', *args, **kwargs):
    """
    Plot SSID using boxplots
    """
    # TODO merge with other boxplot methods

    methods = ['marcenko', 'reinhard', 'lstm_reworked', 'kahn']
    measures = ['ssim', 'lab_volume']

    mtrx_bymethod = []
    df_bymeasure = []

    for key in methods:
        mtrx_bymethod.append(dfs[key].as_matrix(columns=measures))

    bymeasure = np.stack(mtrx_bymethod, axis=0)

    plt.figure(figsize=(10, 5))
    for i, key in enumerate(measures):
        df = pd.DataFrame(data=bymeasure[:, :, i].T, columns=methods)
        plt.subplot(2, 3, i + 1)
        sns.boxplot(data=df, *args, **kwargs)
        plt.title(key)

    return plt


def scatterplot(dfs, methods=["reinhard", "marcenko", "kahn", 'umc', "lstm_reworked"], metrics =['SSD', 'EMD', 'KLD', 'KST']):

    plt.figure(figsize=(4*len(metrics), 3))

    # use different markers for the algorithms
    # use diffrent colors for the protocols

    markers = list("ovdsp")
    colors = sns.color_palette("Set2", 9)

    # metrics are new subplots
    for i, metric in enumerate(metrics):
        # protocols are different colors
        plt.subplot(1, len(metrics), i+1)
        plt.title(metric)

        plots_methods = []
        plots_protocols = []

        for protocol_id, protocol in enumerate(dfs.keys()):
            # methods are different markers
            for m, method in zip(markers, methods):
                data = np.concatenate(dfs[protocol][method].as_matrix(columns=[metric]), axis=0)
                assert data.ndim == 1, data.ndim
                # data is (2700,) matrix
                means, std = data.mean(), data.std()
                scat = plt.scatter(means, std, s=30,c=colors[protocol_id], marker=m)

                if protocol_id==0:
                    plots_methods.append(scat)

            plots_protocols.append(scat)

        if i == len(metrics) - 1:
            plt.legend(plots_methods + plots_protocols,
               methods + list(dfs.keys()),
                loc="right",
                bbox_to_anchor=(1.7, 0.5),
               scatterpoints=1)

    return plt

def boxplot(df_BAS, df_REI, df_MAR, df_KHN, df_FAN, *args, **kwargs):
    """
    Boxplots comparing the different methods subdivided by the different measures.
    :param df_BAS: baseline pandas.DataFrame
    :param df_REI: reinhard pandas.DataFrame
    :param df_MAR: marcenko pandas.DataFrame
    :param df_FAN: FAN method pandas.DataFrame
    :param *args:  additional positional arguments passed to ``sns.boxplot``
    :param *kargs: additional keyword arguments passed to ``sns.boxplot``
    :return: None
    """

    mtrx_BAS = 0*df_BAS.as_matrix(columns=['SSD', 'EMD', 'KLD', 'KST'])
    mtrx_REI = df_REI.as_matrix(columns=['SSD', 'EMD', 'KLD', 'KST'])
    mtrx_MAR = df_MAR.as_matrix(columns=['SSD', 'EMD', 'KLD', 'KST'])
    mtrx_KHN = df_KHN.as_matrix(columns=['SSD', 'EMD', 'KLD', 'KST'])
    mtrx_FAN = df_FAN.as_matrix(columns=['SSD', 'EMD', 'KLD', 'KST'])

    # TODO simplify this, confusing -> np.newaxis is not necessary, simply use stacking
    # TODO quick'n dirty was the first to come to mind, probably better way available in pd anyway
    mtrx_SSD = np.concatenate((mtrx_BAS[:, 0, np.newaxis],
                               mtrx_REI[:, 0, np.newaxis],
                               mtrx_MAR[:, 0, np.newaxis],
                               mtrx_KHN[:, 0, np.newaxis],
                               mtrx_FAN[:, 0, np.newaxis]), axis=1)
    mtrx_EMD = np.concatenate((mtrx_BAS[:, 1, np.newaxis],
                               mtrx_REI[:, 1, np.newaxis],
                               mtrx_MAR[:, 1, np.newaxis],
                               mtrx_KHN[:, 1, np.newaxis],
                               mtrx_FAN[:, 1, np.newaxis]), axis=1)
    mtrx_KLD = np.concatenate((mtrx_BAS[:, 2, np.newaxis],
                               mtrx_REI[:, 2, np.newaxis],
                               mtrx_MAR[:, 2, np.newaxis],
                               mtrx_KHN[:, 2, np.newaxis],
                               mtrx_FAN[:, 2, np.newaxis]), axis=1)
    mtrx_KST = np.concatenate((mtrx_BAS[:, 3, np.newaxis],
                               mtrx_REI[:, 3, np.newaxis],
                               mtrx_MAR[:, 3, np.newaxis],
                               mtrx_KHN[:, 3, np.newaxis],
                               mtrx_FAN[:, 3, np.newaxis]), axis=1)

    df_SSD = pd.DataFrame(data=mtrx_SSD, columns=['BAS', 'REI', 'MAR', 'KHN', 'FAN'])
    df_EMD = pd.DataFrame(data=mtrx_EMD, columns=['BAS', 'REI', 'MAR', 'KHN', 'FAN'])
    df_KLD = pd.DataFrame(data=mtrx_KLD, columns=['BAS', 'REI', 'MAR', 'KHN', 'FAN'])
    df_KST = pd.DataFrame(data=mtrx_KST, columns=['BAS', 'REI', 'MAR', 'KHN', 'FAN'])

    # sns.set_style(style='ticks')
    plt.figure(figsize=(10, 3))
    plt.subplot(141)
    sns.boxplot(data=df_SSD, *args, **kwargs)
    plt.title('SSD')
    plt.subplot(142)
    sns.boxplot(data=df_EMD, *args, **kwargs)
    plt.title('EMD')
    plt.subplot(143)
    sns.boxplot(data=df_KLD, *args, **kwargs)
    plt.title('KLD')
    plt.subplot(144)
    sns.boxplot(data=df_KST, *args, **kwargs)
    plt.title('KST')

    return plt


def boxplot_lab_volumes(df):
    laB_volume = df.pivot(index=None, columns='protocol', values='LABvolume')
    sns.boxplot(data=laB_volume)
    plt.ylim((0, 2000))
    return plt


def compareMeanHistograms(df, kde_kernel=None):
    mtrx_hists = df['avgHistogram'].as_matrix()
    metric = []
    for i in range(len(mtrx_hists)):
        for j in range(i + 1, len(mtrx_hists)):
            histA = mtrx_hists[i]
            histB = mtrx_hists[j]
            mR = computeMetrics(histA[:, 0], histB[:, 0], kde_kernel)
            mG = computeMetrics(histA[:, 1], histB[:, 1], kde_kernel)
            mB = computeMetrics(histA[:, 2], histB[:, 2], kde_kernel)
            metric.append(np.sum([mR, mG, mB], axis=0))

    return pd.DataFrame(data=metric, columns=['SSD', 'EMD', 'KLD', 'KST'])


def plotHistograms_(df):
    """
    Visualize the histograms in df channelwise for manual inspection. Histograms use the global kde_kernel and are
    color coded by their patch number.
    :param df: subset of the data as pandas.DataFrame (e.g. limited to a single block)
    :return: None
    """
    fig, axes = plt.subplots(1, 3, sharex=True, figsize=(10, 5))
    sns.despine(left=True)

    hists = df['histogram'].as_matrix()
    colorIdx = df['patch'].as_matrix()

    for idx, hList in enumerate(hists):
        hist = np.asarray(hList)
        axes[0].plot(np.convolve(hist[:, 0], kde_kernel), color=protoColors[colorIdx[idx]])
        axes[0].set_xlim((0, 255))
        axes[0].set_title('R-Channel')
        axes[1].plot(np.convolve(hist[:, 1], kde_kernel), color=protoColors[colorIdx[idx]])
        axes[1].set_xlim((0, 255))
        axes[1].set_title('G-Channel')
        axes[2].plot(np.convolve(hist[:, 2], kde_kernel), color=protoColors[colorIdx[idx]])
        axes[2].set_xlim((0, 255))
        axes[2].set_title('B-Channel')
        # plt.show()


### Helper methods related to I/O ###

def require_dir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    return dirname

def unpack_dirinfo(dirinfo):
    method_dir, multiprotocol = dirinfo[:2]
    file_ext = dirinfo[2] if len(dirinfo) > 2 else "tif"
    return method_dir, multiprotocol, file_ext


def experiment_iterator(cfg):
    """
    Iterate over experiments (methods and different protocols) given a config dictionary
    :param cfg: dict-like object containing protocols and output directory
    :yields: image directory to read from, filename
    """
    for method, dirinfo in cfg["method_directories"].items():
        method_dir, multiprotocol, file_ext = unpack_dirinfo(dirinfo)
        for protocol in (cfg["protocols"] if multiprotocol else ["baseline"]):
            fname = op.join(cfg["output_directory"], '{}_{}.gz'.format(method, protocol))
            image_dir = method_dir.format(protocol)
            yield image_dir, fname, file_ext


def get_plotname(cfg, name, method, protocol):
    ext = cfg["plot_format"]
    pltname = op.join(cfg["plot_directory"], "{}-{}-{}.{}".format(name, method, protocol, ext))
    return pltname


if __name__ == '__main__':
    import sys, os, json

    testrun = False

    ### +++ Load config, parse params +++ ###
    if len(sys.argv) == 1:
        configfile = "evaluation.json"
    else:
        configfile = sys.argv[1]

    with open(configfile, "r") as fp:
        cfg = json.load(fp)

    require_dir(cfg["output_directory"])
    require_dir(cfg["plot_directory"])

    ### +++ Data Loading +++ ###
    # Loop over all methods and protocols and create histograms
    src_dir, _ = cfg["method_directories"]["unnormalized"]
    for image_dir, fname, file_ext in experiment_iterator(cfg):
        if testrun:
            print("Load {} and write to {}".format(image_dir, fname))
            assert os.path.exists(image_dir), "Directory '{}' does not exist!".format(image_dir)
            continue

        if not os.path.exists(fname):
            data = readData(image_dir, srcDir=(src_dir if image_dir != src_dir else None), ext=file_ext)
            jl.dump(data, fname, compress=4, cache_size=128, protocol=-1)

    ### +++ Evaluation Routines +++ ###
    # Compute metrics on histograms

    metrics_prdf = OrderedDict()
    metrics_ssim = OrderedDict()

    for protocol_id, protocol in enumerate(cfg["protocols"]):
        prdf = {}
        ptch = {}
        avg = {}
        hists = {}
        ssim = {}
        plot_opts = cfg["plot_options"]
        print("Read data: " + protocol)
        for method, dirinfo in cfg["method_directories"].items():
            method_dir, multiprotocol, file_ext = unpack_dirinfo(dirinfo)

            fname = op.join(cfg["output_directory"],
                            '{}_{}.gz'.format(method, protocol if multiprotocol else "baseline"))
            if not os.path.exists(fname):
                continue

            data = jl.load(fname)
            #data = data.loc[data.protocol != protocol_id+1]
            if not cfg["include_thickness"]:
                data = data.loc[data.protocol > 2] # also exclude T+ protocol
            prdf[method] = compareProtocolwise(data,kde_kernel=kde_kernel)
            if plot_opts["patchwise"]:
                ptch[method] = comparePatchwise(data,kde_kernel=kde_kernel)
            # avg[method]  = compareMeanHistograms(data) # "avgHistogram" key is not provided currently
            hists[method] = data.loc[data.block == 74235]
            if multiprotocol:
                ssim[method] = data


        metrics_prdf[protocol] = prdf
        metrics_ssim[protocol] = ssim

        #break

        # Plotting
        print("Plotting: " + protocol)
        if plot_opts["protocolwise"]:
            set_style()
            plt = boxplot(prdf["unnormalized"], prdf["reinhard"], prdf["marcenko"], prdf["kahn"], prdf["lstm_reworked"])
            sns.despine()
            plt.savefig(get_plotname(cfg,"boxplot-protocolwise", "comparison", protocol),
                    bbox_inches="tight")
            plt.close()

        if plot_opts["patchwise"]:
            set_style()
            plt = boxplot(ptch["unnormalized"], ptch["reinhard"], ptch["marcenko"], ptch["kahn"],
                    ptch["lstm_reworked"], showfliers=False)
            plt.savefig(get_plotname(cfg,"boxplot-patchwise", "comparison", protocol),
                    bbox_inches="tight")
            sns.despine()
            plt.close()

        if plot_opts["ssim"]:
            set_style()
            plt = boxplot_similarity(ssim)
            pltname = op.join(cfg["plot_directory"], "boxplot-ssim-{}.pdf".format(protocol))
            plt.savefig(get_plotname(cfg,"ssim-lab", "comparison", protocol), bbox_inches="tight")
            sns.despine()
            plt.close()

        if plot_opts["histograms"]:
            for method, _ in cfg["method_directories"].items():
                set_style()
                plotHistograms_(hists[method])
                plt.savefig(get_plotname(cfg,"histogram", method, protocol), bbox_inches="tight")
                plt.close()

    if plot_opts["scatter"]:
        set_style()
        plt = scatterplot(metrics_prdf)
        pltname = op.join(cfg["plot_directory"], "scatterplot-metrics.pdf")
        plt.savefig(pltname, bbox_inches="tight")
        sns.despine()
        plt.close()

        set_style()
        plt = scatterplot(metrics_ssim, metrics = ['ssim', 'lab_volume'])
        pltname = op.join(cfg["plot_directory"], "scatterplot-ssim-lab.pdf")
        plt.savefig(pltname, bbox_inches="tight")
        sns.despine()
        plt.close()
