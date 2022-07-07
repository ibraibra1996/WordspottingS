# __pipeline__
# 1. Input picture
#   - build local descriptors
#   - cluster local descriptors
#   - find centruids
#   - build descriptive vector with centruids

# 2. Check distance to other descriptive vector from "Database"
#   - make descending list of pictures based on distance

##Helper Funktions

import os
import re
import PIL.Image as Image
import numpy as np
import scipy
from matplotlib import pyplot as plt, cm

from bagOfFeatures import segmentation
from common import features
from scipy.cluster.vq import kmeans2
import scipy.spatial.distance as ds

from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D


class ImagePrep(object):

    @staticmethod
    def bof(documentSegmentation='2700270.gtp',
            documentImage='2700270.png',
            step_size=5,
            cell_size=3,
            step_size_doc=15,
            cell_size_doc=3,
            n_centroids=250,
            iter=20):

        documentSegmentation = os.path.join('GT', documentSegmentation)
        documentImage = os.path.join('pages', documentImage)
        image = Image.open(documentImage)
        imageArray = np.array(image)

        frames, desc = features.compute_sift_descriptors(imageArray,
                                                         step_size=step_size_doc,
                                                         cell_size=cell_size_doc)

        centroids, labels = kmeans2(desc, n_centroids, iter=iter, minit='points')

        wordImageList = segmentation.Segmentation.segmentCut(image, documentSegmentation)

        # init.
        histograms = []
        namesOfWords = []
        # For all Images:
        # - Change Color scale to gray
        # - build SIFT representation
        for word in wordImageList:
            image = np.array(word[0])
            frames, desc = features.compute_sift_descriptors(image,
                                                             step_size=step_size,
                                                             cell_size=cell_size)

            distances = ds.cdist(desc, centroids, metric='euclidean')

            sortiert2d = np.argsort(distances, axis=1)

            hist = np.bincount(sortiert2d[:, 0], minlength=len(centroids))

            histograms.append(hist)

            namesOfWords.append(word[1])

        return histograms, namesOfWords

    @staticmethod
    def bof1(documentSegmentation='2700270.gtp',
             documentImage='2700270.png',
             step_size=5,
             cell_size=3,
             step_size_doc=15,
             cell_size_doc=3,
             n_centroids=250,
             iter=20):

        documentSegmentation = os.path.join('GT', documentSegmentation)
        documentImage = os.path.join('pages', documentImage)
        image = Image.open(documentImage)
        imageArray = np.array(image)

        frames, desc = features.compute_sift_descriptors(imageArray,
                                                         step_size=step_size_doc,
                                                         cell_size=cell_size_doc)

        centroids, labels = kmeans2(desc, n_centroids, iter=iter, minit='points')

        wordImageList = segmentation.Segmentation.segmentCut(image, documentSegmentation)

        # init.
        histograms = []
        namesOfWords = []
        # For all Images:
        # - Change Color scale to gray
        # - build SIFT representation
        for word in wordImageList:
            image = np.array(word[0])
            frames, desc = features.compute_sift_descriptors(image,
                                                             step_size=step_size,
                                                             cell_size=cell_size)

            histograms.append(ImagePrep.histogramms(desc, centroids))
            namesOfWords.append(word[1])

        return histograms, namesOfWords

    @staticmethod
    def histogramms(desc, centriod):
        n = len(centriod)
        Histogram = [0] * n
        for i in desc:
            distans = scipy.spatial.distance.cdist(centriod, [i], metric='euclidean')
            sortiert = np.argsort(distans, axis=0)
            index = sortiert[0, 0]
            x = Histogram[index] + 1
            Histogram[index] = x
        return Histogram

    @staticmethod
    def resultListForPrecision(histograms,
                               namesOfWords,
                               queryIndex,
                               ):

        queryWordHistogram = np.reshape(histograms[queryIndex],
                                        (-1, len(histograms[queryIndex])))

        queryWordString = namesOfWords[queryIndex]

        distances = ds.cdist(histograms, queryWordHistogram, metric='euclidean')

        sortiert2d = np.argsort(distances, axis=0)

        sortiert1d = sortiert2d.ravel()

        namesOfwordsSorted = np.array(namesOfWords)[sortiert1d]

        EntriesWith01 = np.where(namesOfwordsSorted == queryWordString, 1, 0)

        return EntriesWith01[1:]
