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
from common import features
from scipy.cluster.vq import kmeans2
import scipy.spatial.distance as ds


class ImagePrep(object):

    @staticmethod
    def bof(wordImageList, step_size=15, cell_size=3, n_centroids=4, iter=20):
        # init.
        histograms = []
        namesOfWords = []
        # For all Images:
        # - Change Color scale to gray
        # - build SIFT representation
        for word in wordImageList:
            image = np.array(word[0])
            frames, desc = features.compute_sift_descriptors(image, step_size=step_size, cell_size=cell_size)

            # Open CV nutzen

            # clustering of SIFT descriptors
            centroid, labels = kmeans2(desc, n_centroids, iter=iter, minit='points')
            hist = np.bincount(labels)
            histograms.append(hist.tolist())
            namesOfWords.append(word[1])
        return histograms, namesOfWords

    @staticmethod
    def resultListForPrecision(histograms, namesOfwords, queryWordHistogram, queryWordString, metric='euclidean'):
        distances = ds.cdist(histograms, [queryWordHistogram], metric=metric)

        sortiert2d = np.argsort(distances, axis=0)

        sortiert1d = sortiert2d.ravel()

        namesOfwordsSorted = np.array(namesOfwords)[sortiert1d]

        EntriesWith01 = np.where(namesOfwordsSorted == queryWordString, 1, 0)

        return EntriesWith01[1:]


"""
       def getFilterMatrix():
        a1 = np.array([-1,0,1]) #Zeilen-Vektor
        a2 = np.array([1,1,1])

        a3 = np.array([[-1,0,1]]).T #Spalten-Vektor
        a4 = np.array([[1,1,1]]).T

        Sv= a1*a4.T #vertical
        Sh= a2*a3.T #horizontal
        return Sv, Sh

           def get_image(folderName, imageName):
        document_image_filename = os.path.join(folderName, imageName)
        image = Image.open(document_image_filename)
        im_arr = np.asarray(image, dtype='float32')
        return im_arr
   """
