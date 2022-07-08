import itertools
import os
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.cluster.vq import kmeans2

import bagOfFeatures.segmentation as segmentation
from bagOfFeatures.bofFunctions import ImagePrep
from common import features


def main(dataNames,
         step_size,
         cell_size,
         n_centroids,
         step_size_doc,
         cell_size_doc,
         iter):
    totalAvgAllDocs = 0

    for dataName in dataNames:
        histograms, namesOfWords = ImagePrep.bof(documentSegmentation=dataName + '.gtp',
                                                 documentImage=dataName + '.png',
                                                 step_size=step_size,
                                                 cell_size=cell_size,
                                                 step_size_doc=step_size_doc,
                                                 cell_size_doc=cell_size_doc,
                                                 n_centroids=n_centroids,
                                                 iter=iter)

        # hier kann eine Anfrage gemacht werden
        EntriesWith01, Ruckgabeliste = ImagePrep.resultListForPrecision(histograms=histograms,
                                                                        namesOfWords=namesOfWords,
                                                                        queryIndex=33)
        print(EntriesWith01)
        print(Ruckgabeliste)

        precisionForAllWords = ImagePrep.resultListForPrecisionForAllWords(histograms=histograms,
                                                                           namesOfWords=namesOfWords)

        # print(precisionForAllWords[1])

        totalAvg = 0
        for n in precisionForAllWords:
            totalAvg += segmentation.Segmentation.averagePrecision(binlist=n)

        totalAvg = totalAvg / len(precisionForAllWords)

        print(f'totalAvg für {dataName}:{totalAvg}')
        totalAvgAllDocs += totalAvg

    # print(f'totalAvgAllDocs:{totalAvgAllDocs / len(dataNames)}')
    print('_____________')


if __name__ == "__main__":
    dataNames = [str(name) + "0" + str(name) for name in range(270, 280)] + [str(name) + "0" + str(name) for name in
                                                                             range(300, 310)]
    n_centroidsList = [i for i in range(25, 500, 25)]

    main(dataNames=dataNames[10:11],
         step_size=1,  # 2
         cell_size=9,
         step_size_doc=15,
         cell_size_doc=3,
         n_centroids=250,  #

         iter=20)
