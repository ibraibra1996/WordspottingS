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


def main(dataName,
         step_size,
         cell_size,
         n_centroids,
         step_size_doc,
         cell_size_doc,
         iter):
    histograms, namesOfWords = ImagePrep.bof(documentSegmentation=dataName+'.gtp',
                                             documentImage=dataName+'.png',
                                             step_size=step_size,
                                             cell_size=cell_size,
                                             step_size_doc=step_size_doc,
                                             cell_size_doc=cell_size_doc,
                                             n_centroids=n_centroids,
                                             iter=iter)

    # hier kann eine Anfrage gemacht werden
    # EntriesWith01, Ruckgabeliste = ImagePrep.resultListForPrecision(histograms=histograms,
    #                                                                namesOfWords=namesOfWords,
    #                                                               queryIndex=33)
    # print(EntriesWith01)
    # print(Ruckgabeliste)

    precisionForAllWords = ImagePrep.resultListForPrecisionForAllWords(histograms=histograms,
                                                                       namesOfWords=namesOfWords)

    # print(precisionForAllWords[1])

    totalAvg = 0
    for n in precisionForAllWords:
        totalAvg += segmentation.Segmentation.averagePrecision(binlist=n)

    totalAvg = totalAvg / len(precisionForAllWords)

    print(f'totalAvg für {dataName}:{totalAvg}')

    print('_____________')
    return totalAvg


if __name__ == "__main__":
    dataNames = [str(name) + "0" + str(name) for name in range(270, 280)] + [str(name) + "0" + str(name) for name in
                                                                             range(300, 310)]

    n_centroidsList = [i for i in range(100, 300, 50)]  # 25, 500, 25

    step_sizeList = [i for i in range(2, 10, 2)]  # 1, 10, 1
    cell_sizeList = [i for i in range(3, 10, 2)]  # 3, 10, 1

    configurationList = [n_centroidsList,
                         step_sizeList,
                         cell_sizeList,
                         dataNames]

    configurationListProduct = []
    configurationListProduct.append(['n_centroids','step_size','cell_size','totalAvg','dataName'])

    for n_centroids, step_size, cell_size,dataName in itertools.product(*configurationList):
        totalAvg = main(dataName=dataName,
                        step_size=step_size,  # 2  1
                        cell_size=cell_size,  # 3   9
                        step_size_doc=15,
                        cell_size_doc=3,
                        n_centroids=n_centroids,  # 250
                        iter=20)

        configurationListProduct.append([n_centroids,step_size,cell_size,totalAvg,dataName])

    import numpy as np

    np.savetxt("ResultData.csv", configurationListProduct, delimiter=";", fmt='% s')

    print('csv fertig')
