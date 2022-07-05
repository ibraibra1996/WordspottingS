import os
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.cluster.vq import kmeans2

import bagOfFeatures.segmentation as segmentation
from bagOfFeatures.bofFunctions import ImagePrep
from common import features


def main():
    listOfSegmentedImages = list()
    listOfGTPs = list()

    '''
    hier können BoF Repräsentationen generiert werden
    '''
    documentSegmentation = os.path.join('GT', '2700270.gtp')
    documentImage = os.path.join('pages', '2700270.png')
    image = Image.open(documentImage)

    imageArray = np.array(image)

    #histogramDocument = ImagePrep.bof([(image, "2700270")], n_centroids=20)
    #print(histogramDocument)

    #image = np.array(word[0])
    step_size = 15
    cell_size = 3
    n_centroids=20
    iter=20
    frames, desc = features.compute_sift_descriptors(imageArray, step_size=step_size, cell_size=cell_size)
    # clustering of SIFT descriptors
    centroid, labels = kmeans2(desc, n_centroids, iter=iter, minit='points')
    print(f"centroid {len(centroid[0])}")

    wordImageList = segmentation.Segmentation.segmentCut(image, documentSegmentation)
    #print(wordImageList)


    frameList = []
    descList = []
    for word in wordImageList:
        image = np.array(word[0])
        frames, desc = features.compute_sift_descriptors(image, step_size=step_size, cell_size=cell_size)
        frameList.append(frames)
        descList.append(desc)

    centroid, labels = kmeans2(descList[0], n_centroids, iter=iter, minit='points')
    print(len(descList[0]))
    # plt.figure()

    # wordImageList[0][0].show()

    # histograms, namesOfWords = ImagePrep.bof(wordImageList=wordImageList)
    # histograms, namesOfWords = ImagePrep.bof(wordImageList=listOfAllWords)

    precisionForAllWords = []
    '''
    
    for i in range(len(namesOfWords)):
        precision = ImagePrep.resultListForPrecision(histograms=histograms,
                                                     namesOfwords=namesOfWords,
                                                     queryWordHistogram=histograms[i],
                                                     queryWordString=namesOfWords[i])
        precisionForAllWords.append(precision)
    '''
    '''
    hier wird jedes Wortbild als Eingabe ausgewählt, um anhand der BoFs ähnliche Worte zu identifizieren
    '''
    # av = segmentation.Segmentation.averagePrecision(binlist=precisionForAllWords[4])
    #print(len(precisionForAllWords))
    #print(precisionForAllWords[4])
    '''
    Berechnung der Mean Average Precision
        totalAvg = 0
    for n in precisionForAllWords:
        totalAvg += segmentation.Segmentation.averagePrecision(binlist=n)

    totalAvg = totalAvg/len(precisionForAllWords)

    print(totalAvg)
    
    '''


if __name__ == "__main__":
    main()
