import os
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import bagOfFeatures.segmentation as segmentation
from bagOfFeatures.bofFunctions import ImagePrep


def main():
    listOfSegmentedImages = list()
    listOfGTPs = list()

    s1 = segmentation.Segmentation()

    # _, _, files = next(os.walk("C:\Users\Luca\IdeaProjects\WordspottingS\GT"))
    gtpfiles = os.listdir(r"\Users\Luca\IdeaProjects\WordspottingS\GT")
    pngfiles = os.listdir(r"\Users\Luca\IdeaProjects\WordspottingS\pages")
    nrOfItems = len(gtpfiles)

    # later used for calculation of average
    totalWordImages = 0

    ''' segmentiere Bilder 
    #listOfSegmentedImages beinhaltet geordnet nach Dokument tupel mit je einem segmentierten Wort + repräsentiertem Wort
    '''

    for i in range(nrOfItems):
        gtpPath = r"\Users\Luca\IdeaProjects\WordspottingS\GT\\" + gtpfiles[i]
        pngPath = r"\Users\Luca\IdeaProjects\WordspottingS\pages\\" + pngfiles[i]

        pngImage = Image.open(pngPath, "r")
        segmentList = s1.segmentCut(pngImage, gtpPath)
        listOfSegmentedImages.append(segmentList)
        totalWordImages += len(segmentList)

    listOfAllWords = [inner for outer in listOfSegmentedImages for inner in outer]
    print(len(listOfAllWords))
    #print(listOfAllWords)
    '''
    hier können BoF Repräsentationen generiert werden
    '''
    documentSegmentation = os.path.join('GT', '3010301.gtp')
    documentImage = os.path.join('pages', '3010301.png')
    image = Image.open(documentImage)

    wordImageList = segmentation.Segmentation.segmentCut(image, documentSegmentation)
    print(wordImageList)

    # plt.figure()

    # wordImageList[0][0].show()

    histograms, namesOfWords = ImagePrep.bof(wordImageList=wordImageList)
    #histograms, namesOfWords = ImagePrep.bof(wordImageList=listOfAllWords)

    precisionForAllWords = []

    for i in range(len(namesOfWords)):
        precision = ImagePrep.resultListForPrecision(histograms=histograms,
                                                     namesOfwords=namesOfWords,
                                                     queryWordHistogram=histograms[i],
                                                     queryWordString=namesOfWords[i])
        precisionForAllWords.append(precision)

    '''
    hier wird jedes Wortbild als Eingabe ausgewählt, um anhand der BoFs ähnliche Worte zu identifizieren
    '''
    #av = segmentation.Segmentation.averagePrecision(binlist=precisionForAllWords[4])
    print(len(precisionForAllWords))
    print(precisionForAllWords[4])
    '''
    Berechnung der Mean Average Precision
    '''
    totalAvg = 0
    for n in precisionForAllWords:
        totalAvg += segmentation.Segmentation.averagePrecision(binlist=n)

    totalAvg = totalAvg/len(precisionForAllWords)

    print(totalAvg)

if __name__ == "__main__":
    main()
