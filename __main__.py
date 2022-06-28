import os
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import bagOfFeatures.segmentation as segmentation

def main():
    listOfSegmentedImages = list()
    listOfGTPs = list()

    s1 = segmentation.Segmentation()


    #_, _, files = next(os.walk("C:\Users\Luca\IdeaProjects\WordspottingS\GT"))
    gtpfiles = os.listdir(r"\Users\Luca\IdeaProjects\WordspottingS\GT")
    pngfiles = os.listdir(r"C:\Users\Luca\IdeaProjects\WordspottingS\pages")
    nrOfItems = len(gtpfiles)

    ''' segmentiere Bilder 
    #listOfSegmentedImages beinhaltet geordnet nach Dokument tupel mit je einem segmentierten Wort + repräsentiertem Wort
    '''
    for i in range (nrOfItems):
        gtpPath = r"\Users\Luca\IdeaProjects\WordspottingS\GT\\" +   gtpfiles[i]
        pngPath = r"\Users\Luca\IdeaProjects\WordspottingS\pages\\" + pngfiles[i]

        pngImage = Image.open(pngPath,"r")
        listOfSegmentedImages.append(s1.segmentCut(pngImage,gtpPath))

    '''
    hier können BoF Repräsentationen generiert werden
    '''

    '''
    hier wird jedes Wortbild als Eingabe ausgewählt, um anhand der BoFs ähnliche Worte zu identifizieren
    '''



    '''
    Berechnung der Mean Average Precision
    '''


    print(nrOfItems)


if __name__ == "__main__":
    main()
