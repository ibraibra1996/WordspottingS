import os
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

class Segmentation(object):

    @staticmethod
    def segmentCut(text, segments):
        segmentFile=open(segments, 'r')

        lines = segmentFile.readlines()
        listOfWordPics = list()

        for line in lines:
            left0, top0, right0, bottom0, content = line.split(" ")
            left=int(left0)
            right=int(right0)
            top=int(top0)
            bottom=int(bottom0)

            word=text.crop((left, top, right, bottom))
            listOfWordPics.append((word,content))

        return listOfWordPics

    @staticmethod
    def averagePrecision(binlist):
        sum=0
        numberOfOnes=0
        for i in range(len(binlist)):
            if binlist[i]==1:
                numberOfOnes+=1
                sum+=numberOfOnes/(i+1)
        ret = sum/numberOfOnes
        '''
        if(ret < 0.4):
            print(binlist)
            print(ret)
        '''


        return ret



