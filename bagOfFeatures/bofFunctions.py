#__pipeline__
#1. Input picture
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

class ImagePrep(object):
   
    def getFilterMatrix():
        a1 = np.array([-1,0,1]) #Zeilen-Vektor
        a2 = np.array([1,1,1])

        a3 = np.array([[-1,0,1]]).T #Spalten-Vektor
        a4 = np.array([[1,1,1]]).T

        Sv= a1*a4.T #vertical
        Sh= a2*a3.T #horizontal
        return Sv, Sh


    #load image and convert it to float32
    def get_image(folderName, imageName):
        document_image_filename = os.path.join(folderName, imageName)
        image = Image.open(document_image_filename)
        im_arr = np.asarray(image, dtype='float32')
        return im_arr