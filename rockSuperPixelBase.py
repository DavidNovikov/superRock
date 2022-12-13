import re
import glob
import threading

import numpy as np
import cv2

from skimage import io
from skimage.segmentation import slic
from skimage.util import img_as_float
from sklearn.neighbors import KNeighborsClassifier

def convertRegionsToRocks(numToDo, start, segments_slic, img, model):
    
    for i in range(numToDo):
        mask = segments_slic == i + start
        validPixels = img[mask]
        avg = np.mean(validPixels, axis = 0)
        
        closestFile = model.predict(avg.reshape(1,-1))
        
        closestRock = io.imread(closestFile[0])
        
        indexes = np.where(mask)
        rows = indexes[0]
        cols = indexes[1]
        
        startRow = rows.min()
        startCol = cols.min()
        
        for i in range(len(rows)):
            row = rows[i]
            col = cols[i]
            img[row,col] = closestRock[row - startRow,col-startCol]

def convertToRocks(img, model):
    imgAsFloat = img_as_float(img)

    segments_slic = slic(imgAsFloat, n_segments=50000, compactness=3, sigma=1,
                        start_label=0)

    numRegions = len(np.unique(segments_slic))

    numthreads = 45
    threads = []
    
    for i in range(numthreads):
        start = int(i * numRegions/numthreads)
        stop = int((i+1) * numRegions/numthreads)
        numToDo = stop - start
        newT = threading.Thread(target=convertRegionsToRocks, args=(numToDo, start, segments_slic, img, model))
        threads.append(newT)
    
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    
    # cv2.imwrite('img4.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite('img4.png', img)

def createModel(src):
    files = glob.glob(f'{src}/*')
    rgbValues = []
    for file in files:
        numsAsStrings = re.findall("\d+\.\d+", file)
        vals = [float(x) for x in numsAsStrings]
        rgbValues.append(vals)
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(rgbValues, files)
    
    return model
        
        
    
if __name__ == "__main__":
    model = createModel('largeRGBImages')
    
    img = io.imread('dreeseSmall.jpg')
    img = img[:,:,0:3]
    
    convertToRocks(img, model)