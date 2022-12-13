import re
import glob
import threading

import matplotlib.pyplot as plt
import numpy as np
import cv2

from skimage import io
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from sklearn.neighbors import KNeighborsClassifier


def convertRegionsToRocks(h,w,numToDo, start, segments_slic, img, model):
    
    for i in range(numToDo):
        mask = segments_slic == i + start
        validPixels = img[mask]
        avg = np.mean(validPixels, axis = 0)
        
        closestFile = model.predict(np.asarray(avg).reshape(1,-1))
        
        closestRock = io.imread(closestFile[0])
        closestRock = closestRock[:,:,0:3]
        
        avgRock = closestRock.mean(axis=0).mean(axis=0)
        
        mul = avg/avgRock
        
        rows, cols = np.nonzero(mask)
        
        startRow = rows.min()
        startCol = cols.min()
        for r in range(min(40, h - startRow)):
            for c in range(min(40, w - startCol)):
                img[startRow+r, startCol+c] = mul*closestRock[r,c] if mask[startRow+r, startCol+c] else img[startRow+r, startCol+c]

def convertToRocks(img, model):
    imgCopy = np.copy(img).astype('int8')
    imgAsFloat = img_as_float(img)

    segments_slic = slic(imgAsFloat, n_segments=15000, compactness=3, sigma=1,
                        start_label=0)

    numRegions = len(np.unique(segments_slic))
    
    h, w, _ = img.shape

    numthreads = 45
    threads = []
    
    for i in range(numthreads):
        start = int(i * numRegions/numthreads)
        stop = int((i+1) * numRegions/numthreads)
        numToDo = stop - start
        newT = threading.Thread(target=convertRegionsToRocks, args=(h,w,numToDo, start, segments_slic, img, model))
        threads.append(newT)
    
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    
    cv2.imwrite('img.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

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
    model = createModel('rgbImages')
    
    img = io.imread('obama2.jpg')
    img = img[:,:,0:3]
    
    convertToRocks(img, model)