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

def calculateStats(numToDo, start, segments_slic, img, avgColorInRegion, numPixelsInRegion):
    
    for i in range(numToDo):
        mask = segments_slic == i + start
        validPixels = img[mask]
        avg = np.mean(validPixels, axis = 0)
        tot = len(validPixels)
        
        avgColorInRegion[i + start] = avg
        numPixelsInRegion[i + start] = tot

def convertRegionsToRocks(regionNums, segments_slic, img, model):
    
    for i in range(len(regionNums)):
        mask = segments_slic == regionNums[i]
        validPixels = img[mask]
        avg = np.mean(validPixels, axis = 0)
        
        avg = avg / 255
        
        closestFile = model.predict(avg.reshape(1,-1))
        closestRock = io.imread(closestFile[0])
        
        numsAsStrings = re.findall("\d+\.\d+", closestFile[0])
        vals = np.asarray([float(x) for x in numsAsStrings])
        mul = np.asarray([vals[i] / (avg[i] + (1 if avg[i] == 0 else 0)) for i in range(3)])
        
        indexes = np.where(mask)
        rows = indexes[0]
        cols = indexes[1]
        
        startRow = rows.min()
        startCol = cols.min()
        
        for j in range(len(rows)):
            row = rows[j]
            col = cols[j]
            # img[row,col] = closestRock[row - startRow,col-startCol] * mul
            img[row,col] = closestRock[row - startRow,col-startCol]

def calculateStatsOwner(numRegions, segments_slic, img, avgColorInRegion, numPixelsInRegion):
    numthreads = 45
    threads = []
    
    for i in range(numthreads):
        start = int(i * numRegions/numthreads)
        stop = int((i+1) * numRegions/numthreads)
        numToDo = stop - start
        newT = threading.Thread(target=calculateStats, args=(numToDo, start, segments_slic, img, avgColorInRegion, numPixelsInRegion))
        threads.append(newT)
    
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
        
def convertRegionsToRocksOwner(validRegionNumbers, segments_slic, img, model):
    numRegions = len(validRegionNumbers)
    numthreads = 45
    threads = []
    
    for i in range(numthreads):
        start = int(i * numRegions/numthreads)
        stop = int((i+1) * numRegions/numthreads)
        newT = threading.Thread(target=convertRegionsToRocks, args=(validRegionNumbers[start:stop], segments_slic, img, model))
        threads.append(newT)
    
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

def convertToRocks(img, model):
    imgAsFloat = img_as_float(img)

    segments_slic = slic(imgAsFloat, n_segments=100000, compactness=0.5, sigma=1,
                        start_label=0)

    numRegions = len(np.unique(segments_slic))
    
    avgColorInRegion = {}
    numPixelsInRegion = {}
    regionRowCount = {}
    regionColCount = {}

    calculateStatsOwner(numRegions, segments_slic, img, avgColorInRegion, numPixelsInRegion)
    
    h, w, _ = img.shape
    
    for row in range(h-1):
        for col in range(w-1):
            
            baseReg = segments_slic[row,col]
            nextReg = segments_slic[row+1,col+1]
            
            if baseReg != nextReg and sum(abs(avgColorInRegion[baseReg] - avgColorInRegion[nextReg])) < 12:
                segments_slic[segments_slic == nextReg] = baseReg
                
                baseAvg = avgColorInRegion[baseReg]
                baseNumPixels = numPixelsInRegion[baseReg]
                
                nextAvg = avgColorInRegion[nextReg]
                nextNumPixels = numPixelsInRegion[nextReg]
                
                avgColorInRegion[baseReg] = (baseAvg*baseNumPixels + nextAvg*nextNumPixels)/(baseNumPixels + nextNumPixels)
                numPixelsInRegion[baseReg] = baseNumPixels + nextNumPixels
                
                avgColorInRegion.pop(nextReg)
                numPixelsInRegion.pop(nextReg)
    
    for col in range(w-1):
        for row in range(h-1):
            
            baseReg = segments_slic[row,col]
            nextReg = segments_slic[row+1,col+1]
            
            if baseReg != nextReg and sum(abs(avgColorInRegion[baseReg] - avgColorInRegion[nextReg])) < 12:
                segments_slic[segments_slic == nextReg] = baseReg
                
                baseAvg = avgColorInRegion[baseReg]
                baseNumPixels = numPixelsInRegion[baseReg]
                
                nextAvg = avgColorInRegion[nextReg]
                nextNumPixels = numPixelsInRegion[nextReg]
                
                avgColorInRegion[baseReg] = (baseAvg*baseNumPixels + nextAvg*nextNumPixels)/(baseNumPixels + nextNumPixels)
                numPixelsInRegion[baseReg] = baseNumPixels + nextNumPixels
                
                avgColorInRegion.pop(nextReg)
                numPixelsInRegion.pop(nextReg)
    
    assert avgColorInRegion.keys() == numPixelsInRegion.keys()
    
    validRegionNumbers = list(avgColorInRegion.keys())
    
    
    # cv2.imwrite('img3.png', mark_boundaries(img, segments_slic))
    
    convertRegionsToRocksOwner(validRegionNumbers, segments_slic, img, model)
    
    # cv2.imwrite('img3.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.imwrite('img8.png', img)

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
    model = createModel('largeRGBImagesProp2')
    
    img = io.imread('numbers.jpg')
    img = img[:,:,0:3]
    
    convertToRocks(img, model)