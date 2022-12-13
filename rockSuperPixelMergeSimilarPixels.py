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

def calculateStats(numToDo, start, segments_slic, img, avgColorInRegion, numPixelsInRegion, regionRows, regionCols):
    
    for i in range(numToDo):
        mask = segments_slic == i + start
        validPixels = img[mask]
        avg = np.mean(validPixels, axis = 0)
        tot = len(validPixels)
        
        indexes = np.where(mask)
        rows = indexes[0]
        cols = indexes[1]
        
        avgColorInRegion[i + start] = avg
        numPixelsInRegion[i + start] = tot
        regionRows[i + start] = np.unique(rows)
        regionCols[i + start] = np.unique(cols)

def convertRegionsToRocks(regionNums, segments_slic, img, models, regionRows, regionCols):
    
    for i in range(len(regionNums)):
        mask = segments_slic == regionNums[i]
        validPixels = img[mask]
        avg = np.mean(validPixels, axis = 0)
        
        avg = avg / 255
        
        model = None
        
        numRows = regionRows[regionNums[i]].__len__()
        numCols = regionCols[regionNums[i]].__len__()
        
        if numRows <= 40 and numCols <= 40:
            model = models[2]
        elif numRows <= 200 and numCols <= 200:
            model = models[1]
        else:
            model = models[0]
        
        closestFile = model.predict(avg.reshape(1,-1))
        closestRock = io.imread(closestFile[0])
        
        indexes = np.where(mask)
        rows = indexes[0]
        cols = indexes[1]
        
        startRow = rows.min()
        startCol = cols.min()
        
        for j in range(len(rows)):
            row = rows[j]
            col = cols[j]
            img[row,col] = closestRock[row - startRow,col-startCol]

def calculateStatsOwner(numRegions, segments_slic, img, avgColorInRegion, numPixelsInRegion, regionRows, regionCols):
    numthreads = 45
    threads = []
    
    for i in range(numthreads):
        start = int(i * numRegions/numthreads)
        stop = int((i+1) * numRegions/numthreads)
        numToDo = stop - start
        newT = threading.Thread(target=calculateStats, args=(numToDo, start, segments_slic, img, avgColorInRegion, numPixelsInRegion, regionRows, regionCols))
        threads.append(newT)
    
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
        
def convertRegionsToRocksOwner(validRegionNumbers, segments_slic, img, models, regionRows, regionCols):
    numRegions = len(validRegionNumbers)
    numthreads = 45
    threads = []
    
    for i in range(numthreads):
        start = int(i * numRegions/numthreads)
        stop = int((i+1) * numRegions/numthreads)
        newT = threading.Thread(target=convertRegionsToRocks, args=(validRegionNumbers[start:stop], segments_slic, img, models, regionRows, regionCols))
        threads.append(newT)
    
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
        
def rowsAndColsLessThan(rows1, cols1, rows2, cols2, cnt):
    numRows = np.unique(np.concatenate((rows1, rows2))).__len__()
    numCols = np.unique(np.concatenate((cols1, cols2))).__len__()
    return numRows < cnt and numCols < cnt

def avgDiffLessThan(avgColorInRegion, baseReg, nextReg, cnt):
    return sum(abs(avgColorInRegion[baseReg] - avgColorInRegion[nextReg])) < cnt

def canMerge(baseReg, nextReg, avgColorInRegion, regionRows, regionCols):
    if baseReg == nextReg:
        return False
    if not avgDiffLessThan(avgColorInRegion, baseReg, nextReg, 9):
        return False
    return rowsAndColsLessThan(regionRows[baseReg], regionCols[baseReg], regionRows[nextReg], regionCols[nextReg], 450)

def convertToRocks(img, models):
    imgAsFloat = img_as_float(img)
    
    segments_slic = slic(imgAsFloat, n_segments=100000, compactness=0.5, sigma=1,
                        start_label=0)

    numRegions = len(np.unique(segments_slic))
    
    avgColorInRegion = {}
    numPixelsInRegion = {}
    regionRows = {}
    regionCols = {}

    calculateStatsOwner(numRegions, segments_slic, img, avgColorInRegion, numPixelsInRegion, regionRows, regionCols)
    
    h, w, _ = img.shape
    
    for row in range(h-1):
        for col in range(w-1):
            
            baseReg = segments_slic[row,col]
            nextReg = segments_slic[row,col+1]
            
            if canMerge(baseReg, nextReg, avgColorInRegion, regionRows, regionCols):
                segments_slic[segments_slic == nextReg] = baseReg
                
                baseAvg = avgColorInRegion[baseReg]
                baseNumPixels = numPixelsInRegion[baseReg]
                
                nextAvg = avgColorInRegion[nextReg]
                nextNumPixels = numPixelsInRegion[nextReg]
                
                avgColorInRegion[baseReg] = (baseAvg*baseNumPixels + nextAvg*nextNumPixels)/(baseNumPixels + nextNumPixels)
                numPixelsInRegion[baseReg] = baseNumPixels + nextNumPixels
                regionRows[baseReg] = np.unique(np.concatenate((regionRows[baseReg], regionRows[nextReg])))
                regionCols[baseReg] = np.unique(np.concatenate((regionCols[baseReg], regionCols[nextReg])))
                
                avgColorInRegion.pop(nextReg)
                numPixelsInRegion.pop(nextReg)
                regionRows.pop(nextReg)
                regionCols.pop(nextReg)
    
    for col in range(w-1):
        for row in range(h-1):
            
            baseReg = segments_slic[row,col]
            nextReg = segments_slic[row+1,col]
            
            if canMerge(baseReg, nextReg, avgColorInRegion, regionRows, regionCols):
                segments_slic[segments_slic == nextReg] = baseReg
                
                baseAvg = avgColorInRegion[baseReg]
                baseNumPixels = numPixelsInRegion[baseReg]
                
                nextAvg = avgColorInRegion[nextReg]
                nextNumPixels = numPixelsInRegion[nextReg]
                
                avgColorInRegion[baseReg] = (baseAvg*baseNumPixels + nextAvg*nextNumPixels)/(baseNumPixels + nextNumPixels)
                numPixelsInRegion[baseReg] = baseNumPixels + nextNumPixels
                regionRows[baseReg] = np.unique(np.concatenate((regionRows[baseReg], regionRows[nextReg])))
                regionCols[baseReg] = np.unique(np.concatenate((regionCols[baseReg], regionCols[nextReg])))
                
                avgColorInRegion.pop(nextReg)
                numPixelsInRegion.pop(nextReg)
                regionRows.pop(nextReg)
                regionCols.pop(nextReg)
    
    assert avgColorInRegion.keys() == numPixelsInRegion.keys()
    
    validRegionNumbers = list(avgColorInRegion.keys())
    
    # plt.imshow(mark_boundaries(img, segments_slic))

    convertRegionsToRocksOwner(validRegionNumbers, segments_slic, img, models, regionRows, regionCols)
    
    # cv2.imwrite('img3.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.imwrite('img21.png', img)

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
    modelL = createModel('largeRGBImagesProp2')
    modelM = createModel('mediumRGBImagesProp')
    modelS = createModel('smallRGBImagesProp')
    
    img = io.imread('david.jpg')
    img = img[:,:,0:3]
    
    convertToRocks(img, [modelL, modelM, modelS])