import glob
import random
import re
from multiprocessing import Process
import os
import threading
from time import sleep
import urllib.request
import ssl
import cv2
from PIL import Image
import numpy as np


imgDir = 'images2'

def grabRangeThread(subset):
    c = 0
    for image in subset:
        try:
            urllib.request.urlretrieve(f'https://www.natural-stone-database.com/images/{image}.jpg', f"images/{image}.jpg")
        except:
            c = c + 1
    print(c)
    
def grabRangeThread2(subset):
    c = 0
    for image in subset:
        try:
            urllib.request.urlretrieve(image, f"images2/{image[63:-1]}.jpg")
        except:
            c = c + 1
    print(c)
    
def readImagesInFile(file):
    f = open(file)
    line = f.readline()
    positions = [line.start() for line in re.finditer('ISA', line)]
    fileNames = [line[p:p+12] for p in positions]
    return fileNames

def readImagesInFile2(file):
    f = open(file)
    line = f.readline()
    fileNames = []
    while line:
        if 'href="https:' in line:
            positions = [i for i, letter in enumerate(line) if letter == '"']
            fileNames.append(line[positions[2]+1:positions[3]])
        line = f.readline()
    return fileNames
    
def grabFromFile(file):
    fileNames = readImagesInFile2(file)
    numFiles = len(fileNames)
    threads = []
    threadC = 40
    for i in range(threadC):
        low = int(i * numFiles/threadC)
        high = int((i+1) * numFiles/threadC)
        
        subset = fileNames[low:high]
        
        newT = threading.Thread(target=grabRangeThread2, args=(subset,))
        threads.append(newT)
        
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

def grabFilesFromInternet():
    ssl._create_default_https_context = ssl._create_unverified_context
    src = 'txtFiles2'
    files = glob.glob(f'{src}/*')
    processes = []
    for file in files:
        p = Process(target=grabFromFile, args=(file,))
        processes.append(p)
        p.start()
        sleep(0.01)

    for process in processes:
        process.join()
        
def cropAndGenerateMiniImages():
    src = 'images2'
    files = glob.glob(f'{src}/*')
    src = 'croppedImages'
    files.extend(glob.glob(f'{src}/*'))
    numFiles = len(files)
    processes = []
    processCount = 40
    for i in range(processCount):
        low = int(i * numFiles/processCount)
        high = int((i+1) * numFiles/processCount)
        
        subset = files[low:high]
        p = Process(target=cropAndGenerateMiniImagesSubset, args=(subset,))
        processes.append(p)
        p.start()
        sleep(0.001)

    for process in processes:
        process.join()
        
def cropAndGenerateMiniImagesSubset(subset):
    cropSize = 200
    for imgPath in subset:
        img = cv2.imread(imgPath)
        h, w, _ = img.shape
        # if w == 500 and h == 500:
        for i in range(10):
            xStart = random.randint(0,w - cropSize)
            yStart = random.randint(0,h - cropSize)
            
            miniCrop = img[yStart:yStart + cropSize,xStart:xStart + cropSize,:]
            
            avg = miniCrop.mean(axis=0).mean(axis=0)
            
            # tot = sum(avg)
            
            avg = [i / 255 for i in avg]
            
            c1 = '%07.3f'%avg[0]
            c2 = '%07.3f'%avg[1]
            c3 = '%07.3f'%avg[2]
            
            cv2.imwrite(f'mediumRGBImagesProp/{c1}_{c2}_{c3}.png', miniCrop)
        # else:
        #     print(imgPath)
    
def findEdges():
    fileNames = os.listdir(imgDir)
    numFiles = len(fileNames)
    threads = []
    threadC = 40
    for i in range(threadC):
        low = int(i * numFiles/threadC)
        high = int((i+1) * numFiles/threadC)
        
        subset = fileNames[low:high]
        
        newT = threading.Thread(target=findEdgesSubset, args=(subset,))
        threads.append(newT)
        
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    
def findEdgesSubset(subset):
    for greyBorderImgPath in subset:
        
        greyBorderImg = cv2.imread(f'{imgDir}/{greyBorderImgPath}')
        greyBorderImg = greyBorderImg[0:329]

        if np.all(greyBorderImg[0:3,0:3] == (192,192,192)):
            
            left = 55
            top = 65
            bot = 55
            
            greyBorderImg = greyBorderImg[top:329-bot,left:440-left]
        
        cv2.imwrite(f'croppedImages/{greyBorderImgPath}', greyBorderImg)
        
def nameAfterRGBValue(srcDir, dstDir):
    fileNames = os.listdir(srcDir)
    numFiles = len(fileNames)
    threads = []
    threadC = 40
    for i in range(threadC):
        low = int(i * numFiles/threadC)
        high = int((i+1) * numFiles/threadC)
        
        subset = fileNames[low:high]
        
        newT = threading.Thread(target=findEdgesSubset, args=(subset,dstDir))
        threads.append(newT)
        
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
        
def convertToPNGs():
    originalFiles = os.listdir(imgDir)
    originalFiles.sort()

    for fileName in originalFiles:
        if '.jpg' in fileName:
            originalName = f'{imgDir}/{fileName}'
            im = Image.open(originalName).convert("RGBA")
            fileName = fileName.removesuffix('.jpg')

            newName = f'{imgDir}/{fileName}.png'
            im.save(newName, "png")

            os.remove(originalName)

if __name__ == "__main__":
    # cropAndGenerateMiniImagesSubset(['testImages/ISA_00001201.png'])
    # readImagesInFile2('txtFiles2/travertine.txt')
    cropAndGenerateMiniImages()
    # convertToPNGs()
    # grabFilesFromInternet()
