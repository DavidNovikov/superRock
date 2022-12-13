import re
import glob
import cv2

from skimage import io
from sklearn.neighbors import KNeighborsClassifier

def determineGaps(img, model):

    h, w, _ = img.shape
    
    for row in range(h):
        for col in range(w):
            pix = img[row,col]
            closestFile = model.predict(pix.reshape(1,-1))
            
            numsAsStrings = re.findall("\d+\.\d+", closestFile[0])
            vals = [float(x) for x in numsAsStrings]
            if sum(abs(vals - pix )) < 81:
                img[row,col] = [0,0,0]
            
                
    
    # cv2.imwrite('img4.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite('displayColors81.png', img)

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
    
    img = io.imread('rgbtricub.jpg')
    img = img[:,:,0:3]
    
    determineGaps(img, model)