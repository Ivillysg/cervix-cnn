import os,cv2
import numpy as np
from tqdm import tqdm

def filterGreenImages(path):
  image = cv2.imread(path)
  lower = np.array([36,0,0], dtype = "uint8")
  upper = np.array([86,255,255], dtype = "uint8")
  mask = cv2.inRange(image, lower, upper)
  output = np.array(cv2.bitwise_and(image, image, mask = mask))

  total = np.sum(output)
  if total == 0:
    return True
  return False

def filterBadImages(path):
  img = cv2.imread(path,1)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  hist_full, bins = np.histogram(img.ravel(),256,[0,256])
  hist_full = np.array(hist_full)
  totalSum = np.sum(hist_full)
  if(totalSum < 36000000):
    return True
  return False


def getDataImages(folder):
    categories = sorted(os.listdir(folder))
    labels=[i for i in range(len(categories))]

    labels_dict = dict(zip(categories,labels))
    print("[INFO] Loading dataset...")
    print('='*90)
    print(labels_dict)
    print(categories)
    print(labels)

    imgSize=[128,128]
    data = []
    labels = []

    for category in categories:
        foldPath = os.path.join(folder,category)
        imgNames = os.listdir(foldPath)
        for imgName in tqdm(imgNames,desc=category):
          imgPath = os.path.join(foldPath,imgName)
          _,ftype = os.path.splitext(imgPath)
          if ftype == '.jpg':
              img = cv2.imread(imgPath)
              img - cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
              img = cv2.resize(img, (imgSize[0],imgSize[1]))
              data.append(img)
              labels.append(labels_dict[category])
    return [data,labels]