import cv2
import os
from tqdm import tqdm
import random
from src.utils import filterBadImages, filterGreenImages


def resizedImages(folder,destinate, IMG_WIDTH=256, IMG_HEIGHT=256):
    countBadImages = 0
    countGoodImages = 0
    print('\t\t\t Diminuindo tamanho das imagens')
    print('='*90)
    #Listando todas as pastas
    categories = os.listdir(folder)
    for category in categories:
        #Concatenando a category com o folder, para chegar ao path deles.
        folderPath = os.path.join(folder, category)
        #Buscando todas as imagens dentro as pastas
        imgNames = os.listdir(folderPath)
        random.shuffle(imgNames)

        #caminho para as onde as novas imagens irão
        OUTPUT_FOLDER_RESIZED = os.path.join(destinate,category)

        #verificar se a pasta ja foi criada
        if(not os.path.exists(OUTPUT_FOLDER_RESIZED)): os.makedirs(OUTPUT_FOLDER_RESIZED)

        for imgName in tqdm(imgNames[:10],desc=category):
            OUTPUT_FILENAME = os.path.join(OUTPUT_FOLDER_RESIZED, imgName)

            imgPath = os.path.join(folderPath, imgName)
            _,ftype = os.path.splitext(imgPath)
            if ftype == '.jpg':
                #usa as functions para filtrar as melhores images
                if(filterBadImages(imgPath) or filterGreenImages(imgPath)):
                    countBadImages += 1
                else:
                    countGoodImages += 1
                    img = cv2.imread(imgPath)
                    resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    cv2.imwrite(OUTPUT_FILENAME, resized)
    print('='*30)
    print('Images Ruins:', countBadImages)
    print('Images Boas:', countGoodImages)
    print('='*30)

