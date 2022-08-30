from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tensorflow.keras.preprocessing.image import img_to_array

def predict(bestModel,path,labelEnconder):
    #Load model
    model = load_model(bestModel)

    data_test = glob('{}/**/*.jpg'.format(path))
    random.shuffle(data_test)
    """
    Display Some Images
    """
    # Display first 15 images of moles, and how they are classified
    w = 60
    h = 40
    fig = plt.figure(figsize=(15, 15))
    columns = 4
    rows = 3

    for i in range(1, columns*rows + 1):
        img = cv2.imread(data_test[i])
        output = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output = cv2.resize(output, (128,128))
        image = output.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        (categoryProba) = model.predict(image)
        categoryIdx = categoryProba[0].argmax()
        categoryLabel = labelEnconder.classes_[categoryIdx]
        ax = fig.add_subplot(rows, columns, i)
        if categoryLabel == 0:
            ax.title.set_text("predição: Tipo 1 ({:.2f}%)\n\nTipo Verdadeiro: {}\n\nNome do arq: {}".format(categoryProba[0][categoryIdx] * 100,data_test[i].split('/')[1],data_test[i].split('/')[2]))

        if categoryLabel == 1:
            ax.title.set_text("Predição: Tipo 2 ({:.2f}%)\n\nTipo Verdadeiro: {}\n\nNome do arq: {}".format(categoryProba[0][categoryIdx] * 100,data_test[i].split('/')[1],data_test[i].split('/')[2]))

        if categoryLabel == 2:
            ax.title.set_text("Predição: Tipo 3 ({:.2f}%)\n\nTipo Verdadeiro: {}\n\nNome do arq: {}".format(categoryProba[0][categoryIdx] * 100,data_test[i].split('/')[1],data_test[i].split('/')[2]))



        # else:
        #   ax.title.set_text('Nao conseguiu prever!\n\nTipo Verdadeiro: {}\n\nNome do arq: {}'.format(test[i].split('/')[2],test[i].split('/')[3]))

        plt.xticks([])
        plt.yticks([])
        plt.imshow(output)
        plt.savefig('predict.png')
