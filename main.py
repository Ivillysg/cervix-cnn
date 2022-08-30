from src.resizedImages import *
from settings import *
from src.model import CervixCNN
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from keras.utils import to_categorical
import numpy as np
from src.predict import *
from src.utils import getDataImages
import keras
print('Version Keras:',keras.__version__)
import tensorflow
print('Version TensorFlow:',tensorflow.__version__)

#Caso nao exista a pasta INPUT, ira fazer o split dos dados e a redução das dimensões
if(not os.path.exists('input')):
    resizedImages(DATASET, TRAINSET_RESIZED_FOLDER, IMG_WIDTH, IMG_HEIGHT)


dataset = getDataImages(TRAINSET_RESIZED_FOLDER)
# converta os dados e rótulos em matrizes NumPy enquanto dimensiona o pixel
# intensidades no intervalo [0, 255]
trainData = dataset[0]
trainData = np.array(trainData)/255.0

train_labels = dataset[1]
train_labels = np.array(train_labels)

labelEnconder = LabelEncoder()
train_labels = labelEnconder.fit_transform(train_labels)
train_labels = to_categorical(train_labels, num_classes=3)
#se existe o modelo ja treinado, vai carrega-lo e execução a função de predicao.
if(os.path.exists('model.h5')):
    print('='*30)
    print('Criando predição')
    print('='*30)
    predict('model.h5','test',labelEnconder)


else:
    # particione os dados em treinamentos e testes de divisão usando 80% dos
    # os dados para treinamento e os 20% restantes para teste
    (trainX, testX, trainY, testY) = train_test_split(trainData, train_labels,
        test_size=0.20, random_state=42)

    print(trainX.shape, trainY.shape)
    # inicialize o objeto de aumento de dados de treinamento
    trainAug = ImageDataGenerator(
        rotation_range=15,
        fill_mode="nearest")

    print("[INFO] Create model...")
    model = CervixCNN.VGG16(IMG_WIDTH,IMG_HEIGHT,3)

    # compile our model
    print("[INFO] compiling model...")
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt,
        metrics=["accuracy"])


    checkpoint = ModelCheckpoint('model.h5',
                                monitor='val_loss',
                                mode='min',
                                save_best_only=True,
                                verbose=1)

    earlystop = EarlyStopping(monitor='val_loss',
                            min_delta=0,
                            patience=5,
                            verbose=1,
                            restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                factor=0.2,
                                patience=5,
                                verbose=1,
                                min_delta=0.0001)

    callbacks = [earlystop,checkpoint,reduce_lr]

    # train the head of the network
    print("[INFO] training head...")
    H = model.fit_generator(
        trainAug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        callbacks=callbacks,
        epochs=3)

    print('='*90)
    print('Criando grafico...')
    print('='*90)
    print('Criando testes...')
    plt.plot(H.history['loss'],'r',label='training loss')
    plt.plot(H.history['val_loss'],label='validation loss')
    plt.xlabel('# epochs')
    plt.ylabel('loss')
    plt.legend()

    plt.plot(H.history['accuracy'],'r',label='training accuracy')
    plt.plot(H.history['val_accuracy'],label='validation accuracy')
    plt.xlabel('# epochs')
    plt.ylabel('loss')
    plt.legend()

    plt.savefig('grafico.png')

    predict('model.h5','test')
