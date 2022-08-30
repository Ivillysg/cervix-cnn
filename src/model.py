from keras.models import Sequential
from keras.layers import AveragePooling2D, Conv2D, MaxPool2D, Flatten, Dense

class CervixCNN:
    def VGG16(width, height, num_classes=3, finalAct='sigmoid'):
        model = Sequential()
        # CONV => CONV => RELU => POOL
        model.add(Conv2D(input_shape=(width, height, 3), filters=64,
                        kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=64, kernel_size=(3, 3),
                        padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        # CONV => CONV => RELU => POOL
        model.add(Conv2D(filters=128, kernel_size=(
            3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(
            3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        # CONV => CONV => CONV => RELU => POOL
        model.add(Conv2D(filters=256, kernel_size=(
            3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(
            3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(
            3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        # CONV => CONV => CONV => RELU => POOL
        model.add(Conv2D(filters=512, kernel_size=(
            3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(
            3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(
            3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        # CONV => CONV => CONV => RELU => POOL
        model.add(Conv2D(filters=512, kernel_size=(
            3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(
            3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(
            3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dense(units=num_classes, activation=finalAct))

        return model
