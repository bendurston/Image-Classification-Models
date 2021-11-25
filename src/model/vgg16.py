import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, RandomFlip, RandomCrop, BatchNormalization, LeakyReLU, GlobalAveragePooling2D
# from src.data_preprocessing.data_augmentation import RandomErasing
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD



class VGG16:

  def __init__(self):
    self.vgg16 = self.build_model()
    self.vgg16.summary()

  def build_model(self, input_shape=(None,None,3)):
    model = Sequential([
      Input(shape=input_shape),                               # Block 1
      Conv2D(32, (3, 3), padding='same'),
      BatchNormalization(),
      LeakyReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(32, (3, 3), padding='same'),
      BatchNormalization(axis = 3),
      LeakyReLU(),
      MaxPool2D((2, 2), strides=(2, 2)),                      # Block 2
      Conv2D(64, (3, 3), padding='same'),
      BatchNormalization(),
      LeakyReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(64, (3, 3), padding='same'),
      BatchNormalization(axis = 3),
      LeakyReLU(),
      MaxPool2D((2, 2), strides=(2, 2)),                      # Block 3
      Conv2D(128, (3, 3), padding='same'),
      BatchNormalization(),
      LeakyReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(128, (3, 3), padding='same'),
      BatchNormalization(),
      LeakyReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(128, (3, 3), padding='same'),
      BatchNormalization(axis = 3),
      LeakyReLU(),
      MaxPool2D((2, 2), strides=(2, 2)),                      # Block 4  
      Conv2D(256, (3, 3), padding='same'),
      BatchNormalization(),
      LeakyReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(256, (3, 3), padding='same'),
      BatchNormalization(),
      LeakyReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(256, (3, 3), padding='same'),
      BatchNormalization(axis = 3),
      LeakyReLU(),
      MaxPool2D((2, 2), strides=(2, 2)),                      # Block 5  
      Conv2D(256, (3, 3), padding='same'),
      BatchNormalization(),
      LeakyReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(256, (3, 3), padding='same'),
      BatchNormalization(),
      LeakyReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(256, (3, 3), padding='same'),
      BatchNormalization(axis = 3),
      LeakyReLU(),
      MaxPool2D((2, 2), strides=(2, 2)),                      # Fully Connected Layers
      GlobalAveragePooling2D(),
      Dense(512, activation='relu'),
      Dense(10, activation='softmax')
    ])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
  
  def train_model(self, x_train, y_train, x_test, y_test):
    self.vgg16.fit(x_train, y_train, epochs=10, verbose=1, batch_size=16)
    test_loss, test_acc = self.vgg16.evaluate(x_test, y_test)
    print(f'\nTest lost: {test_loss} -- Test accuracy: {test_acc}')

