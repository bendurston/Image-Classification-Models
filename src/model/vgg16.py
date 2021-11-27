import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, BatchNormalization, ReLU, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

class VGG16:

  def __init__(self, input_shape=(None, None, 3)):
    self.vgg16 = self.create_model(input_shape)
    self.vgg16.summary()

  def create_model(self, input_shape):
    model = Sequential([
      Input(shape=input_shape),                               # Block 1
      Conv2D(32, (3, 3), padding='same'),
      BatchNormalization(),
      ReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(32, (3, 3), padding='same'),
      BatchNormalization(axis = 3),
      ReLU(),
      MaxPool2D((2, 2), strides=(2, 2)),                      # Block 2
      Conv2D(64, (3, 3), padding='same'),
      BatchNormalization(),
      ReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(64, (3, 3), padding='same'),
      BatchNormalization(axis = 3),
      ReLU(),
      MaxPool2D((2, 2), strides=(2, 2)),                      # Block 3
      Conv2D(128, (3, 3), padding='same'),
      BatchNormalization(),
      ReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(128, (3, 3), padding='same'),
      BatchNormalization(),
      ReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(128, (3, 3), padding='same'),
      BatchNormalization(axis = 3),
      ReLU(),
      MaxPool2D((2, 2), strides=(2, 2)),                      # Block 4  
      Conv2D(256, (3, 3), padding='same'),
      BatchNormalization(),
      ReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(256, (3, 3), padding='same'),
      BatchNormalization(),
      ReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(256, (3, 3), padding='same'),
      BatchNormalization(axis = 3),
      ReLU(),
      MaxPool2D((2, 2), strides=(2, 2)),                      # Block 5  
      Conv2D(256, (3, 3), padding='same'),
      BatchNormalization(),
      ReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(256, (3, 3), padding='same'),
      BatchNormalization(),
      ReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(256, (3, 3), padding='same'),
      BatchNormalization(axis = 3),
      ReLU(),
      MaxPool2D((2, 2), strides=(2, 2)),                      # Fully Connected Layers
      GlobalAveragePooling2D(),
      Dense(512, activation='relu'),
      Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

  def fit_model(self, model, x_train, y_train, epochs, batch_size, verbose):
    model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batch_size)
    
  def evaluate_model(self, model, x_test, y_test, batch_size):
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'\nTest lost: {test_loss} -- Test accuracy: {test_acc}')

    y_pred = model.predict(x_test, batch_size=batch_size, verbose=1)
    

    
    y_pred=np.argmax(y_pred, axis=1)
    y_test=np.argmax(y_test, axis=1)
    
    # Names of possible classes
    class_names = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
    # Generate confusion matrix and display using sklearn
    conf_mat = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=class_names)
    # Increases the size of the displayed confusion matrix
    fig, ax = plt.subplots(figsize=(10,10))
    disp.plot(ax=ax, values_format='')
    print(classification_report(y_test, y_pred, target_names=class_names))

