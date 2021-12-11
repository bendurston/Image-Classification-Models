import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, GlobalAveragePooling2D, ReLU, RandomFlip, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

class VGG16:
  """
  VGG16 model class.
  """

  def __init__(self, input_shape=(None, None, None, 3)):
    """
    Default constructor.
    """
    self.model = self.create_model(input_shape)
    self.model.summary()

  def create_model(self, input_shape):
    """
    Purpose:
      Creates the VGG16 model.
    Args:
      self - class instance.
      input_shape - the shape of the input into the model. 
                    (BATCH SIZE, HEIGHT, WIDTH, NUMBER OF CHANNELs)
    Returns:
      The compiled tensorflow model.
    """
    # Data augmentation.
    data_aug = Sequential([RandomFlip("horizontal")])

    # Model.
    model = Sequential([
      # Data augmentation layer.
      data_aug,

      # First Convolutional Block.
      Conv2D(32, (3, 3), padding='same'),
      BatchNormalization(),
      ReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(32, (3, 3), padding='same'),
      BatchNormalization(axis=3),
      ReLU(),
      MaxPool2D((2, 2), strides=(2, 2)),

      # Second Convolutional Block.
      Conv2D(64, (3, 3), padding='same'),
      BatchNormalization(),
      ReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(64, (3, 3), padding='same'),
      BatchNormalization(axis=3),
      ReLU(),
      MaxPool2D((2, 2), strides=(2, 2)),

      # Third Convolutional Block.
      Conv2D(128, (3, 3), padding='same'),
      BatchNormalization(),
      ReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(128, (3, 3), padding='same'),
      BatchNormalization(),
      ReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(128, (3, 3), padding='same'),
      BatchNormalization(axis=3),
      ReLU(),
      MaxPool2D((2, 2), strides=(2, 2)),

      # Fourth Convolutional Block.
      Conv2D(256, (3, 3), padding='same'),
      BatchNormalization(),
      ReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(256, (3, 3), padding='same'),
      BatchNormalization(),
      ReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(256, (3, 3), padding='same'),
      BatchNormalization(axis=3),
      ReLU(),
      MaxPool2D((2, 2), strides=(2, 2)),

      # Fifth Convolutional Block.
      Conv2D(256, (3, 3), padding='same'),
      BatchNormalization(),
      ReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(256, (3, 3), padding='same'),
      BatchNormalization(),
      ReLU(),
      MaxPool2D((1, 1), strides=(1, 1)),
      Conv2D(256, (3, 3), padding='same'),
      BatchNormalization(axis=3),
      ReLU(),
      MaxPool2D((2, 2), strides=(2, 2)),

      # Fully Connected Layers.
      GlobalAveragePooling2D(),
      Dense(1024, activation='relu'),
      Dropout(0.2),
      Dense(10, activation='softmax')
    ])
    # Build the model with the provided input shape.
    model.build(input_shape=input_shape)
    # Compile the model with adam optimizer and categorical crossentropy.
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

  def fit_model(self, x_train, y_train, epochs, batch_size, verbose):
    """
    Purpose:
      Fits the model with the training data sets.
    Args:
      self - class instance.
      x_train - the input training data set.
      y_train - the output training data set.
      epochs - the number of epochs to train the model.
      batch_size - the batch size to use in training.
      verbose - the verbose flag to use to display the training progress.
    Returns:
      None
    """
    self.model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batch_size)
    
  def evaluate_model(self, x_test, y_test):
    """
    Purpose:
      Evaluates the model with the test data set.
    Args:
      self - class instance.
      x_test - the input test data set.
      y_test - the output test data set.
    Returns:
      None
    """
    test_loss, test_acc = self.model.evaluate(x_test, y_test)
    print(f"Test lost: {test_loss} -- Test accuracy: {test_acc}")

  def predict_model(self, x_test, y_test, batch_size, verbose):
    """
    Purpose:
      Predict the output of the model with the input test set.
    Args:
      self - class instance.
      x_test - the input test data set.
      y_test - the output test data set.
      verbose - the verbose flag to use to display the predicting progress.
    Returns:
      None
    """
    y_pred = self.model.predict(x_test, batch_size=batch_size, verbose=verbose)
    # Display confusion matrix and accuracy of predicted values.
    self.display_prediction_results(y_pred, y_test)

  def display_prediction_results(self, y_pred, y_test):
    """
    Purpose:
      Displays the confusion matrix and the models accuracy, recall, precision and F1 score.
    Args:
      self - class instance.
      y_pred - the predicted outputs.
      y_test - the actual outputs.
    Returns:
      None
    """
    # Get the index of the max value in each sub array.
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    # Names of possible classes.
    class_names = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
    # Generate confusion matrix.
    conf_mat = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=class_names)
    # Increases the size of the displayed confusion matrix
    fig, ax = plt.subplots(figsize=(10,10))
    # Plot the confusion matrix.
    display.plot(ax=ax, values_format='')
    # Output the models accuracy, recall, precision and F1 score.
    print(classification_report(y_test, y_pred, target_names=class_names))
