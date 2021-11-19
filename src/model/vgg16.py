from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, RandomFlip, RandomCrop
from model.data_augmentation import RandomErasing

class VGG16:

  def __init__(self):
    self.data_augmentation = self.data_aug()    
    self.vgg16 = self.build_model()
    self.vgg16.summary()

  def data_aug(self):
    data_augmentation = Sequential([
      RandomFlip("horizontal"),
      RandomCrop(224, 224),
      RandomErasing(0.5, 0.02, 0.4, 0.3, [0.4914, 0.4822, 0.4465])
    ])
    return data_augmentation

  def build_model(self, input_shape=(336,336,1)):
    model = Sequential([
      Input(shape=input_shape),                               # Block 1
      self.data_augmentation,
      Conv2D(64, (3, 3), activation='relu', padding='same'),
      Conv2D(64, (3, 3), activation='relu', padding='same'),
      MaxPool2D((2, 2), strides=(2, 2)),                      # Block 2
      Conv2D(128, (3, 3), activation='relu', padding='same'),
      Conv2D(128, (3, 3), activation='relu', padding='same'),
      MaxPool2D((2, 2), strides=(2, 2)),                      # Block 3
      Conv2D(256, (3, 3), activation='relu', padding='same'),
      Conv2D(256, (3, 3), activation='relu', padding='same'),
      Conv2D(256, (3, 3), activation='relu', padding='same'),
      MaxPool2D((2, 2), strides=(2, 2)),                      # Block 4  
      Conv2D(512, (3, 3), activation='relu', padding='same'),
      Conv2D(512, (3, 3), activation='relu', padding='same'),
      Conv2D(512, (3, 3), activation='relu', padding='same'),
      MaxPool2D((2, 2), strides=(2, 2)),                      # Block 5  
      Conv2D(512, (3, 3), activation='relu', padding='same'),
      Conv2D(512, (3, 3), activation='relu', padding='same'),
      Conv2D(512, (3, 3), activation='relu', padding='same'),
      MaxPool2D((2, 2), strides=(2, 2)),                      # Fully Connected Layers
      Flatten(),
      Dense(4096, activation='relu'),
      Dense(4096, activation='relu'),
      Dense(1000, activation='softmax')
    ])
    return model
  
  def train_model(self):
    # Compile model
    # model.compile(optimizer='adam', 
    #           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #           metrics=[keras.metrics.SparseCategoricalAccuracy()])
    # model.fit(train_images, train_labels, epochs=epochs)
    pass

if __name__ == '__main__':
  vgg16 = VGG16()
