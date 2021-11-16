import os
import numpy as np
import cv2
from dotenv import load_dotenv
from data_preprocessing.preprocessing import PreProcessing
from model.data_augmentation import random_erasing_image
from model.vgg16 import VGG16
import matplotlib.pyplot as plt

HEIGHT = 336
WIDTH = 336

if __name__ == '__main__':
  
  load_dotenv()
  PATH = os.getenv('SAMPLE_PATH')
  p = PreProcessing(PATH)
#   x_train, y_train, driver_ids = p.load_train_data(HEIGHT, WIDTH)
  x_test, x_test_id = p.load_test_data(HEIGHT, WIDTH)
  x1 = VGG16()
  # plt.figure(figsize=(10, 10))

  # for index, x in enumerate(x_test):
    
  #   img = x1.da(x)
    
  #   # img = random_erasing_image(x)
  #   ax = plt.subplot(3, 3, index + 1)
  #   plt.imshow(img)
  #   plt.axis("off")
  # plt.show()

