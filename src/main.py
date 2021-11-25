import os
import numpy as np
from dotenv import load_dotenv
from data_preprocessing.preprocessing import PreProcessing
from data_loading.data_loading import LoadData
from model.vgg16 import VGG16






if __name__ == '__main__':
  
  HEIGHT = 96
  WIDTH = 96

  load_dotenv()
  PATH = os.getenv('SAMPLE_PATH')
  x, y = LoadData(PATH).load_data(HEIGHT, WIDTH)
  
  p = PreProcessing()
  x_train, y_train, x_test, y_test = p.split_data(x, y, HEIGHT, WIDTH)
  # x, y, driver_ids = p.load_data(HEIGHT, WIDTH)

  # x_train, y_train, x_test, y_test = split_data(x, y)

