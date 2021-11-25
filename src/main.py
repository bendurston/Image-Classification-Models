import os
import numpy as np
from dotenv import load_dotenv
from data_preprocessing.preprocessing import PreProcessing
from data_loading.data_loading import LoadData
from model.vgg16 import VGG16






if __name__ == '__main__':
  
  HEIGHT = 128
  WIDTH = 128

  load_dotenv()
  PATH = os.getenv('PATH_TO_DATA')
  x, y = LoadData(PATH).load_data(HEIGHT, WIDTH)
  
  p = PreProcessing()
  x_train, y_train, x_test, y_test = p.split_data(x, y, HEIGHT, WIDTH)
  
  model = VGG16((16, 128, 128, 3))
  model.train_model(x_train, y_train, x_test, y_test)
