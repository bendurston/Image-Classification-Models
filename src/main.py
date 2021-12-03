import os
from dotenv import load_dotenv
from data_preprocessing.preprocessing import PreProcessing
from data_loading.data_loading import LoadData
from model.vgg16 import VGG16






if __name__ == '__main__':
  
  HEIGHT = 128
  WIDTH = 128

  load_dotenv()
  PATH = os.getenv('PATH_TO_DATA')
  x, y = LoadData(PATH).load_data()

  p = PreProcessing()
  x_train, y_train, x_test, y_test = p.split_data(x, y, HEIGHT, WIDTH)

  EPOCHS = 15
  VERBOSE = 1
  BATCH_SIZE = 16


  model = VGG16((None, HEIGHT, WIDTH, 3))
  model.fit_model(x_train, y_train, EPOCHS, BATCH_SIZE, VERBOSE)
  model.evaluate_model(x_test, y_test)
  model.predict_model(x_test, y_test, BATCH_SIZE, VERBOSE)
