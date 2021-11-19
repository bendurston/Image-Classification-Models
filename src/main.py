import os

from dotenv import load_dotenv
from data_preprocessing.preprocessing import PreProcessing
from model.vgg16 import VGG16

HEIGHT = 336
WIDTH = 336

def split_data(x, y):
  x_train = []
  y_train = []
  x_test = []
  y_test = []
  split_points = percent_indexes(x)
  for index, (xi, yi) in enumerate(zip(x, y)):
    for image_number, image, output in enumerate(zip(xi, yi)):
      if image_number < split_points[index]:
        x_train.append(image)
        y_train.append(output)
      else:
        x_test.append(image)
        y_test.append(output)
  return x_train, y_train, x_test, y_test
  

def percent_indexes(x):
  split_points = []
  for xi in x:
    number_of_images = len(xi)
    split_point = number_of_images//20
    split_points.append(split_point)
  return split_points


if __name__ == '__main__':
  
  load_dotenv()
  PATH = os.getenv('PATH_TO_DATA')
  p = PreProcessing(PATH)
  x, y, driver_ids = p.load_data(HEIGHT, WIDTH)
  
  x1 = VGG16()
  x_train, y_train, x_test, y_test = split_data(x, y)
  # plt.figure(figsize=(10, 10))

  # for index, x in enumerate(x_test):
    
  #   img = x1.da(x)
    
  #   # img = random_erasing_image(x)
  #   ax = plt.subplot(3, 3, index + 1)
  #   plt.imshow(img)
  #   plt.axis("off")
  # plt.show()

