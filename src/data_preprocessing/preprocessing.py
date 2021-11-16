import cv2
import os
import glob
import numpy as np

class PreProcessing:
  """
  Goes through all images, returns preprocessed tensor.
  """

  def __init__(self, base_path):
      self.base_path = base_path
      self.kernel = np.array([[-1, -1, -1],
                   [-1, 8,-1],
                   [-1, -1, -1]])

  def get_colour_type(self, img_path):
    image = cv2.imread(img_path)
    if len(image.shape) == 3: return 3
    else: return 1

  def preprocess_image(self, img_path, height, width):
    """
    Function takes the path to the image and applys the preprocessing.
    """

    color_type = self.get_colour_type(img_path)

    if color_type == 1:
        img = cv2.imread(img_path, 0)
        img_gray = cv2.threshold(img,0,255,cv2.THRESH_TRUNC+cv2.THRESH_OTSU) 
        image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=self.kernel)

    elif color_type == 3:
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.threshold(img_gray,0,255,cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
        image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=self.kernel)
        image_sharp = cv2.cvtColor(image_sharp, cv2.COLOR_BGR2GRAY)
    

    combined = cv2.add(image_sharp, img_gray[1])
    dst = cv2.resize(combined, (width, height))
    img = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    return img

  def get_driver_data(self):
    """
    Returns a dictionary of image name as the key and driver and class as value.
    """
    driver_data = {}
    path = os.path.join(self.base_path,'driver_imgs_list.csv')

    print('Read drivers data')

    with open(path, 'r') as file:
      lines = file.readlines()
      lines = lines[1:]
    file.close()

    for line in lines:
      arr = line.strip().split(',')
      driver_data[arr[2]] = (arr[0], arr[1])
    
    return driver_data

  def load_train_data(self, height, width):
    """
    
    """
    x_train = []
    y_train = []
    driver_ids = []

    driver_data = self.get_driver_data()

    print('Read train images')
    for class_number in range(10):
        print(f'Load folder c{class_number}')
        class_number_str = 'c' + str(class_number)
        path = os.path.join(self.base_path, 'imgs/train', class_number_str, '*.jpg')
        file_paths = glob.glob(path)  # Gets all file names matching given path.
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            image = self.preprocess_image(file_path, height, width)
            x_train.append(image)
            y_train.append(class_number)
            driver_id = driver_data[file_name][0]
            driver_ids.append(driver_id)

    return x_train, y_train, driver_ids


  def load_test_data(self, height, width):
    x_test = []
    x_test_ids = []
    
    print('Read test images')

    path = os.path.join(self.base_path, 'imgs/test/*.jpg')
    file_paths = glob.glob(path)
    number_of_files = len(file_paths)

    for count, file_path in enumerate(file_paths):
        file_name = os.path.basename(file_path)
        image = self.preprocess_image(file_path, height, width)
        x_test.append(image)
        x_test_ids.append(file_name)
        if count % 1000 == 0:
            print(f"Read {count} images from {number_of_files}")

    return x_test, x_test_ids
