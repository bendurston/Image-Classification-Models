import cv2
import os
import glob
import numpy as np
# from src.data_preprocessing.data_augmentation import random_erasing

class PreProcessing:
  """
  Goes through all images, returns preprocessed tensor.
  """

  def __init__(self):
    self.kernel = np.array([[-1, -1, -1],
                  [-1, 8,-1],
                  [-1, -1, -1]])
  
  def get_colour_type(self, img_path):
    image = cv2.imread(img_path)
    if len(image.shape) == 3: return 3
    return 1

  def preprocess_image(self, img_path, height, width, training):
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

    if training:
      img = self.random_erasing(img)
    return img

  def random_erasing(self, image, probability=0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
    """
    Function performs random erasing
    """
    if np.random.uniform(0, 1) > probability:
        return image
    area = image.shape[0] * image.shape[1]
    for _ in range(100):
        target_area = np.random.uniform(sl, sh) * area
        aspect_ratio = np.random.uniform(r1, 1/r1)

        h = int(round(np.sqrt(target_area * aspect_ratio)))
        w = int(round(np.sqrt(target_area / aspect_ratio)))

        if w < image.shape[1] and h < image.shape[0]:
            x1 = np.random.randint(0, image.shape[0] - h)
            y1 = np.random.randint(0, image.shape[1] - w)
            if image.shape[2] == 3:
                image[x1:x1+h, y1:y1+w, 0] = mean[0]
                image[x1:x1+h, y1:y1+w, 1] = mean[1]
                image[x1:x1+h, y1:y1+w, 2] = mean[2]
            else:
                image[x1:x1+h, y1:y1+w, 0] = mean[0]
            return image
    return image
  
  def split_data(self, x, y, height, width):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    split_points = self.percent_indexes(x)
    for class_num, (xi, yi) in enumerate(zip(x, y)):
      print(f"Preprocessing class: {class_num}.")
      for image_number, (image_path, out) in enumerate(zip(xi, yi)):
        if image_number < split_points[class_num]:
          
          image = self.preprocess_image(image_path, height, width, False)
          x_test.append(image)
          y_test.append(out)
          
        else:
          image = self.preprocess_image(image_path, height, width, True)
          x_train.append(image)
          y_train.append(out)
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    

  def percent_indexes(self, x):
    split_points = []
    for xi in x:
      number_of_images = len(xi)
      split_point = int(number_of_images*0.2)
      split_points.append(split_point)
    return split_points