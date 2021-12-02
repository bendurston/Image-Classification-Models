import cv2
import numpy as np

class PreProcessing:
  """
  PreProcessing class.
  """

  def __init__(self):
    """
    Default constructor.
    """
    self.kernel = np.array([[-1, -1, -1],
                  [-1, 8,-1],
                  [-1, -1, -1]])
  
  def get_colour_type(self, img_path):
    """
    Purpose:
      Gets the colour type of the given image.
    Args:
      self - class instance.
      img_path - the path of the image to check the colour type of.
    Returns:
      The number of channels the image has. 3 for RBG/HSV and 1 for grayscale.
    """
    image = cv2.imread(img_path)
    if len(image.shape) == 3: return 3
    return 1

  def preprocess_image(self, img_path, height, width, training):
    """
    Purpose:
      Takes image path, reads it and applies image processing to it.
      Threshold grayscale image, sharpen a copy of the base image,
      add the results to get the final image, and resize the image 
      to the specified height and width. If training is true,
      image will have random erasing applied to it.
    Args:
      self - class instance.
      img_path - the path to the image.
      height - the height of the final image.
      width - the width of the final image.
      training - the training flag.
    Returns:
      Preprocessed image of type ndarray.
    """

    color_type = self.get_colour_type(img_path)
    # Image is in grayscale.
    if color_type == 1:
      img = cv2.imread(img_path, 0)
      # Apply adaptive thresholding.
      img_gray = cv2.threshold(img,0,255,cv2.THRESH_TRUNC+cv2.THRESH_OTSU) 
      # Sharpen image using laplacian filter.
      image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=self.kernel)

    # Image is in BGR/HSV
    elif color_type == 3:
      img = cv2.imread(img_path)
      # Convert to grayscale.
      img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      # Apply adaptive thresholding.
      img_gray = cv2.threshold(img_gray,0,255,cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
      # Sharpen image using laplacian filter.
      image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=self.kernel)
      # Convert sharpened image to grayscale.
      image_sharp = cv2.cvtColor(image_sharp, cv2.COLOR_BGR2GRAY)
    
    # Combine thresholded image and sharpened image.
    combined = cv2.add(image_sharp, img_gray[1])
    # Resize image.
    dst = cv2.resize(combined, (width, height))
    # Convert to BGR.
    img = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    # Checks if image is apart of training set.
    if training:
      # Apply random erasing to processed image.
      img = self.random_erasing(img)
    return img

  def random_erasing(self, image, probability=0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
    """
    Function that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    ------
    Source: https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py
    """
    # Checks if random erasing will be performed.
    if np.random.uniform(0, 1) > probability:
        return image
    # Get area of image.
    area = image.shape[0] * image.shape[1]
    # Loops attempting to apply random erasing.
    for _ in range(100):
      # Get the target area and the aspect ratio that will apply random erasing to.
      target_area = np.random.uniform(sl, sh) * area
      aspect_ratio = np.random.uniform(r1, 1/r1)
      # Get height and width of random erasing area.
      h = int(round(np.sqrt(target_area * aspect_ratio)))
      w = int(round(np.sqrt(target_area / aspect_ratio)))
      # Check if height and width fits within the bounds of the image.
      if w < image.shape[1] and h < image.shape[0]:
        # Get lower bounds of height and width.
        x1 = np.random.randint(0, image.shape[0] - h)
        y1 = np.random.randint(0, image.shape[1] - w)
        # Replace bounds with mean value.
        if image.shape[2] == 3:
            image[x1:x1+h, y1:y1+w, 0] = mean[0]
            image[x1:x1+h, y1:y1+w, 1] = mean[1]
            image[x1:x1+h, y1:y1+w, 2] = mean[2]
        else:
            image[x1:x1+h, y1:y1+w, 0] = mean[0]
        return image
    return image
  
  def split_data(self, x, y, height, width):
    """
    Purpose:
      Split the data into test and train, and process the images.
    Args:
      self - class instance.
      x - the paths of all of the images.
      y - the classifications, related to x by index.
      height - the height of the image used in resizing.
      width - the width of the image used in resizing.
    Returns:
      ndarrays of x and y train and test sets.
    """
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    # Get the split points of each class.
    split_points = self.percent_indexes(x)
    # Loop through all classes
    for class_num, (xi, yi) in enumerate(zip(x, y)):
      print(f"Preprocessing class: {class_num}.")
      # Loop through each image in the class.
      for image_number, (image_path, out) in enumerate(zip(xi, yi)):
        # Check if its less than the split point.
        if image_number < split_points[class_num]:
          # Process the image with training flag set to false, add it to the x_test
          # list along with the one hot encoding classification for the image.
          image = self.preprocess_image(image_path, height, width, False)
          x_test.append(image)
          y_test.append(out)
        else:
          # Process the image with training flag set to True, add it to the x_train
          # list along with the one hot encoding classification for the image.
          image = self.preprocess_image(image_path, height, width, True)
          x_train.append(image)
          y_train.append(out)
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
  
  def percent_indexes(self, x):
    """
    Purpose:
      Used to get the points of where the x_test and x_train will be split.
      Split point is 20% of the number of images in each class.
    Args:
      self - class instance.
      x - the paths of all of the images.
    Returns:
      Returns a list of 10 index points to split each class.
    """
    split_points = []
    for xi in x:
      number_of_images = len(xi)
      # Get the index that makes up 20% of the data.
      split_point = int(number_of_images*0.2)
      split_points.append(split_point)
    return split_points
