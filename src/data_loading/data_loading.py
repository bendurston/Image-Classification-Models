import os
import glob
import numpy as np

class LoadData:
  """
  LoadData class
  """

  def __init__(self, base_path):
    """
    Default constructor.
    """
    self.base_path = base_path

  def load_data(self):
    """
    Purpose:
      Loads the paths of all of the images, and create a one hot encoding for 
      their classifications. x and y lists share an index relation.
    Args:
      self - class instance.
    Returns:
      x - list of image paths.
      y - list of one hot encodings of each image's classification.
    """
    x = []
    y = []
    # Loop through each directory.
    for class_number in range(10):
      class_number_str = 'c' + str(class_number)
      # Path to all images in current class directory.
      path = os.path.join(self.base_path, 'imgs/data', class_number_str, '*.jpg')
      # Gets all file names matching given path.
      file_paths = glob.glob(path)
      sub_x = []
      sub_y = []
      # Loops through each path in the current class directory.
      for file_path in file_paths:
          sub_x.append(file_path)
          # Create one hot encoding.
          temp = np.zeros(10)
          temp[class_number] = 1
          sub_y.append(temp)
      # Shuffle the paths.
      self.shuffle_data(sub_x)
      x.append(sub_x)
      y.append(sub_y)
    print("Saved all image paths.")
    return x, y

  def shuffle_data(self, x):
    """
    Purpose:
      Shuffles the values of the given array.
    Args:
      self - class instance.
      x - the array to shuffle.
    """
    np.random.shuffle(x)