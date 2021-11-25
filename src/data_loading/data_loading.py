import os
import glob
import numpy as np

class LoadData:
    """
    LoadData class
    """
    def __init__(self, base_path):
        self.base_path = base_path

    def load_data(self, height, width):
      """
      
      """
      x = []
      y = []

      print('Read images')
      for class_number in range(10):
        print(f'Load folder c{class_number}')
        class_number_str = 'c' + str(class_number)
        path = os.path.join(self.base_path, 'imgs/data', class_number_str, '*.jpg')
        file_paths = glob.glob(path)  # Gets all file names matching given path.
        sub_x = []
        sub_y = []
        for file_path in file_paths:
            sub_x.append(file_path)
            temp = np.zeros(10)
            temp[class_number] = 1
            sub_y.append(temp)
        self.shuffle_data(sub_x)
        x.append(sub_x)
        y.append(sub_y)
        
      return x, y

    def shuffle_data(self, x):
      np.random.shuffle(x)