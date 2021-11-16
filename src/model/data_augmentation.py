import numpy as np
import tensorflow

from tensorflow.keras import layers

class RandomErasing(layers.Layer):
    """
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    ------
    Source: https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py
    """
    def __init__(self, probability, sl, sh, r1, mean, **kwargs):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = mean
        super().__init__(**kwargs)        
    
    def __call__(self, image):
        return random_erasing_image(image, self.probability, self.sl, self.sh, self.r1, self.mean)


def random_erasing_image(image, probability=0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
    """
    Function performs random erasing
    """
    if np.random.uniform(0, 1) > probability:
        return image
    area = image.shape[1] * image.shape[2]
    for _ in range(100):
        target_area = np.random.uniform(sl, sh) * area
        aspect_ratio = np.random.uniform(r1, 1/r1)

        h = int(round(np.sqrt(target_area * aspect_ratio)))
        w = int(round(np.sqrt(target_area / aspect_ratio)))

        if w < image.shape[2] and h < image.shape[1]:
            x1 = np.random.randint(0, image.shape[1] - h)
            y1 = np.random.randint(0, image.shape[2] - w)
            image[x1:x1+h, y1:y1+w] = mean[0]
            return image
    return image
        

def random_erasing(image, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
    """
    lamdba layer.
    """
    return layers.Lambda(lambda: random_erasing_image(image, probability, sl, sh, r1, mean))
