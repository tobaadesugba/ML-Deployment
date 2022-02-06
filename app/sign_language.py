import random
import numpy as np
from keras.preprocessing import image
from os import listdir
from os.path import isdir, join


def path_to_tensor(img, size=64):
    # loads RGB image as PIL.Image.Image type
    ##img = image.load_img(img_path, target_size=(size, size))
    # convert PIL.Image.Image type to 3D tensor
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor 
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, size=64):
    list_of_tensors = [path_to_tensor(img_paths, size)]
    return np.vstack(list_of_tensors)

"""
    num_types = len(data['target_names'])
    targets = np_utils.to_categorical(np.array(data['target']), num_types)
"""