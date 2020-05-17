# Run-Length Encode and Decode

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import tensorflow as tf
import os

# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

#########
######### these methods are for moving files around and making new masks images
#########


def same_id(filename, masks, display=False, image_size=(10,10)):
    """
    filename: string, the filename of the image. ex. "328_3982.jpg"
    masks: a dataframe of the masks
    display: True- plot the image, False- not
    image_size: in plt.figure(figsize= ?) dimensions
    return: (768,768) array with all the masks with same id added together
    """
    img_masks = masks.loc[masks.ImageId == filename, "EncodedPixels"].tolist()
    SHAPE = (768,768)
    allmasks = np.zeros(SHAPE).astype('uint8')
    for img_mask in img_masks:
        allmasks += rle_decode(img_mask,SHAPE).astype("uint8").T
    if display:
        plt.figure(figsize=image_size)
        plt.imshow(allmasks)
    return (allmasks, filename)


def save_img(arr, filename, target_dir):
    """
    parameters-
        arr: array of (height, width, channels=1) with dtype of uint8 and 
    scaled bn 0 and 255-- or the output from add_masks()
        filename: a string
        target_dir: a string
    """
    if not os.path.isdir(target_dir):
        raise FileNotFoundError("cant find the train directory")

    new_path = os.path.join(target_dir, filename)
    encoded = tf.image.encode_jpeg(arr[:,:,np.newaxis])
    tf.io.write_file(new_path, encoded)
    

# use this function in map, loop or list comprehension to perform on all filenames
# in a list


# def add_masks_and_save(filename):
#     """
#     use case:
#     filenames = string filenames of jpeg files
#     target_dir = where to save the new masks images
#     pool = multiprocessing.Pool()
#     result = pool.map(add_masks_and_save, filenames)
#     """
#     new_masks_dir = "/media/thistle/Passport/Kaggle_Data/airbus/fromkaggle/combined_masks"

#     z = same_id(filename, masks)
#     save_img(*z, new_masks_dir)
