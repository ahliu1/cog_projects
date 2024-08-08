from coco import load_coco
import numpy as np
import random

def init_data(): # function should only be ran once
    """
    Forming our training data is the following process:
    - Separate out image IDs into distinct sets for training and validation
    - Pick a random training image and one of its associated captions. We will call these our “true image” and “true caption”
    - Pick a different image. We will call this our “confusor image”.
    Returns:
    training_data: np.array, shape (# of images, 3) 
        - format: np.array[(true-caption-ID, true-image-ID, confusor-image-ID),
                            (true-caption-ID, true-image-ID, confusor-image-ID),
                            ...
                            (true-caption-ID, true-image-ID, confusor-image-ID)]
    """
    dtb = load_coco()
    # data = np.zeros((len(img_IDs), 3))
    data = dtb.return_captionIDs_and_imageIDs() # np.array of IDs. (413258, 2)
    # print(data.shape)
    confuser_img_ID = np.reshape(data[:, 1], (-1, 1)) # reshape array so it's 2D instead of 1D
    np.random.shuffle(confuser_img_ID)
    # print(confuser_img_ID.shape)
    data = np.hstack((data, confuser_img_ID))
    n_rows = data.shape[0]
    training_data = data[:round(n_rows * 0.8)] # 4/5 of data for training, (330606, 3)
    validation_data = data[round(n_rows * 0.8):] # 1/5 of data for testing, (82652, 3)
    # print(training_data.shape) # 
    # print(validation_data.shape) # 
    with open("datasets/train.npy", mode="wb") as f:
        np.save(f, training_data)
    with open("datasets/validation.npy", mode="wb") as f:
        np.save(f, training_data)
    # return training_data, validation_data

def get_training_and_validation_data():
    training_data = np.load("datasets/train.npy")
    validation_data = np.load("datasets/validation.npy")
    return training_data, validation_data
