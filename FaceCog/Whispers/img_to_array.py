import skimage.io as io
def img_to_array(image_path):
    """ Converts image from image path into numpy array
    Parameters:
    image_path: string

    Returns:
    numpy array
    """
    # shape-(Height, Width, Color)
    image = io.imread(str(image_path))
    if image.shape[-1] == 4:
    # Image is RGBA, where A is alpha -> transparency
    # Must make image RGB.
        image = image[..., :-1]  # png -> RGB

    return image