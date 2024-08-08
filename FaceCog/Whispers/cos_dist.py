import numpy as np
def cos_dist(vector_a, vector_b):
    """ Calculates the cosine distance between two vectors
    Parameters:
        vector_a: numpy array of shape (N,)
        vector_b: numpy array of shape (N,)
    Return:
        cos_dist: float
    """
    a_magn = np.sqrt(np.sum(vector_a ** 2))
    b_magn = np.sqrt(np.sum(vector_b ** 2))
    cos_dist = 1 - np.dot(vector_a, vector_b)/(a_magn * b_magn)
    return cos_dist
