from Node import Node
from cos_dist import cos_dist
from img_to_array import img_to_array
import numpy as np
from facenet_models import FacenetModel


def adjacency_graph(image_paths):
    """Creating list of nodes and adjacency matrix
    Parameters:
    List of image paths: shape = (N,)

    Returns:
    Tuple(list of nodes, adjacency matrix)
    """
    # default cos threshold - CHANGE LATER AFTER TESTING
    cos_threshold = 0.5
    # converting image_paths into np array and descriptors
    model = FacenetModel()
    nodes_list = []
    descriptors_list = []
    # print(image_paths)
    # print(len(image_paths))
    one_face_img_paths = []
    for image_path in image_paths:
        picture = img_to_array(image_path)
        # detect all faces in an image
        # returns a tuple of (boxes, probabilities, landmarks)
        # assumes ``pic`` is a numpy array of shape (R, C, 3) (RGB is the last dimension)   
        # If N faces are detected then arrays of N boxes, N probabilities, and N landmark-sets are returned.
        boxes, probabilities, landmarks = model.detect(picture)
        # Crops the image once for each of the N bounding boxes and produces a shape-(512,) descriptor for that face.
        # If N bounding boxes were supplied, then a shape-(N, 512) array is returned, 
        # corresponding to N descriptor vectors
        assert len(boxes) != 0, "This photo has no faces detected"
        for i in range(boxes.shape[0]):
            if not probabilities[i] > 0.9:
                boxes = np.delete(boxes,i,0) # delete probs and boxes for detections under tolerance
                probabilities = np.delete(probabilities,i,0)
        if len(boxes) > 1:
            print(image_path, " contained more than one face- skipping image")
            continue
        image_descriptor = model.compute_descriptors(picture, boxes)
        descriptors_list.append(image_descriptor)
        one_face_img_paths.append(image_path)
    
    # adjacency matrix
    face_count = len(descriptors_list)
    # print("face_count", face_count)
    adjacency_matrix = np.zeros((face_count, face_count)) # initialize N x N array
    # looping over adjacency matrix
    for row in range(face_count):
        for column in range(face_count):
            descriptor_distance = cos_dist(descriptors_list[row].ravel(), descriptors_list[column].ravel())
            if (descriptor_distance < cos_threshold) and (row != column):
                weighting_function = 1 / (descriptor_distance ** 2)
                adjacency_matrix[row, column] = weighting_function


    for face_index, image_path in enumerate(one_face_img_paths):
        # print(face_index, image_path) 
        #           ID,     neighbors, descriptor
        node = Node(ID = face_index, 
                    neighbors = np.nonzero(adjacency_matrix[face_index])[0], 
                    descriptor = descriptors_list[face_index],
                    file_path = image_path)
        nodes_list.append(node)

    return (nodes_list, adjacency_matrix)