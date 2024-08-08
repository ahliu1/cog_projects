from copyreg import pickle
import pickle
import numpy as np
from coco import load_coco
import io
import matplotlib.pyplot as plt
import requests
from PIL import Image
from embedding import *

class query_database:
    """Matches image id to embedding vectors. Computes top-k nearest matches to query. Downloads and displays those images"""
    def __init__(self, directory) -> None:
        "Initializes the database"
        self.directory = directory
        self.open_database()

    def open_database(self):
        """Opens the pickle file and allows for the database to be accessed"""
        with open(self.directory,"rb") as f:
            self.database = pickle.load(f)
            f.close()

    def save_database(self):
        """Saves updates to the databse and closes the pickle file"""
        with open(self.directory,"wb") as f:
            pickle.dump(self.database,f)
            f.close()
    
    def populate(self, data, matrix):
        """Populates the database(dictionary) with image IDs as keys and embedding matrix as values. Then saves and reopens the database
        Parameters:
            data: class containing information (descriptors, ids, urls etc.) related to the images
            model: trained model that can convert image descriptors to image embeddings
        """
        for image in range(len(data.coco_metadata["images"])):
            id = data.coco_metadata["images"][image]["id"]
            try:
                self.database[id] = (data.resnet18_features[id]) @ matrix 
            except KeyError:
                pass
        self.save_database()
        self.open_database()

    def query(self, caption, num_img, data):
        """Given a user input (string), converts that to a caption embeddings, queries the database for top-k image matches and returns the ids of those images
        Parameters:
            caption: str
                Can be any string that the user wants to query for images
            num_img: int
                The number of images that most closely resemble the caption provided
        Return:
            image_ID: int
                The image IDs of the closest matches.
        """
        caption_vector = embedding_text(tokenizing(caption), pickle.load(open("datasets/idfs.pkl","rb")), data)
        caption_vector = caption_vector / np.linalg.norm(caption_vector)
        ids = np.array(list(self.database.keys()))
        img_embeddings = np.array(list(self.database.values()))
        img_embed = np.reshape(img_embeddings, (-1, 200))
        dot_prods = np.dot(img_embed, caption_vector)
        sort_idxs = np.argsort(dot_prods)
        # print(sort_idxs[-num_img:])
        top_k_ids = ids[sort_idxs[-num_img:]]
        return top_k_ids

    def display_img(self, img_ids, data):
        """Uses the URLs associated with each image ID to download the image and display them.
        Parameters:
            img_ids: list of image IDs
            data: class containing information (descriptors, ids, urls etc.) related to the images
        """
        for num, id in enumerate(img_ids):
            img_url = data.imageID_to_URL(id)
            response = requests.get(img_url[0])
            im = Image.open(io.BytesIO(response.content))
            plt.subplot(1, len(img_ids), num+1)
            plt.imshow(im)
        plt.show()
