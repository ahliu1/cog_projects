'''
Handles loading and managing the coco database
'''
from cogworks_data.language import get_data_path
from gensim.models import KeyedVectors
from pathlib import Path
import json
import pickle

from sklearn.tree import export_text
from embedding import *
import numpy as np
import random
from embedding import embedding_text, tokenizing


class load_coco:
    '''
    Class to handle COCO intialization
    '''

    def __init__(self):
        self.metadata = get_data_path("captions_train2014.json")
        with Path(self.metadata).open() as f:
            self.coco_metadata = json.load(f)
        self.load_glove_embeddings()
        self.load_image_descriptors()

    def load_glove_embeddings(self):
        '''
        Handles loading the gloves unnormalized
        '''
        self.glove_filename = "glove.6B.200d.txt.w2v"
        self.gloves = KeyedVectors.load_word2vec_format(get_data_path(self.glove_filename), binary=False)

    def load_image_descriptors(self):
        '''
        image-ID -> shape-(512,) descriptor. ResNet descriptors for images in the COCO dataset
        Be sure to discard any COCO images that do not have descriptors with them
        '''
        with Path(get_data_path('resnet18_features.pkl')).open('rb') as f:
            self.resnet18_features = pickle.load(f)

    @property
    def parameters(self):
        '''
        Returns all data related to the COCO dataset and ResNet descriptors
        '''
        return self.coco_metadata, self.gloves, self.resnet18_features

    def normalize_descriptors(self,descriptors_array):
        '''
        Takes in an array of descriptors and returns an array of normalized descriptors
        '''
        pass

    def save_embedded_caption_and_img_descriptors(self):
        """
        Returns:
            caption_embeddings: nd.array, (# of captions, 200)
            img_descriptors: nd.array, (# of captions, 512)
            # captions > images
            There may be duplicate img_descriptors because there are multiple captions per image
        """
        print("returning captions and imageIDs")
        caption_embeddings = []
        img_descriptors = []
        IDFs = self.init_idf()
        print("annotations", len(self.coco_metadata["annotations"]))
        print("images", len(self.coco_metadata["images"]))
        for img in self.coco_metadata["annotations"]:
            try:
                img_descriptors.append(self.resnet18_features[img["image_id"]].ravel())
                caption_embeddings.append(embedding_text(tokenizing(img["caption"]),
                                                                    IDFs,self.gloves))
            except KeyError:
                continue
            except AttributeError:
                continue
        caption_embeddings = np.array(caption_embeddings)
        img_descriptors = np.array(img_descriptors)
        print("here", caption_embeddings.shape)
        print("here", img_descriptors.shape)
        np.save("datasets/caption_embeddings.npy", caption_embeddings)
        np.save("datasets/img_descriptors.npy", img_descriptors)
        
    def init_idf(self):
        captions = []
        for img in self.coco_metadata["annotations"]:
            try:
                captions.append(img["caption"])
            except KeyError:
                continue
        with open("datasets/idfs.pkl", mode="w+b") as f:
            IDFs = idf(captions)
            pickle.dump(IDFs,f)
        return IDFs

    def get_embedded_caption_and_img_descriptors(self):
        caption_embeddings = np.load("datasets/caption_embeddings.npy")
        img_descriptors = np.load("datasets/img_descriptors.npy")
        return caption_embeddings, img_descriptors

    def imageID_to_URL(self, imageID):
        return [i["coco_url"] for i in self.coco_metadata["images"] if i["id"] == imageID]

    def return_imageID_from_captionID(self, captionID):
        return [i["image_id"] for i in self.coco_metadata["annotations"] if i["id"] == captionID]

    def return_captionID_from_imageID(self, imageID):
        """Return one captionID out of multiple from an imageID"""
        return random.choice([i["id"] for i in self.coco_metadata["annotations"] if i["image_id"] == imageID])

    def captionID_to_caption(self, captionID):
        return [i["caption"] for i in self.coco_metadata["annotations"] if i["id"] == captionID]

    def return_imageID_captions(self, imageID):
        return [i["caption"] for i in self.coco_metadata["annotations"] if i["image_id"] == imageID]
