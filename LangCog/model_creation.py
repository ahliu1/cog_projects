# https://rsokl.github.io/CogWeb/Language/SemanticImageSearch.html#Training

import mygrad as mg
import mynn
import numpy as np

from mygrad.nnet.initializers import glorot_normal
from mynn.layers.dense import dense
from mygrad.nnet.losses.margin_ranking_loss import margin_ranking_loss
from mynn.optimizers.sgd import SGD
from mynn.optimizers.adam import Adam

import matplotlib.pyplot as plt


class Model:
    """
    MyNN model that embeds image descriptors: d(img) => w(img)
    Extract sets of (caption-ID, image-ID, confusor-image-ID) triples (training and validation sets)
    Our model simply consists of one matrix that maps a shape-(512,) 
    image descriptor into a shape-(D=200,) embedded vector, and normalizes that vector.
    """

    def __init__(self, D_full=512, D_hidden=200):
        """ This initializes all of the layers in our model, and sets them
        as attributes of the model.
        
        Parameters
        ----------
        D_full : int
            The size of the inputs.
            
        D_hidden : int
            The size of the 'bottleneck' layer (i.e., the reduced dimensionality).
        """
        self.dense1 = dense(D_full, D_hidden, weight_initializer=glorot_normal, bias=False)

    def __call__(self, x):
        '''Passes data as input to our model, performing a "forward-pass".
        
        This allows us to conveniently initialize a model `m` and then send data through it
        to be classified by calling `m(x)`.
        
        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor], shape=(M, D_full)
            A batch of data consisting of M pieces of data,
            each with a dimentionality of D_full.
            
        Returns
        -------
        mygrad.Tensor, shape=(M, D_full)
            The model's prediction for each of the M pieces of data.
        '''
        out =  self.dense1(x) 
        return out / np.linalg.norm(out, axis=1).reshape(-1, 1)
        
    def assert_parameters(self,parameters1):
        self.dense1.parameters = (mg.tensor(parameters1))
        # self.parameters[0].data = parameters1

    @property
    def parameters(self):
        """ A convenience function for getting all the parameters of our model.
        
        This can be accessed as an attribute, via `model.parameters` 
        
        Returns
        -------
        Tuple[Tensor, ...]
            A tuple containing all of the learnable parameters for our model """
        return self.dense1.parameters
