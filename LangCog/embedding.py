from collections import Counter, defaultdict
from math import log10
from cogworks_data.language import get_data_path
import re, string
from gensim.models import KeyedVectors
import numpy as np

def tokenizing(text):
    """
    Takes in captions and queries and lowercases text, removes punctuation, and tokenizes words from white space

    Parameters
    ------------
    Text: String "token1 token 2..."

    Returns
    ------------
    Tokens: List [token1, token2, ...]
    """
    punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
    stripped_punc = punc_regex.sub('', text)
    lowered_and_split = stripped_punc.lower().split()
    tokens = list(lowered_and_split)
    return tokens

def idf(captions):
    '''
    Takes in the captions for all documenets/captions,
    compute the IDF for each unique term 

    Parameters
    ----------
    captions : List[str]

    Returns
    -------
    dict{word : idf_value}
    '''
    N = len(captions)
    
    tally = Counter()
    
    for caption in captions:
        token_capt = tokenizing(caption)
        unique_words = set(token_capt)
        tally.update(unique_words)

    idf_dict = {word : log10(N/cnt) for word, cnt in tally.items()}

    return idf_dict


def embedding_text(tokens, idf_dict, data):
    """
    Takes in a list of tokens and forms an IDF-weighted sum of the GloVe embedding for each token

    Parameters
    ------------
    Tokens: List of strings [token1, token2, ...]
    idf_dict: Dictionary that makes token to idf

    Returns
    ------------
    Embeddings: A shape (200,) numpy array
    """
    embeddings = np.zeros(200)
    for token in tokens:
        try:
            embed = data.gloves[token]
            weighted_vector = idf_dict[token] * embed
        except KeyError:
            weighted_vector = np.zeros(200)
        embeddings += weighted_vector
    return embeddings
    
    