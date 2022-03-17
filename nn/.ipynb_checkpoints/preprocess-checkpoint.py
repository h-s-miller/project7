# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
from random import randrange 


# Defining preprocessing functions
def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one hot encoding of a list of nucleic acid sequences
    for use as input into a fully connected neural net.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence
            length due to the one hot encoding scheme for nucleic acids.

            For example, if we encode 
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    """
    encodings=[]
    encode_dict={'A': [1, 0, 0, 0],
                'T': [0, 1, 0, 0],
                'C': [0, 0, 1, 0],
                'G': [0, 0, 0, 1]}
    for x in seq_arr: 
        encodings.extend(encode_dict[x])
        
    return encodings


def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample your sequences to account for class imbalance. 
    Consider this as a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # split into samples based on labels
    positive_seqs=[x for x,y in zip(seqs,labels) if y==1]
    neg_seqs=[x for x,y in zip(seqs,labels) if y==0]

    # want 50:50 ratio of postive to negative samples, ssince there are a small amount of positive samples, want to upsample the positive samples by randomly sampling with replacement 
    
    new_pos=[]
    for i in range(len(neg_seqs)):
        x=randrange(0,len(positive_seqs)) # random sampling 
        new_pos.append(positive_seqs[x])

    sampled_seqs=new_pos+neg_seqs
    sampled_labels=[1]*len(new_pos)+[0]*len(neg_seqs)
    return sampled_seqs, sampled_labels
