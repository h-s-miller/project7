# BMI 203 Project 7: Neural Network


# Importing Dependencies
from typing import List


#
# DO NOT MODIFY ANY OF THESE FUNCTIONS THEY ARE ALREADY COMPLETE!
#


# Defining I/O functions
def read_text_file(filename: str) -> List[str]:
    """
    This function reads in a text file into a list of sequences.

    Args:
        filename: str
            Filename, should end in .txt.

    Returns:
        arr: List[str]
            List of sequences.
    """
    with open(filename, "r") as f:
        seq_list = [line.strip() for line in f.readlines()]
    return seq_list


def read_fasta_file(filename: str) -> List[str]:
    """
    This function reads in a fasta file into a numpy array of sequence strings.

    Args:
        filename: str
            File path and name of file, filename should end in .fa or .fasta.

    Returns:
        seqs: List[str]
            List of sequences.
    """
    with open(filename, "r") as f:
        seqs = []
        seq = ""
        for line in f:
            if line.startswith(">"):
                seqs.append(seq)
                seq = ""
            else:
                seq += line.strip()
        seqs = seqs[1:]
        return seqs
