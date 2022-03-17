# BMI 203 Project 7: Neural Network

# Import necessary dependencies here


# TODO: Write your test functions and associated docstrings below.

def test_forward():
    pass


def test_single_forward():
    pass


def test_single_backprop():
    pass


def test_predict():
    pass


def test_binary_cross_entropy():
    pass


def test_binary_cross_entropy_backprop():
    pass


def test_mean_squared_error():
    pass


def test_mean_squared_error_backprop():
    pass


def test_one_hot_encode():
    seq=list('ATCG')
    encoded=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    assert one_hot_encode_seqs(seq) == encoded


def test_sample_seqs():
    # class one is unbalanced to class two 
    seqs=['a','b','c','d','e','f','g']
    labels=[1,1,0,0,0,0,0]
    
    sampled_seqs, sampled_labels = sample_seqs(seqs, labels)
    
    # check that the dataset was upsampled to have length 10 (len(neg_class)x2=10)
    n_class_0=5
    assert len(sampled_seqs)==5
    
    # check that class balance = 50:50
    n_class_0=[x for x in sampled_labels if x==0]
    n_class_1=[x for x in sampled_labels if x==1]
    assert n_class_0==n_class_1