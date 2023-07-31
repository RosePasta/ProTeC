import re
from numpy import asarray

def evaluation(correct_pb, all_pb, correct_tb, all_tb):
    # for PBR
    f1_score_pbr = 0
    if correct_pb > 0:
        tp = correct_pb
        fn = all_pb - correct_pb
        fp = all_tb - correct_tb
        tn = correct_tb
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        f1_score_pbr = 2 * ((recall * precision) /(recall + precision))

    # for TBR
    f1_score_tbr = 0
    if correct_tb > 0:
        tp = correct_tb
        fn = all_tb - correct_tb
        fp = all_pb - correct_pb
        tn = correct_pb
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        f1_score_tbr = 2 * ((recall * precision) /(recall + precision))
    return f1_score_pbr, f1_score_tbr

def load_w2v(path):    
    glove_embedding = dict()
    f = open(path, "r", encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        glove_embedding[word] = coefs
    f.close()
    # print('Loaded %s GLOVE word vectors.' % len(glove_embedding), emb_size)
    return glove_embedding

def get_ml_parameters():
    max_length_base = 10000
    return max_length_base


def get_dl_parameters():
    max_length_base = 1000
    epoch = 30
    batch = 16
    patience = 5
    val_rate = 0.1
    lr = 0.001
    return max_length_base, epoch, batch, patience, val_rate, lr


def get_cnn_parameters():
    w2v_dim = 300
    w2v_path = "D:\\ExpData\\Pretrained_W2V\\glove_word_embeddings\\glove.6B.300d.txt"
    drop_out=0.5                
    l2_reg_lambda=0.0
    filter_sizes = [3, 4, 5]
    num_filters = 128
    batch = 16
    lr = 0.001
    return w2v_dim, w2v_path, drop_out, l2_reg_lambda, filter_sizes, num_filters, batch, lr

def get_dataset_path():        
    base_path ="../traininig_dataset/"
    return base_path
