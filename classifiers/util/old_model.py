
import numpy as np
from numpy import asarray
from numpy import zeros

import tensorflow as tf
from tensorflow import keras

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

import json
import io
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D,MaxPool2D,\
     Dense, Input, Flatten, Concatenate, Reshape, Conv2D
from tensorflow.keras import initializers

import tensorflow_addons.metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers

import sys

def get_textcnn_double(drop_out, l2_reg_lambda, filter_sizes, num_filters, vocab_size, emb_size, glove_embedding_matrix, max_length_long, max_length_short, lr):

    input_x1 = Input(shape=(max_length_long,), name='input_x1')
    # embedding layer
    embedding1 = Embedding(vocab_size, emb_size, weights=[glove_embedding_matrix], name='embedding1', trainable=True)(input_x1)
    expend_shape1 = [embedding1.get_shape().as_list()[1], embedding1.get_shape().as_list()[2], 1]
    # embedding_chars = K.expand_dims(embedding, -1)    # 4D tensor [batch_size, seq_len, embeding_size, 1] seems like a gray picture
    embedding_chars1 = Reshape(expend_shape1)(embedding1)        

    
    input_x2 = Input(shape=(max_length_short,), name='input_x2')
    # embedding layer
    embedding2 = Embedding(vocab_size, emb_size, weights=[glove_embedding_matrix], name='embedding2', trainable=True)(input_x2)
    expend_shape2 = [embedding2.get_shape().as_list()[1], embedding2.get_shape().as_list()[2], 1]
    # embedding_chars = K.expand_dims(embedding, -1)    # 4D tensor [batch_size, seq_len, embeding_size, 1] seems like a gray picture
    embedding_chars2 = Reshape(expend_shape2)(embedding2)       

    # conv->max pool
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        conv1 = Conv2D(filters=num_filters, 
                        kernel_size=[filter_size, emb_size],
                        strides=1,
                        padding='valid',
                        activation='relu',
                        kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                        bias_initializer=initializers.constant(value=0.1),
                        name=('conv1_%d' % filter_size))(embedding_chars1)
        # print("conv-%d: " % i, conv)
        max_pool1 = MaxPool2D(pool_size=[max_length_long - filter_size + 1, 1],
                            strides=(1, 1),
                            padding='valid',
                            name=('max_pool1_%d' % filter_size))(conv1)
        pooled_outputs.append(max_pool1)
        # print("max_pool-%d: " % i, max_pool)
        
    for i, filter_size in enumerate(filter_sizes):
        conv2 = Conv2D(filters=num_filters, 
                        kernel_size=[filter_size, emb_size],
                        strides=1,
                        padding='valid',
                        activation='relu',
                        kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                        bias_initializer=initializers.constant(value=0.1),
                        name=('conv2_%d' % filter_size))(embedding_chars2)
        # print("conv-%d: " % i, conv)
        max_pool2 = MaxPool2D(pool_size=[max_length_short - filter_size + 1, 1],
                            strides=(1, 1),
                            padding='valid',
                            name=('max_pool2_%d' % filter_size))(conv2)
        pooled_outputs.append(max_pool2)
        # print("max_pool-%d: " % i, max_pool)
    # combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes) * 2
    h_pool = Concatenate(axis=-1)(pooled_outputs)
    h_pool_flat = Reshape([num_filters_total])(h_pool)
    # add dropout
    fcn = Dense(num_filters_total, kernel_initializer='glorot_normal', 
                    bias_initializer=initializers.constant(np.log([lr])),activation='relu')(h_pool_flat)
    dropout = Dropout(drop_out)(fcn)
    
    # output layer
    output = Dense(2,
                    kernel_initializer='glorot_normal',
                    # bias_initializer=initializers.constant(0.1),
                    bias_initializer=initializers.constant(np.log([lr])),
                    activation='softmax',
                    name='output')(dropout)
    
    model = Model(inputs=[input_x1, input_x2], outputs=output)
    model.compile(loss='categorical_crossentropy',
                #loss='categorical_crossentropy',
                optimizer=optimizers.Adam(lr),
                metrics=['AUC',tensorflow_addons.metrics.F1Score(num_classes=2)])
    return model


def get_textcnn(drop_out, l2_reg_lambda, filter_sizes, num_filters, vocab_size, emb_size, glove_embedding_matrix, max_length, lr):

    input_x = Input(shape=(max_length,), name='input_x')
    # embedding layer
    embedding = Embedding(vocab_size, emb_size, weights=[glove_embedding_matrix], name='embedding', trainable=True)(input_x)
    expend_shape = [embedding.get_shape().as_list()[1], embedding.get_shape().as_list()[2], 1]
    # embedding_chars = K.expand_dims(embedding, -1)    # 4D tensor [batch_size, seq_len, embeding_size, 1] seems like a gray picture
    embedding_chars = Reshape(expend_shape)(embedding)        

    # conv->max pool
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        conv = Conv2D(filters=num_filters, 
                        kernel_size=[filter_size, emb_size],
                        strides=1,
                        padding='valid',
                        activation='relu',
                        kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                        bias_initializer=initializers.constant(value=0.1),
                        name=('conv_%d' % filter_size))(embedding_chars)
        # print("conv-%d: " % i, conv)
        max_pool = MaxPool2D(pool_size=[max_length - filter_size + 1, 1],
                            strides=(1, 1),
                            padding='valid',
                            name=('max_pool_%d' % filter_size))(conv)
        pooled_outputs.append(max_pool)
        # print("max_pool-%d: " % i, max_pool)
    
    # combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = Concatenate(axis=-1)(pooled_outputs)
    h_pool_flat = Reshape([num_filters_total])(h_pool)
    # add dropout
    fcn = Dense(num_filters_total, kernel_initializer='glorot_normal', 
                    bias_initializer=initializers.constant(np.log([lr])),activation='relu')(h_pool_flat)
    dropout = Dropout(drop_out)(fcn)
    # dropout = Dropout(drop_out)(h_pool_flat)
    
    # output layer
    output = Dense(2,
                    kernel_initializer='glorot_normal',
                    # bias_initializer=initializers.constant(0.1),
                    bias_initializer=initializers.constant(np.log([lr])),
                    activation='softmax',
                    name='output')(dropout)
    
    model = Model(inputs=input_x, outputs=output)
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adam(lr),
                metrics=['acc',tensorflow_addons.metrics.F1Score(num_classes=2)])
    return model

