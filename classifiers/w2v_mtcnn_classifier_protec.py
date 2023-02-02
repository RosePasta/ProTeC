import os
import util.old_util as own_util
import util.old_model as own_model
import shutil
import numpy as np
from numpy import zeros
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow import keras

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D,MaxPool2D,\
     Dense, Input, Flatten, Concatenate, Reshape, Conv2D
from tensorflow.keras import initializers
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import sys
import pickle
import random
import sys
import time
import warnings
warnings.filterwarnings(action='ignore')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

drop_out=0.5                
l2_reg_lambda=0.0
filter_sizes = [3, 4, 5]
num_filters = 128
batch = 16
lr = 0.001

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_bugs(file_path,max_length):
    all_terms = set()
    lines = open(file_path,"r",encoding="utf8").readlines()
    ids = []
    short_texts = []
    long_texts = []
    labels = []
    for line in lines:
        tokens = line.replace("\n","").split("\t")
        b_id = "B"+tokens[0]
        short = tokens[1]
        long = tokens[2]
        text = short +" "+long
        if len(long) < 2:
            long = "<PAD>"
        if len(long.split(" ")) > max_length:
            long = ' '.join(long.split(" ")[:max_length])
        
        label = tokens[3]
        if label == "PB-TF":
            label = "PB"
        
        ids.append(b_id)
        short_texts.append(short)
        long_texts.append(long)
        labels.append(label)
        tokens = set(text.split(" "))
        all_terms.update(tokens)
    return ids, short_texts, long_texts, labels, all_terms

def load_files(file_path,max_length):    
    all_terms = set()
    lines = open(file_path,"r",encoding="utf8").readlines()
    ids = []
    short_texts = []
    long_texts = []
    labels = []
    for line in lines:
        tokens = line.replace("\n","").split("\t")
        b_id = "F"+tokens[0]
        short = tokens[1]
        long = tokens[2]
        text = short +" "+long
        label = tokens[3]
        if label == "PF":
            label = "PB"
        if label == "TF":
            label = "TB"
        if len(long) < 2:
            long = "<PAD>"
        if len(long.split(" ")) > max_length:
            long = ' '.join(long.split(" ")[:max_length])
        ids.append(b_id)
        short_texts.append(short)
        long_texts.append(long)
        labels.append(label)
        tokens = set(text.split(" "))
        all_terms.update(tokens)
    return ids, short_texts, long_texts, labels, all_terms



base_path = own_util.get_dataset_path()

w2v_dim, w2v_path, drop_out, l2_reg_lambda, filter_sizes, num_filters, batch, lr = own_util.get_cnn_parameters()
w2v_embedding = own_util.load_w2v(w2v_path)
print(len(w2v_embedding), "W2V load finish")

model_name = "MTCNN"
verbose = 1

max_length_base, epoch, batch, patience, val_rate, lr = own_util.get_dl_parameters()
max_length_threshold = max_length_base

base_path = "D:\\ProTeC\\dataset\\traininig_dataset\\"
base_path ="D:\\ExpData\\ProTeC_SplittedData\\"

executed_file_name = sys.argv[0].split("/")[-1].replace(".py","")

result_file_name = executed_file_name+"_"+str(time.time())
writer = open("./results/"+result_file_name+".txt","w",encoding="utf8")
writer.close()

iterator = 0

total_bugs = set() 
datasets = os.listdir(base_path)
for dataset in datasets:    
    projects = os.listdir(base_path+dataset)
    for project in projects:
        
        all_bugs = 0
        all_pb_num = 0
        correct_pb = 0
        all_tb_num = 0
        correct_tb = 0
        for i in range(0, 10):
            
            if os.path.exists(base_path+dataset+"\\"+project+"\\"+str(i)+"_bugs_train.txt") == False:
                print(dataset, project, "no file",base_path+dataset+"\\"+project+"\\"+str(i)+"_bugs_train.txt")
                continue

            # 0. Prepare dataset                
            train_ids, train_texts_short,train_texts_long, train_labels, all_terms = load_files(base_path+dataset+"\\"+project+"\\"+str(i)+"_files_train.txt", max_length_base)                
            split_index = len(train_labels)
            bug_ids = train_ids 
            bug_texts_short = train_texts_short
            bug_texts_long = train_texts_long 
            bug_labels = train_labels 
            
            # 1. For Padding
            length_list = [len(text.split(" ")) for text in bug_texts_long]
            tokenizer = Tokenizer(oov_token="<OOV>")
            tokenizer.fit_on_texts(bug_texts_long)
            max_length = max(length_list)
            if max_length > max_length_threshold:
                max_length = max_length_threshold
            if max_length < max(filter_sizes):
                max_length = max(filter_sizes)
            length_list_short = [len(text.split(" ")) for text in bug_texts_short]
            max_length_short = max(length_list_short)
            if max_length_short < max(filter_sizes):
                max_length_short = max(filter_sizes)

            path_name = "./model/"+executed_file_name+"_tokenizer.pickle"
            with open(path_name, "wb") as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # print(dataset, project, i, "TOKENIZE FINISH")

            # 2. Encoding           
            encoded_bugs_short = tokenizer.texts_to_sequences(bug_texts_short)
            padded_bugs_short = pad_sequences(encoded_bugs_short, maxlen=max_length_short, padding='post', dtype='float32')
            X_short = padded_bugs_short
            X_short = np.array(X_short, np.int32)
                            
            encoded_bugs = tokenizer.texts_to_sequences(bug_texts_long)
            padded_bugs = pad_sequences(encoded_bugs, maxlen=max_length, padding='post', dtype='float32')
            X = padded_bugs
            X = np.array(X, np.int32)
                        
            y = []
            for index in range(len(bug_labels)):
                if bug_labels[index] == "PB":
                    y.append([1, 0])
                elif bug_labels[index] == "TB":
                    y.append([0, 1])
            y = np.array(y, np.int32)

            # 3. Embedding vector
            vocab_size = len(tokenizer.word_index)+1        
            w2v_embedding_matrix = zeros((vocab_size, w2v_dim))
            for word, wi in tokenizer.word_index.items():
                embedding_vector = w2v_embedding.get(word)
                if embedding_vector is not None:
                    w2v_embedding_matrix[wi] = embedding_vector
            
            # 4. Split datasets
            train_bug_id,test_bug_id = bug_ids, bug_ids
            X_bug_train, X_bug_train_short, y_bug_train= X, X_short, y
            X_bug_test, X_bug_test_short, y_bug_test =  X, X_short, y

            X_bug_train = np.array(X_bug_train, np.int32)
            X_bug_train_short = np.array(X_bug_train_short,np.int32)
            X_bug_test = np.array(X_bug_test, np.int32)
            X_bug_test_short = np.array(X_bug_test_short,np.int32)
                                        
            X_bug_train, XValidation, X_bug_train_short, XValidation_short, y_bug_train, YValidation = train_test_split(X_bug_train, X_bug_train_short, y_bug_train,stratify=y_bug_train,test_size=val_rate)
            
            # print(dataset, project, i, "TRAIN START")
            
            model = own_model.get_textcnn_double(drop_out, l2_reg_lambda, filter_sizes, num_filters, vocab_size, w2v_dim, w2v_embedding_matrix, max_length, max_length_short, lr)
            callbacks = [EarlyStopping(monitor='val_loss',patience=patience),
                        ModelCheckpoint(filepath="./model/dump_"+executed_file_name, monitor='val_loss',save_best_only=True)]
            model.fit([X_bug_train, X_bug_train_short], y_bug_train,  epochs=epoch, batch_size=batch, verbose = verbose,callbacks=callbacks, validation_data = ([XValidation,XValidation_short],YValidation) )    
            trained_model = keras.models.load_model("./model/dump_"+executed_file_name)


            # Prepare tuning dataset
            train_ids, train_texts_short,train_texts_long, train_labels, all_terms = load_bugs(base_path+dataset+"\\"+project+"\\"+str(i)+"_bugs_train.txt", max_length_base)
            test_ids, test_texts_short,test_texts_long, test_labels, all_terms = load_bugs(base_path+dataset+"\\"+project+"\\"+str(i)+"_bugs_test.txt", max_length_base)
            split_index = len(train_labels)
            bug_ids = train_ids + test_ids
            bug_texts_short = train_texts_short + test_texts_short
            bug_texts_long = train_texts_long + test_texts_long
            bug_labels = train_labels + test_labels
            for b_id in test_ids:
                total_bugs.add(dataset+"_"+project+"_"+b_id)


            # Encoding tuning dataset
            encoded_bugs_short = tokenizer.texts_to_sequences(bug_texts_short)
            padded_bugs_short = pad_sequences(encoded_bugs_short, maxlen=max_length_short, padding='post', dtype='float32')
            X_short = padded_bugs_short
            X_short = np.array(X_short, np.int32)
                            
            encoded_bugs = tokenizer.texts_to_sequences(bug_texts_long)
            padded_bugs = pad_sequences(encoded_bugs, maxlen=max_length, padding='post', dtype='float32')
            X = padded_bugs
            X = np.array(X, np.int32)
                        
            y = []
            for index in range(len(bug_labels)):
                if bug_labels[index] == "PB":
                    y.append([1, 0])
                elif bug_labels[index] == "TB":
                    y.append([0, 1])
            y = np.array(y, np.int32)


            # Split dataset                
            train_bug_id,test_bug_id = bug_ids[:split_index], bug_ids[split_index:]       
            X_bug_train, X_bug_train_short, y_bug_train= X[:split_index], X_short[:split_index], y[:split_index]
            X_bug_test, X_bug_test_short, y_bug_test = X[split_index:], X_short[split_index:], y[split_index:]

            X_bug_train = np.array(X_bug_train, np.int32)
            X_bug_train_short = np.array(X_bug_train_short,np.int32)
            X_bug_test = np.array(X_bug_test, np.int32)
            X_bug_test_short = np.array(X_bug_test_short,np.int32)
                                        
            X_bug_train, XValidation, X_bug_train_short, XValidation_short, y_bug_train, YValidation = train_test_split(X_bug_train, X_bug_train_short, y_bug_train,stratify=y_bug_train,test_size=val_rate)
            

            # Tuinng the model              
            trained_model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr*0.1),metrics=['acc'])
            callbacks = [EarlyStopping(monitor='val_loss',patience=patience),
                    ModelCheckpoint(filepath="./model/dump_"+executed_file_name, monitor='val_loss',save_best_only=True)]
            trained_model.fit([X_bug_train, X_bug_train_short], y_bug_train,  epochs=epoch, batch_size=batch, verbose = verbose, callbacks=callbacks, validation_data = ([XValidation,XValidation_short],YValidation) )    
            trained_model = keras.models.load_model("./model/dump_"+executed_file_name)


            y_predict = trained_model.predict([X_bug_test, X_bug_test_short])
            # print(dataset, project, i, "TRAIN FINISH")

            correct = 0
            for t_i in range(len(y_bug_test)):
                y_real = np.argmax(y_bug_test[t_i], axis=0)   
                y_pred = np.argmax(y_predict[t_i], axis=0)
                if y_real == 0:
                    all_pb_num += 1
                    all_bugs += 1
                    if y_pred == 0:
                        correct_pb += 1
                        correct += 1
                if y_real == 1:
                    all_tb_num += 1
                    all_bugs += 1
                    if y_pred == 1:
                        correct_tb += 1
                        correct += 1
                
                bug_id = str(test_bug_id[t_i])
                writer = open("./results/"+result_file_name+".txt","a",encoding="utf8")
                identifier = dataset+"\t"+project+"\t"+"\t"+bug_id
                real_type = "PB"
                if y_real == 1:
                    real_type = "TB"
                pred_type = "PB"
                if y_pred == 1:
                    pred_type = "TB"
                writer.write(model_name+"_MT_PROTEC_"+str(iterator)+"\t"+str(i)+"\t"+identifier+"\t"+real_type+"\t"+pred_type+"\n")
                writer.close()

            f1_pbr, f1_tbr = own_util.evaluation(correct_pb, all_pb_num, correct_tb, all_tb_num)
            print(model_name+"_MT_PROTEC_"+str(iterator)+"\t"+str(i), dataset, project, correct_pb, '/', all_pb_num, correct_tb, '/', all_tb_num, round(f1_pbr,3), round(f1_tbr,3))


print(len(total_bugs))