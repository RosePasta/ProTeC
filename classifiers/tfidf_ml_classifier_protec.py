import os
import util.old_util as own_util
import shutil
import numpy as np
from numpy import asarray
from numpy import zeros

import sys
import pickle
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

import random
import time

def load_bugs(file_path,which_field, max_length):
    all_terms = set()
    lines = open(file_path,"r",encoding="utf8").readlines()
    ids = []
    texts = []
    labels = []
    for line in lines:
        tokens = line.replace("\n","").split("\t")
        b_id = "B"+tokens[0]
        text = tokens[1]+" "+tokens[2]
        if which_field == "SHORT":
            text = tokens[1]
        elif which_field == "LONG":
            text = tokens[2]
        label = tokens[3]
        if label == "PB-TF":
            label = "PB"
        if len(text.split(" ")) > max_length:
            text = ' '.join(text.split(" ")[:max_length])
        ids.append(b_id)
        texts.append(text)
        labels.append(label)
        tokens = set(text.split(" "))
        all_terms.update(tokens)
    return ids, texts, labels, all_terms

def load_files(file_path,which_field, max_length):    
    all_terms = set()
    lines = open(file_path,"r",encoding="utf8").readlines()
    ids = []
    texts = []
    labels = []
    for line in lines:
        tokens = line.replace("\n","").split("\t")
        b_id = "F"+tokens[0]
        text = tokens[1]+" "+tokens[2]
        if which_field == "SHORT":
            text = tokens[1]
        elif which_field == "LONG":
            text = tokens[2]
        label = tokens[3]
        if label == "PF":
            label = "PB"
        if label == "TF":
            label = "TB"
        if len(text.split(" ")) > max_length:
            text = ' '.join(text.split(" ")[:max_length])
        ids.append(b_id)
        texts.append(text)
        labels.append(label)
        tokens = set(text.split(" "))
        all_terms.update(tokens)
    return ids, texts, labels, all_terms


iterator = 0
model_name = "MNB" # "MNB", "SVM"
training_set = "PROTEC"
which_field = "ALL" # "ALL", "SHORT", "LONG"


base_path = own_util.get_dataset_path()
max_length_base = own_util.get_ml_parameters()


executed_file_name = sys.argv[0].split("/")[-1].replace(".py","")

model_names = ["MNB", "SVM"]
which_fields = ["ALL", "SHORT", "LONG"]
set_types = ["BR", "SF"]

result_file_name = executed_file_name+"_"+str(time.time())
writer = open("./results/"+result_file_name+".txt","w",encoding="utf8")
writer.close()

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

            # 0. Prepare training dataset                                                            
            train_ids, train_texts, train_labels, train_terms  = load_files(base_path+dataset+"\\"+project+"\\"+str(i)+"_files_train.txt", which_field, max_length_base)                            
            split_index = len(train_labels)
            bug_ids = train_ids
            bug_texts = train_texts
            bug_labels = train_labels

            # 1. Vectorize the documents
            max_length = len(train_terms)
            vec = TfidfVectorizer(max_features=max_length)
            X_bug_train = vec.fit_transform(train_texts).toarray()            
            max_length = len(vec.get_feature_names())
            y_bug_train = [1 if tmp_y=='TB' else 0 for tmp_y in train_labels]
            y_bug_train = np.array(y_bug_train, np.int32)                            
            
            train_bug_id,test_bug_id = bug_ids[:split_index], bug_ids[split_index:]                       

            # 2. Training source model
            # print(dataset, project, i, "TRAIN START", end="\t")
            if model_name == "MNB":
                classifier = MultinomialNB()
            elif model_name == "SVM":                
                classifier = SGDClassifier()
            classifier.fit(X_bug_train, y_bug_train)            
            # print("TRAIN FINISH")


            # 3. Prepare the tuning dataset
            train_ids, train_texts, train_labels, train_terms = load_bugs(base_path+dataset+"\\"+project+"\\"+str(i)+"_bugs_train.txt", which_field, max_length_base)
            test_ids, test_texts, test_labels, test_terms = load_bugs(base_path+dataset+"\\"+project+"\\"+str(i)+"_bugs_test.txt", which_field, max_length_base)
            split_index = len(train_labels)
            bug_ids = train_ids + test_ids
            bug_texts = train_texts + test_texts
            bug_labels = train_labels + test_labels
            for b_id in test_ids:
                total_bugs.add(dataset+"_"+project+"_"+b_id)            

            # 4. Vectorize the tuning dataset
            X_bug_train = vec.transform(train_texts).toarray()            
            y_bug_train = [1 if tmp_y=='TB' else 0 for tmp_y in train_labels]
            y_bug_train = np.array(y_bug_train, np.int32)
            X_bug_test = vec.transform(test_texts).toarray()
            y_bug_test = [1 if tmp_y=='TB' else 0 for tmp_y in test_labels]
            y_bug_test = np.array(y_bug_test, np.int32)                            
            
            train_bug_id,test_bug_id = bug_ids[:split_index], bug_ids[split_index:]                         

            # print(dataset, project, i, "TUNE START")

            # 5. Tuning the source model for target model
            classifier.partial_fit(X_bug_train, y_bug_train)
            y_predict = classifier.predict(X_bug_test)
                                            
            # 6. Evaluation
            correct = 0
            for t_i in range(len(y_bug_test)):  
                prob = y_predict[t_i]                
                y_real = y_bug_test[t_i]
                y_pred = 0
                if prob >= 0.5:
                    y_pred = 1
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
                writer.write(model_name+"_"+which_field+"_"+training_set+"_"+str(iterator)+"\t"+str(i)+"\t"+identifier+"\t"+real_type+"\t"+pred_type+"\n")
                writer.close()

            f1_pbr, f1_tbr = own_util.evaluation(correct_pb, all_pb_num, correct_tb, all_tb_num)
            print(model_name+"_"+which_field+"_"+training_set+"_"+str(iterator)+"\t"+str(i), dataset, project, correct_pb, '/', all_pb_num, correct_tb, '/', all_tb_num, round(f1_pbr,3), round(f1_tbr,3))

print(len(total_bugs))