import time
import argparse
import json
import os
import warnings
import nltk.translate.gleu_score as gleu

import util.old_util as own_util

# warnings.filterwarnings(action='ignore')
import nltk.translate.bleu_score as bleu

from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import T5ForConditionalGeneration
from transformers import AutoTokenizer
import numpy as np
import torch
from datasets import load_metric

import warnings
warnings.filterwarnings(action='ignore')
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import set_seed
import random
import time

from transformers import AutoTokenizer


def read_protec_dataset(data_path, data_type, max_length):
    bug_ids = []
    texts = []
    labels = []
    
    for line in open(data_path,"r", encoding="utf8").readlines():
        line = line.replace("\n",'')
        tokens = line.split("\t")
        bug_id = tokens[0]
        if data_path.find("files_train") > -1:
            bug_id = "F"+bug_id
        else:
            bug_id = "B"+bug_id
        bug_ids.append(bug_id)
        
        text = tokens[1]+" "+tokens[2]
        if data_type =="SHORT":
            text = tokens[1]
        elif data_type =="LONG":
            text = tokens[2]            
        if len(text.split(" ")) > max_length:
            text = ' '.join(text.split(" ")[:max_length])
        texts.append(text)            
            
        label = tokens[3]
        if data_path.find("files_train") > -1:
            class_id = (0 if label == "PF" else 1)
        else:
            class_id = (0 if label == "PB" else 1)
        
        labels.append(class_id)
    return bug_ids, texts, labels

class ProTeCDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

metric = load_metric("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

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

import sys
import os
from transformers import EarlyStoppingCallback

model_name = "distilroberta-base" #"distilbert-base-uncased" #,"distilroberta-base"
training_set_type = "PROTEC"
which_field = "SHORT" # "ALL", "SHORT", "LONG"

max_length_base, epoch, batch, patience, val_rate, lr = own_util.get_dl_parameters()
lr = 0.00001 # base values notified from HuggingFace


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to("cuda:0")

base_path = own_util.get_dataset_path()

executed_file_name = sys.argv[0].split("/")[-1].replace(".py","")
result_file_name = executed_file_name+"_"+str(time.time())
writer = open("./results/"+result_file_name+".txt","w",encoding="utf8")
writer.close()

iterator = 0

for iterator in range(5):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)        
    total_bugs = set() 
    datasets = os.listdir(base_path)
    for dataset in datasets:
        projects = os.listdir(base_path+dataset)
        for project in projects:
            for i in range(0, 10):
                    
                
                if os.path.exists(base_path+dataset+"/"+project+"/"+str(i)+"_bugs_train.txt") == False:
                    print(dataset, project, "no file",base_path+dataset+"/"+project+"/"+str(i)+"_bugs_train.txt")
                    continue

                # 0. Prepare dataset    
                train_path = base_path+dataset+"/"+project+"/"+str(i)+"_files_train.txt"   

                train_ids, train_texts, train_labels = read_protec_dataset(train_path,which_field, max_length_base)                    

                from sklearn.model_selection import train_test_split
                train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=val_rate)

                train_encodings = tokenizer(train_texts, truncation=True, padding=True)
                val_encodings = tokenizer(val_texts, truncation=True, padding=True)
                                    
                train_dataset = ProTeCDataset(train_encodings, train_labels)
                val_dataset = ProTeCDataset(val_encodings, val_labels)            
                
                from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
                model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
                training_args = TrainingArguments(
                    output_dir="./outputs_"+model_name,
                    logging_strategy="no",
                    evaluation_strategy = "steps",
                    metric_for_best_model = 'f1',
                    load_best_model_at_end = True,
                    eval_steps = 100,
                    learning_rate=lr,
                    per_device_train_batch_size=batch,
                    per_device_eval_batch_size=batch,
                    num_train_epochs=epoch,
                    weight_decay=0.01,                
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics,
                    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)],
                    
                )

                trainer.train()
                trainer.model.eval()
                trainer.evaluate()       

                
                train_path = base_path+dataset+"/"+project+"/"+str(i)+"_bugs_train.txt"   
                test_path = base_path+dataset+"/"+project+"/"+str(i)+"_bugs_test.txt"

                train_ids, train_texts, train_labels = read_protec_dataset(train_path,which_field, max_length_base)
                test_ids, test_texts, test_labels = read_protec_dataset(test_path, which_field, max_length_base)

                from sklearn.model_selection import train_test_split
                train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=val_rate)

                train_encodings = tokenizer(train_texts, truncation=True, padding=True)
                val_encodings = tokenizer(val_texts, truncation=True, padding=True)
                test_encodings = tokenizer(test_texts, truncation=True, padding=True)
                
                train_dataset = ProTeCDataset(train_encodings, train_labels)
                val_dataset = ProTeCDataset(val_encodings, val_labels)      

                
                training_args = TrainingArguments(
                    output_dir="./results_"+model_name,
                    logging_strategy="no",
                    evaluation_strategy = "steps",
                    metric_for_best_model = 'f1',
                    load_best_model_at_end = True,
                    eval_steps = 100,
                    learning_rate=lr*0.1,
                    per_device_train_batch_size=batch,
                    per_device_eval_batch_size=batch,
                    num_train_epochs=epoch,
                    weight_decay=0.01,                
                )

                trainer = Trainer(
                    model=trainer.model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics,
                    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)],
                    
                )

                trainer.train()
                trainer.model.eval()
                trainer.evaluate()       


                correct_bugs = 0
                correct_pbs = 0
                correct_tbs = 0
                all_bugs = 0
                for t_i, test_instance in enumerate(test_texts):
                    bug_id = test_ids[t_i]
                    real_class_id = test_labels[t_i]
                    inputs = tokenizer(test_instance, truncation=True, padding=True, return_tensors="pt")
                    inputs.to("cuda:0")
                    with torch.no_grad():
                        logits = trainer.model(**inputs).logits
                    predicted_class_id = logits.argmax().item()

                    writer = open("./results/"+result_file_name+".txt","a",encoding="utf8")
                    identifier = dataset+"\t"+project+"\t"+"\t"+bug_id
                    real_type = "PB"
                    if real_class_id == 1:
                        real_type = "TB"
                    pred_type = "PB"
                    if predicted_class_id == 1:
                        pred_type = "TB"
                    writer.write(model_name+"_"+which_field+"_PROTEC_"+str(iterator)+"\t"+str(i)+"\t"+identifier+"\t"+real_type+"\t"+pred_type+"\n")
                    writer.close()
                    
                    if real_class_id == predicted_class_id:
                        correct_bugs += 1
                        if real_class_id == 0:
                            correct_pbs += 1
                        else: 
                            correct_tbs += 1
                    all_bugs += 1
                
                # f1_pbr, f1_tbr = evaluation(correct_pb, all_pb_num, correct_tb, all_tb_num)
                print(model_name+"_"+which_field+"_PROTEC_"+str(iterator)+"\t"+str(i), dataset, project, )#correct_pb, '/', all_pb_num, correct_tb, '/', all_tb_num, round(f1_pbr,3), round(f1_tbr,3))

print(len(total_bugs))