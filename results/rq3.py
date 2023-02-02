
def evaluation(correct_pb, all_pb, correct_tb, all_tb):
    # for PBR
    recall_pbr = 0
    precision_pbr = 0
    f1_score_pbr = 0
    if correct_pb > 0:
        tp = correct_pb
        fn = all_pb - correct_pb
        fp = all_tb - correct_tb
        tn = correct_tb
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        f1_score_pbr = 2 * ((recall * precision) /(recall + precision))
        recall_pbr = recall
        precision_pbr = precision

    # for TBR
    recall_tbr = 0
    precision_tbr = 0
    f1_score_tbr = 0    
    if correct_tb > 0:
        tp = correct_tb
        fn = all_tb - correct_tb
        fp = all_pb - correct_pb
        tn = correct_pb
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        f1_score_tbr = 2 * ((recall * precision) /(recall + precision))
        recall_tbr = recall
        precision_tbr = precision
    return f1_score_pbr, f1_score_tbr, recall_pbr, recall_tbr, precision_pbr, precision_tbr

def read_results(file_path):
    result_dict = {}
    lines = open(file_path, "r", encoding="utf8").readlines()
    for line in lines:
        tokens = line.replace("\n","").split("\t")
        learning_type = tokens[0]        
        fold_num = tokens[1]
        project = tokens[2]+"_"+tokens[3]
        bug_id = tokens[5]
        real_type = tokens[6]
        pred_type = tokens[7]
        
        if learning_type not in result_dict.keys():
            result_dict[learning_type] = {}
        if project not in result_dict[learning_type].keys():
            result_dict[learning_type][project] = {}
        if fold_num not in result_dict[learning_type][project].keys():
            result_dict[learning_type][project][fold_num] = {}
        result_dict[learning_type][project][fold_num][bug_id] = []
        result_dict[learning_type][project][fold_num][bug_id].append(real_type)
        result_dict[learning_type][project][fold_num][bug_id].append(pred_type)

    case_names = {}
    for case_name in result_dict.keys():
        tokens = case_name.split("_")
        model = tokens[0]
        text_field = tokens[1]
        training_set = tokens[2]
        iter_num = tokens[3]
        case_name = model+"_"+text_field+"_"+training_set
        if case_name not in case_names:
            case_names[case_name] = {}
        case_names[case_name][iter_num] = 0
    return case_names, result_dict


result_path = "./classifier_results/"
import os
file_list = os.listdir(result_path)
average_result_dict = {}
for file_name in file_list:
    case_results, result_dict = read_results(result_path+"/"+file_name)    
    case_results_detail = {}
    for learning_type in result_dict.keys():
        all_bug_num = 0
        all_pb_num = 0
        all_tb_num = 0
        correct_pb_num = 0
        correct_tb_num = 0
        for project in result_dict[learning_type].keys():
            for fold_num in result_dict[learning_type][project].keys():
                for bug_id in result_dict[learning_type][project][fold_num]:
                    classified_result = result_dict[learning_type][project][fold_num][bug_id]
                    real_type = classified_result[0]
                    pred_type = classified_result[1]
                    all_bug_num += 1
                    if real_type == "PB":
                        all_pb_num += 1
                        if pred_type == "PB":
                            correct_pb_num += 1
                    elif real_type == "TB":
                        all_tb_num += 1
                        if pred_type == "TB":
                            correct_tb_num += 1
        f1pbr, f1tbr, recall_pbr, recall_tbr, precision_pbr, precision_tbr \
              = evaluation(correct_pb_num, all_pb_num, correct_tb_num, all_tb_num)
        # print(file_name, learning_type, all_bug_num, correct_pb_num, "/", all_pb_num, correct_tb_num, "/", all_tb_num)
        # print(file_name, learning_type, round(f1pbr,3), round(f1tbr, 3))
        
        case_name = "_".join(learning_type.split("_")[:-1])      
        if case_name not in case_results_detail.keys():
            case_results_detail[case_name] = {}  
        case_results_detail[case_name] = [recall_pbr, precision_pbr, recall_tbr, precision_tbr, correct_pb_num, correct_tb_num]
        # bug_nums[case_name][learning_type.split("_")[-1]] = all_bug_num
    
    for case_name in case_results.keys():
        average_result_dict[case_name] = case_results_detail[case_name]


model_names = ["MNB","SVM","DNN","CNN","MTCNN","distilbert-base-uncased","distilroberta-base"]
text_features = ["SHORT","LONG","ALL"]
training_types = ["BR","SF"]

for model_name in model_names:
    for text_feature in text_features:
        key = model_name+"_"+text_feature        
        if model_name == "MTCNN":
            key = model_name+"_MT"        
        br_recall_pbr, br_precision_pbr, br_recall_tbr, br_precision_tbr, br_correct_pb_num, br_correct_tb_num = average_result_dict[key+"_BR"]
        sf_recall_pbr, sf_precision_pbr, sf_recall_tbr, sf_precision_tbr, sf_correct_pb_num, sf_correct_tb_num = average_result_dict[key+"_SF"]
        print(key, br_correct_pb_num, sf_correct_pb_num, br_precision_pbr, sf_precision_pbr, br_recall_pbr, sf_recall_pbr, \
              br_correct_tb_num, sf_correct_tb_num, br_precision_tbr, sf_precision_tbr, br_recall_tbr, sf_recall_tbr)
