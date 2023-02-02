
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


target_case_names = ["distilroberta-base_SHORT_BR", "distilroberta-base_SHORT_PROTEC"]
result_path = "./classifier_results/"
import os
file_list = os.listdir(result_path)
bug_fold_results = {}
protec_fold_results = {}
for file_name in file_list:
    case_results, result_dict = read_results(result_path+"/"+file_name)    
    case_results_detail = {}
    for learning_type in result_dict.keys():
        case_name = "_".join(learning_type.split("_")[:-1])      
        if case_name not in target_case_names:
            continue
        for project in result_dict[learning_type].keys():            
            sorted_folds = sorted(result_dict[learning_type][project].keys())
            for fold_num in sorted_folds:    
                all_bug_num = 0
                all_pb_num = 0
                all_tb_num = 0
                correct_pb_num = 0
                correct_tb_num = 0
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
                f1pbr, f1tbr = evaluation(correct_pb_num, all_pb_num, correct_tb_num, all_tb_num)
                macf1 = (f1pbr + f1tbr) / 2

                if case_name.endswith("BR") == True:
                    if project not in bug_fold_results.keys():
                        bug_fold_results[project] = []
                    bug_fold_results[project].append(macf1)
                if case_name.endswith("PROTEC") == True:
                    if project not in protec_fold_results.keys():
                        protec_fold_results[project] = []
                    protec_fold_results[project].append(macf1)

print(len(bug_fold_results), len(protec_fold_results))
for project in protec_fold_results.keys():
    end_fold_num = len(bug_fold_results[project])
    print(project, "BR", end=" ")
    for bug_result in bug_fold_results[project]:
        print(bug_result, end=" ")
    print()
    print(project, "PROTEC", end=" ")
    for protec_result in protec_fold_results[project]:
        print(protec_result, end=" ")
    print()


for i in range(10):    
    bug_win = 0
    protec_win = 0
    tie = 0
    for project in protec_fold_results.keys():    
        if i >= len(bug_fold_results[project]):
            continue
        bug_result = bug_fold_results[project][i]
        protec_result = protec_fold_results[project][i]
        if bug_result > protec_result:
            bug_win += 1
        if bug_result < protec_result:
            protec_win += 1
        if bug_result == protec_result:
            tie += 1
    print(i, bug_win, protec_win, tie)


    
