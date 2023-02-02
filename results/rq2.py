
def get_perf(correctly_classified, incorrectly_classified, base_irbl_perf, ss_irbl_perf):    
    bug_num = 0
    top1, top5, top10, mrr, map = 0, 0, 0, 0, 0

    for identifier in correctly_classified:
        ss_perf = ss_irbl_perf[identifier]
        rr, ap = ss_perf["rr"], ss_perf["ap"]
        top_rank = 1/rr
        if top_rank <= 1.1:
            top1 += 1
        if top_rank <= 5.1:
            top5 += 1
        if top_rank <= 10.1:
            top10 += 1
        mrr += rr
        map += ap
        bug_num += 1
        
    bug_num += len(incorrectly_classified)

    top1 = top1 / bug_num
    top5 = top5 / bug_num
    top10 = top10 / bug_num
    mrr = mrr / bug_num
    map = map / bug_num
    return top1, top5, top10, mrr, map, bug_num

def get_perf_per_br(correctly_classified, incorrectly_classified, base_irbl_perf, ss_irbl_perf):
    rr_pos, ap_pos, rr10_pos, rr10_neg = 0, 0, 0, 0    
    for identifier in correctly_classified:
        base_perf = base_irbl_perf[identifier]
        base_rr, base_ap = base_perf["rr"], base_perf["ap"]
        ss_perf = ss_irbl_perf[identifier]
        ss_rr, ss_ap = ss_perf["rr"], ss_perf["ap"]
        if base_rr < ss_rr:
            rr_pos += 1
        if base_ap < ss_ap:
            ap_pos += 1
        
        if base_rr < 0.1 and ss_rr >= 0.1:
            rr10_pos += 1
            
    for identifier in incorrectly_classified:
        base_perf = base_irbl_perf[identifier]
        base_rr, base_ap = base_perf["rr"], base_perf["ap"]
        ss_perf = ss_irbl_perf[identifier]
        ss_rr, ss_ap = ss_perf["rr"], ss_perf["ap"]
        if base_rr >= 0.1:
            rr10_neg += 1
    return rr_pos, ap_pos, rr10_pos, rr10_neg

def load_irbl_results():
    irbl_metric_list = ["t1","t5","t10","rr","ap"]
    origin_perf = {}
    for metric_name in irbl_metric_list:
        origin_perf[metric_name] = 0
        
    datasets = ["bench4bl","bugl","denchmark","tse"]    
    base_irbl_perf = {}
    ss_irbl_perf = {}
    ss_correctly_classified = []
    for dataset in datasets:
        lines = open("./irbl_results/irbl_results_"+dataset+".txt","r", encoding="utf8").readlines()
        for line in lines:
            line = line.replace("\n","")
            tokens = line.split("\t")
            project = tokens[1]
            bug_id = tokens[3]
            if bug_id.startswith("B") == False:
                bug_id = "B"+bug_id
            identifier = str(dataset+"_"+project+"_"+bug_id).upper()
            
            loc_type = tokens[4].upper()
            rr, ap, top_rank = float(tokens[5]), float(tokens[6]), 1.0 / float(tokens[5])
            

            if loc_type == "ALL":
                base_irbl_perf[identifier] = {}
                for irbl_metric in irbl_metric_list:
                    base_irbl_perf[identifier][irbl_metric] = 0
                base_irbl_perf[identifier]["rr"] = rr
                base_irbl_perf[identifier]["ap"] = ap
                
                origin_perf["rr"] += rr
                origin_perf["ap"] += ap

                if top_rank <= 1.1:
                    base_irbl_perf[identifier]["t1"] = rr
                    origin_perf["t1"] += 1
                if top_rank <= 5.1:                    
                    base_irbl_perf[identifier]["t5"] = rr
                    origin_perf["t5"] += 1
                if top_rank <= 10.1:                    
                    base_irbl_perf[identifier]["t10"] = rr
                    origin_perf["t10"] += 1
            else:
                ss_irbl_perf[identifier] = {}
                for irbl_metric in irbl_metric_list:
                    ss_irbl_perf[identifier][irbl_metric] = 0
                ss_irbl_perf[identifier]["rr"] = rr
                ss_irbl_perf[identifier]["ap"] = ap
                                
                if top_rank <= 1.1:
                    ss_irbl_perf[identifier]["t1"] = 1
                if top_rank <= 5.1:
                    ss_irbl_perf[identifier]["t5"] = 1
                if top_rank <= 10.1:
                    ss_irbl_perf[identifier]["t10"] = 1
                ss_correctly_classified.append(identifier)                    
    
    print("AllFiles", end=" ")
    for irbl_metric in irbl_metric_list:
        print(origin_perf[irbl_metric]/len(ss_irbl_perf), end=" ")    
    print()

    print("SuitableFiles", end=" ")
    top1, top5, top10, mrr, map, bug_num = get_perf(ss_correctly_classified, [], base_irbl_perf, ss_irbl_perf)    
    rr_pos, ap_pos, rr10_pos, rr10_neg = get_perf_per_br(ss_correctly_classified, [], base_irbl_perf, ss_irbl_perf)    
    print(top1, top5, top10, mrr, map, bug_num, len(ss_correctly_classified), \
                    rr_pos, ap_pos, rr10_pos, 0, rr10_neg)
        
    return base_irbl_perf, ss_irbl_perf

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

base_irbl_perf, ss_irbl_perf = load_irbl_results()

result_path = "./classifier_results/"
import os
file_list = os.listdir(result_path)
case_correctly_classified = {}
case_incorrectly_classified = {}
for file_name in file_list:
    if file_name.lower().find("_protec") == -1:
        continue
    case_results, result_dict = read_results(result_path+"/"+file_name)        
    for learning_type in result_dict.keys():
        case_name = "_".join(learning_type.split("_")[:-1])

        if case_name not in case_correctly_classified.keys():
            case_correctly_classified[case_name] = []
            case_incorrectly_classified[case_name] = []        
        
        for project in result_dict[learning_type].keys():
            for fold_num in result_dict[learning_type][project].keys():
                for bug_id in result_dict[learning_type][project][fold_num]:
                    identifier = str(project+"_"+bug_id).upper()
                    if identifier not in base_irbl_perf.keys():
                        print(file_name, learning_type, identifier)
                    classified_result = result_dict[learning_type][project][fold_num][bug_id]
                    real_type = classified_result[0]
                    pred_type = classified_result[1]
                    if real_type == pred_type:
                        case_correctly_classified[case_name].append(identifier)
                    else:
                        case_incorrectly_classified[case_name].append(identifier)


model_names = ["MNB","SVM","DNN","CNN","MTCNN","distilbert-base-uncased","distilroberta-base"]
text_features = ["SHORT","LONG","ALL"]
training_types = ["PROTEC"]

for model_name in model_names:
    for text_feature in text_features:
        key = model_name+"_"+text_feature        
        if model_name == "MTCNN":
            key = model_name+"_MT"
        print(key, end=" ")
        for training_type in training_types:
            correctly_classified = case_correctly_classified[key+"_"+training_type]
            incorrectly_classified = case_incorrectly_classified[key+"_"+training_type]
            
            top1, top5, top10, mrr, map, bug_num = get_perf(correctly_classified, incorrectly_classified, base_irbl_perf, ss_irbl_perf)    
            rr_pos, ap_pos, rr10_pos, rr10_neg = get_perf_per_br(correctly_classified, incorrectly_classified, base_irbl_perf, ss_irbl_perf)    
            print(top1, top5, top10, mrr, map, bug_num, len(correctly_classified), \
                            rr_pos, ap_pos, rr10_pos, len(incorrectly_classified), rr10_neg, end = " ")
        print()
