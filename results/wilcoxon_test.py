# Open classification result files
def read_data(file_path):
    result_dict = {}
    lines = open(file_path, "r", encoding="utf8").readlines()
    for line in lines:
        line = line.replace("\n","")
        tokens = line.split("\t")
        identifier = tokens[3]+"_"+tokens[5]
        real = tokens[6]
        pred = tokens[7]
        result = 1
        if real != pred:
            result = 0 
        result_dict[identifier] = result
    return result_dict

result_path = "./results/classifier_results/"
file_name = ["transformer_classifier_","_distilroberta-base.txt"]
br_path = result_path+file_name[0]+"bugreport"+file_name[1]
sf_path = result_path+file_name[0]+"sourcefile"+file_name[1]
protec_path = result_path+file_name[0]+"protec"+file_name[1]

br_results = read_data(br_path)
sf_results = read_data(sf_path)
protec_results = read_data(protec_path)
print(len(br_results), len(sf_results), len(protec_results))

br_result_list = []
sf_result_list = []
protec_result_list = []

for identifier in protec_results.keys():
    protec_result = protec_results[identifier]
    br_result = br_results[identifier]
    sf_result = sf_results[identifier]

    br_result_list.append(br_result)
    sf_result_list.append(sf_result)
    protec_result_list.append(protec_result)

print(len(br_result_list),len(sf_result_list),len(protec_result_list))

# Analysis
import scipy
print(scipy.stats.wilcoxon(br_result_list, protec_result_list))
print(scipy.stats.wilcoxon(sf_result_list, protec_result_list))
