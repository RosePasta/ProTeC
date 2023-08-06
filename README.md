# ProTeC

## Structure
```
ProTeC
│   README.md          
│   requirements.yaml   : dependency file
│
└─── classifiers
│   │   {vectorization}_{model}_classifier_{method}.py  : model variations
│   │
│   └───util
│       │    old_model.py   : model examples
│       └─── old_util.py    : common functions
│   
└─── dataset  : sample data (Bench4BL/ROO)
│   │    IRBL_dataset
│   └─── training_dataset
│
└─── results  : experimental results
    │    experiments.xlsx   :  results presented in the paper
    │    rq{#}.py           :  scripts to print the experimental results
    │    clasifier_results  :  classification results for each model variation
    └─── irbl_results       :  IRBL results for each model variation
```

## Key files and folders
### classifiers
* code for the model variations used in experiments
* ./util/old_util.py
    - Contains common functions & configuration settings
    - Modify the base_path for experiments
* {vectorization}_{model}_classifier\_{method}.py
    - Python files with model variations for experiments, where:
      1) tfidf_{dl|ml}_classifier\_{baseline|protec}.py
      2) w2v_{cnn|mtcnn}_classifier\_{baseline|protec}.py
      3) transformer_classifier_{bugreport|sourcefile|protec|bugreport_sampling}.py

### results
* Contains experimental results presented in the paper
* classifier_results
    - Holds experimental results for each model variation
* rq{1|2|3|4|5}.py
    - Scripts to print experimental results for answering each research question


## Reproduction step
1. Set up Glove Model
   1. Download and unpack from `https://nlp.stanford.edu/data/glove.6B.zip`
   2. Modify the **W2V_PATH** in 57 line of `./classifier/util/old_util.py`
2. Set up the dataset path **BASE_PATH** in 67 line of `./classifier/util/old_util.py` (Already set, so can be omitted.)
3. Install the conda Python ENV by installing the dependencies with requirements.yaml
```
conda env create -f requirements.yaml
```
4. Change the model architecture, textual features, and training/tuning dataset (refer to the comments in each variation file)
  - If the model is "ml", change the variables "model_name", "training_set", and "which_field"
  - If the model is "dl", change the variables "training_set", and "which_field"
5. Execute the Python file, for example, to run the tfidf+dl+baseline variation:
```
python ./classifier/tfidf_dl_classifier_baseline.py
```



