# LLMTB
This is the repository for LLMTB. A BERT model for antimicrobial resistance prediction in *Mycobacterium tuberculosis*.

## Citation
If you use this repository please cite:
```
\@article{testagrose2025leveraging,
  title={Leveraging large language models to predict antibiotic resistance in Mycobacterium tuberculosis},
  author={Testagrose, Conrad and Pandey, Sakshi and Serajian, Mohammadali and Marini, Simone and Prosperi, Mattia and Boucher, Christina},
  journal={Bioinformatics},
  volume={41},
  number={Supplement\_1},
  pages={i40--i48},
  year={2025},
  publisher={Oxford University Press}
}
```

## Example Repository Structure
```
LLMTB/
├── Data/                             # Sample MTB sequences + metadata
├── Scripts/
│   ├── train_model.py                # Entry point for training
│   ├── Evaluater.py                  # Evaluation utilities
│   ├── src/
│   │   ├── Utils/
│   │   │   ├── Trainer.py            # Training logic
│   │   │   ├── Model_Manager.py      # Model loading/saving
│   │   │   └── ...
│   │   └── Models/
│   │       └── BERT/
│   │           └── BERT.py
│   └── Config/
│       ├── Gene_Configs/
│       ├── Model_Configs/
│       └── train_config.json
└── Trained_Models/                   # Output directory

```

## Running Scripts
### Setup
```
git clone https://github.com/<your-user>/LLMTB.git
cd LLMTB
```
### Install Dependencies
```
mamba create -n llmtb_env python=3.11.13
mamba activate llmtb_env
pip install -r requirements.txt
```
### Configuring a Training Run
Modify ```train_config.json```. Example:
```
{
    "sequence_dir": "/LLMTB/Data/IR_Variable_Isolates_Train",
    "gene_file": "/LLMTB/Scripts/Config/Gene_Configs/genes.json",
    "target_file": "/LLMTB/Data/cryptic_targets_all.json",
    "num_epochs": 1,
    "batch_size": 2,
    "use_holdout": false,
    "use_gene_file": true,
    "model_config": "/LLMTB/Scripts/Config/Model_Configs/Base_BERT/base_config_binary.json",
    "antibiotic": "INH",
    "oversample": true,
    "adversarial_training": true,
    "distributed": false
    "save_path": "/LLMTB/Trained_Models"
}
```
Important Fields:

* ```oversample```: use oversampling
* ```distributed```: True = muli-gpu training, False = single GPU training
* ```use_holdout```: Use a holdout test set or not
* ```antibiotic```: Valid choices ["INH", "RIF", "RFB", "EMB", "ETH", "LEV", "MXF", "KAN", "AMI", "DLM", "CFZ", "LZD", "BDQ"]
* ```save_path```: Location to save output

## Training
### Single GPU/CPU/MPS: 
Run the code from the Scripts folder using:
```
python train_model.py
```
### Mult-GPU (CUDA)
Update the flag in the config file and run using:
```
torchrun --nproc-per-node <NUM GPUS PER NODE> train_model.py
```

## Outputs
Training will output log files in the directory designated for the saving of the models. 
```
<save_path>/<antibiotic>/
│── tokenizer.json
│── best_model_fold_<N>.pth
│── train_seq_ids.json
│── val_seq_ids.json
└── log_train_fold_1.txt
```
Example Log:
```
Namespace(sequence_dir='/Users/conrad/LLMTB-main/Data/IR_Variable_Isolates_Train', num_epochs=1, batch_size=2, model_config='/Users/conrad/Downloads/LLMTB-main/Scripts/Config/Model_Configs/Base_BERT/base_config_binary.json', antibiotic='AMI', save_path='/Users/conrad/Downloads/LLMTB-main/Trained_Models', use_holdout=False, use_gene_file=True, gene_file='/Users/conrad/Downloads/LLMTB-main/Scripts/Config/Gene_Configs/genes.json', oversample=True, adversarial_training=True, distributed=False, target_file='/Users/conrad/Downloads/LLMTB-main/Data/cryptic_targets_all.json')
Training on Rank 0 for 1 epochs...
alpha: 0.5
gamma: 1.0
Ordering: ['RIF', 'INH', 'EMB', 'AMI', 'KAN', 'RFB', 'LEV', 'MXF', 'ETH', 'LZD', 'CFZ', 'DLM', 'BDQ']
Training for 1 epochs...
--------------------
Epoch 1, Learning Rate: 2e-05
Train Loss: 0.3985236747524677
Average Validation Loss: 0.18106456371870908
Validation Accuracy: 0.1905
Validation F1 Score: 0.1053
Precision: 0.0556 Recall: 1.0000
Threshold: 0.28
AUC: 0.2000
Best F1 Score: 0.0000 at Epoch: 0
Hamming Loss: 0.8095
Jaccard Score: 0.0556
Confusion Matrix:
[[ 3 17]
 [ 0  1]]
Classification Report:
              precision    recall  f1-score   support

           0     1.0000    0.1500    0.2609        20
           1     0.0556    1.0000    0.1053         1

    accuracy                         0.1905        21
   macro avg     0.5278    0.5750    0.1831        21
weighted avg     0.9550    0.1905    0.2535        21

Best Threshold: 0.28

No statistically significant genes found (z-score > 2).
Model and threshold saved to /Users/conrad/LLMTB-main/Trained_Models/AMI/best_model_fold_1.pth
Wrong Resistant Isolates: []
Training Finished
```

## Evaluation
Code is provided to load models and test. Adjust the Evaluation Config:

```
{
    "sequence_dir": "/LLMTB/Data/IR_Variable_Isolates_Eval",
    "gene_file": "/LLMTB/Scripts/Config/Gene_Configs/genes.json",
    "target_file": "/LLMTB/Data/cryptic_targets_all.json",
    "use_holdout": false,
    "use_gene_file": true,
    "model_config": "/LLMTB/Scripts/Config/Model_Configs/Base_BERT/base_config_binary.json",
    "antibiotic": "AMI",
    "save_path": "/LLMTB/Trained_Models",
    "fold": 0
}
```
Run evaluation using:
```
python evaluate_model.py
```
Output is saved in location where the model was previously saved. 

### Running on Custom Dataset
We are currently adapting the code to work on datasets outside of the CRyPTIC dataset used in the study. 
This current version of the code has not been validated on datasets outside of CRyPTIC but should work in cases where:
1. Target File has same structure of isolate number and antibiotic.
```
{
    "ERR4810489": {
        "AMI": 0.0,
        "BDQ": NaN,
        "CFZ": 0.0,
        "DLM": 0.0,
        "EMB": 0.0,
        "ETH": 0.0,
        "INH": 0.0,
        "KAN": 0.0,
        "LEV": 0.0,
        "LZD": 0.0,
        "MXF": 0.0,
        "RIF": 0.0,
        "RFB": 0.0
    },
    ...
}  
```
2. Fasta files follow a similar naming scheme:
```
IsolateNum_SuffixText.fasta
```
3. Fasta file headers are setup as such:
```
>unique_169|GJGLMLCD_03391||tuberculosis|ERR2510875|FROM:52-731|NODE_45_length_35900_cov_21.074292|IR:0-52|BEFORE|STRAND:+
...
>unique_169|GJGLMLCD_03391||tuberculosis H37Rv|CDS|Rv1765c|1997418-1998515|-|Rv1765c|downstream:0|upstream:0
...
>unique_169|GJGLMLCD_03391||tuberculosis|ERR2510875|FROM:52-731|NODE_45_length_35900_cov_21.074292|IR:731-1081|AFTER|STRAND:+
...
```
BEFORE or AFTER are contained in the intergenic headers while the gene name is the 3rd to last in the header for the gene seqeunce.