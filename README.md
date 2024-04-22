# DBPNet

# Code for paper: DBPNet: Dual-Branch Parallel Network with Temporal-Frequency Fusion for Auditory Attention Detection
This paper introduces DBPNet, a novel AAD network that integrates temporal and frequency domains to enhance EEG signal decoding. 

Qinke Ni, Hongyu Zhang, Shengbing Pei, Chang Zhou, Zhao Lv, Cunhang Fan. DBPNet: Dual-Branch Parallel Network with Temporal-Frequency Fusion for Auditory Attention Detection. In Ijcai 2024.

# Preprocess
* Please download the AAD dataset for training.
* The public [KUL dataset](https://zenodo.org/records/4004271), [DTU dataset](https://zenodo.org/record/1199011#.Yx6eHKRBxPa) and MM-AAD(not yet open) are used in this paper.

# Requirements
+ Python3.11.4 \
`pip install -r requirements.txt`

# Run
* Modify the `args.*` variable in model.py to match the dataset
* Using model.py to train and test the model 
* Using multi_processing.py to train and test the model in parallel
