# Siamese CNN-based EEG-decoding Model for Extracting Relative Pattern to Drowsiness Estimation

## Objective
- EEG-based drowsiness estimation
- Predict the relative change of drowsiness level between 2 inputs by capturing the patterns of relative change in EEG
- Propose a Siamese CNN-based EEG decoding model to achieve these goals

## Dataset
- Using the open source sustained-attention driving task from scientific data
- The main purpose in this dataset is trying to use 3 sec EEG signal prior to deviation onset to predict the Drowsiness Index(DI)
![](https://i.imgur.com/fFUG7JC.png)

## Pre-processing
Pre-processing the data using Matlab, and the procedures including

Re-reference
Band pass filter (2~30)
down sampling (500 to 250 Hz)
Epoching (3 sec EEG signal prior to deviation onset)
convert reaction time to Drowsiness Index

## Method
* Backbone Network
    * Directly using oringal SCCNet(regression version) to predict the Drowsiness Index
    * SCCNet to extract spatial and temporal feature. And a regression head followed by the feature extractor to predict DI

* Multi-window input
    * To stablize the process of EEG decoding and reduce the impact of outliers
    * Combine 9 trials prior to current trial as 1 input
    * After feature extraction, concatenate 10 latent features and applying averaging pooling to obtain average power-like features

* Siamese SCCNet
    * Apply Siamese architecture to capture the latent feature from 2 inputs
    * Need to form a pair dataset
    * Predict the delta DI between 2 trials
    * In testing phase, predict the delta DI between baseline and any other trials (DI that remove the baseline) 

## training scheme
* Within-subject
* Cross-subject

## File
1. DataLoader.py: Customized Pytorch dataloader for forming pair dataset
2. models.py: Model architecture including backbone network and Siamese architecture
3. utils.py: Some helper fucntion including read config file, loading dataset, and plotting figure
4. train.py: Model training, validation, and testing functions
5. main.py: To train and inference
