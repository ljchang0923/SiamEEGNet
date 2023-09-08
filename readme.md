# SiamEEGNet: Siamese Neural Network-Based EEG Decoding for Drowsiness Detection
This repository is the official implementation of SiamEEGNet: Siamese Neural Network-Based EEG Decoding for Drowsiness Detection

## Objective
We proposed SiamEEGNet, a Siamese neural network structure dedicated to processing EEG relative change information, to enhance the performance of EEG-based drowsiness detection.

## Datase
Open sustained-attention driving dataset
Z. Cao, C.H. Chuang, J.K. King, and C.T. Lin, “Multi-channel eeg recordings during a sustained-attention driving task,” Scientific data, vol. 6, no. 1, pp. 1–8, 2019.
- Using the open source sustained-attention driving task from scientific data
- The main purpose in this dataset is trying to use 3 sec EEG signal prior to deviation onset to predict the Drowsiness Index(DI)
![](https://i.imgur.com/fFUG7JC.png)

## Pre-processing
Pre-processing the data using Matlab, and the procedures including

Re-reference
* Band pass filter (2~30 Hz)
* Down sampling (500 to 250 Hz)
* ASR to remove artifacts (standard deviation=10)
* Epoching (3 sec EEG signal prior to deviation onset)
* convert reaction time to Drowsiness Index

## Method
* Multi-window input
    * Single branch EEG decoding model with multi-window processing

* SiamEEGNet
    * Double branch EEG decoding model with multi-window processing
    * Predict the changes in drowsiness level between baseline and any trials

## Training scenario
* Within-subject
* Cross-subject

## File
1. train_within_subject.py: train SiamEEGNet with the within-subject training scheme
2. train_cross_subject.py: train SiamEEGNet with the cross-subject training scheme
3. Inference.py: Inference on the trained models
4. molel: Including SiamEEGNet and different EEG decoding backbone model
5. utils: Including useful function such as loading dataset, gettign dataloader, computing evaluation metrics, et al.
