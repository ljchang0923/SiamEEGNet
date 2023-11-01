# SiamEEGNet
This repository is the official implementation of 'SiamEEGNet: Siamese Neural Network-Based EEG Decoding for Drowsiness Detection'.

## Requirements
### Step 1.
To install requirements:
```
conda env create -f /path/to/SiamEEGNet_env.yml
conda activate SiamEEGNet_env
```
### Step 2.
- Create a new empty folder 'data' in this folder. Download processed dataset and put them to the folder 'data'.

## Dataset
- Lane-keeping driving dataset task: https://figshare.com/articles/dataset/Multi-channel_EEG_recordings_during_a_sustained-attention_driving_task/6427334
- Processed dataset download: https://drive.google.com/drive/folders/1_b4Fz9B7xE18z0IBJ3Dcn_mXzRcEaJ-7?usp=sharing

## Training and inference

Train SiamEEGNet with within-subject training scheme.
```
python train_within_subject.py
```
Train SiamEEGNet with cross-subject training scheme.
```
python cross_within_subject.py
```
Inference using existing models.
```
python inference.py --model_path /trained_models
```
All default hyperparameters are already set in files.

## References
```
@article{lawhern2018eegnet,
  title={EEGNet: a compact convolutional neural network for EEG-based brain--computer interfaces},
  author={Lawhern, Vernon J and Solon, Amelia J and Waytowich, Nicholas R and Gordon, Stephen M and Hung, Chou P and Lance, Brent J},
  journal={Journal of neural engineering},
  volume={15},
  number={5},
  pages={056013},
  year={2018},
  publisher={iOP Publishing}
}
```
```
@inproceedings{wei2019spatial,
  title={Spatial component-wise convolutional network (SCCNet) for motor-imagery EEG classification},
  author={Wei, Chun-Shu and Koike-Akino, Toshiaki and Wang, Ye},
  booktitle={2019 9th International IEEE/EMBS Conference on Neural Engineering (NER)},
  pages={328--331},
  year={2019},
  organization={IEEE}
}
```

```
@article{schirrmeister2017deep,
  title={Deep learning with convolutional neural networks for EEG decoding and visualization},
  author={Schirrmeister, Robin Tibor and Springenberg, Jost Tobias and Fiederer, Lukas Dominique Josef and Glasstetter, Martin and Eggensperger, Katharina and Tangermann, Michael and Hutter, Frank and Burgard, Wolfram and Ball, Tonio},
  journal={Human brain mapping},
  volume={38},
  number={11},
  pages={5391--5420},
  year={2017},
  publisher={Wiley Online Library}
}
```

```
@article{gao2019eeg,
  title={EEG-based spatio--temporal convolutional neural network for driver fatigue evaluation},
  author={Gao, Zhongke and Wang, Xinmin and Yang, Yuxuan and Mu, Chaoxu and Cai, Qing and Dang, Weidong and Zuo, Siyang},
  journal={IEEE transactions on neural networks and learning systems},
  volume={30},
  number={9},
  pages={2755--2763},
  year={2019},
  publisher={IEEE}
}
```

```
@article{cui2022eeg,
  title={EEG-based cross-subject driver drowsiness recognition with an interpretable convolutional neural network},
  author={Cui, Jian and Lan, Zirui and Sourina, Olga and M{\"u}ller-Wittig, Wolfgang},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022},
  publisher={IEEE}
}
```
```
@inproceedings{ingolfsson2020eeg,
  title={EEG-TCNet: An accurate temporal convolutional network for embedded motor-imagery brain--machine interfaces},
  author={Ingolfsson, Thorir Mar and Hersche, Michael and Wang, Xiaying and Kobayashi, Nobuaki and Cavigelli, Lukas and Benini, Luca},
  booktitle={2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
  pages={2958--2965},
  year={2020},
  organization={IEEE}
}
```

```
@article{altuwaijri2022multi,
  title={A multi-branch convolutional neural network with squeeze-and-excitation attention blocks for eeg-based motor imagery signals classification},
  author={Altuwaijri, Ghadir Ali and Muhammad, Ghulam and Altaheri, Hamdi and Alsulaiman, Mansour},
  journal={Diagnostics},
  volume={12},
  number={4},
  pages={995},
  year={2022},
  publisher={MDPI}
}
```

```
@article{cao2019multi,
  title={Multi-channel EEG recordings during a sustained-attention driving task},
  author={Cao, Zehong and Chuang, Chun-Hsiang and King, Jung-Kai and Lin, Chin-Teng},
  journal={Scientific data},
  volume={6},
  number={1},
  pages={1--8},
  year={2019},
  publisher={Nature Publishing Group}
}
```
