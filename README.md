# SkelNet_motion_prediction
This repository contains our work in [AAAI2019](https://aaai.org/Conferences/AAAI-19/wp-content/uploads/2018/11/AAAI-19_Accepted_Papers.pdf). We proposed novel architectures for the Human motion prediction from motion capture data. First, Skeleton Network (SkelNet) learns different local moving pattern from body components, and employs such locality for predicting the future human motion. Then, we built up Skeleton Temporal Network (Skel-TNet) that consists SkelNet and RNN, which have  advantages in learning spatial and temporal dependencies for predicting human motion, respectively. Our methods achieve state-of-the-art results on the Human3.6M dataset and the CMU motion capture dataset. You can also check our paper for a details.

<img src="https://github.com/CHELSEA234/SkelNet_motion_prediction/blob/master/Img/Figure_1.png" width="400" /> <img src="https://github.com/CHELSEA234/SkelNet_motion_prediction/blob/master/Img/Figure_2.png" width="350" />

Xiao Guo, Jongmoo Choi. *Human Motion Prediction via Learning Local Structure Representations and Temporal Dependencies*. In AAAI2019. [Paper](https://arxiv.org/abs/1902.07367).

### Dependencies
* Tensorflow-gpu 1.3.0 
* Python 3.6 or 3.5 

### Files and commands:
User can decide long-term or short-term prediction based on input argument. Specifically, three components in proposed Skel-TNet should be optimized independently. 

Supposed target action is walking, required commands are shown below: 
* **SkelNet:** 

`python3 SkelNet/src/long_pre_extraction.py --action walking`
* **C-RNN:** 

`python3 C-RNN/src/long_pre_extraction.py --action walking`
* **Merging_network:** 

`python3 Merging_network/src/long_pre_extraction.py --action walking --iterations 1500 `
* **Data** contains preprocessing the Human3.6M dataset
* **spatial_model_walking** and **temporal_model_walking** represent pretained models sending to the Merging Network.

### Citation:

if you find our work useful in your research, please consider citing (since the official proceeding of AAAI19 has not been published yet):
@inproceedings{guo2019human,
    title={Human motion prediction via learning local structure representations and temporal dependencies},
    author={Guo, Xiao and Choi, Jongmoo},
    booktitle={AAAI},
    year={2019}
}
```
