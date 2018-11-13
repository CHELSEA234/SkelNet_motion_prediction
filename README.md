# SkelNet_motion_prediction
This is the implementation of the paper
Xiao Guo, Jongmoo Choi. *Human motion prediction via learning local structure representations and temporal dependencies*. In AAAI2019. (The arxiv version is coming soon)

### Dependencies
Tensorflow-gpu 1.3.0
Python 3.6 or 3.5 

### Files and commands:
User can decide long-term or short-term prediction based on input argument. Specifically, three components in proposed Skel-TNet should be optimized independently. Required commands are shown below:
**SkelNet:** python3 SkelNet/src/long_pre_extraction.py --action walking
**C-RNN:** python3 C-RNN/src/long_pre_extraction.py --action walking
**Merging_network:** python3 Merging_network/src/long_pre_extraction.py --action walking --iterations 1500 
**Data** contains preprocessing the Human3.6M dataset, **spatial_model_walking** and **temporal_model_walking** represent pretained models sending to the Merging Network.
