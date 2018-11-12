# SkelNet_motion_prediction

This is the implementation of the Human motion prediction via learning local structure representations and temporal dependencies

Dependencies:
Tensorflow 1.3.0
Python 3.6 or 3.5 


Commands:
SkelNet: python3 SkelNet/src/long_pre_extraction.py --action walking

C-RNN: python3 C-RNN/src/long_pre_extraction.py --action walking

Merging_network: python3 Merging_network/src/long_pre_extraction.py --action walking --iterations 1500 