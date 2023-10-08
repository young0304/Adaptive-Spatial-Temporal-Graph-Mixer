# Adaptive-Spatial-Temporal-Graph-Mixer
Official PyTorch Implementation of the paper: Adaptive Spatial-Temporal Graph-Mixer for Human Motion Prediction.

## About Datasets: ##
  ### Human3.6M:
  * Human3.6M is one of the largest 3D human pose datasets, containing 15 actions and 3.6 million video frames from 4 perspectives by 11 professional actors (6 males and 5 females). 
  * We adpot the 22-joint pose and use subjects (S1, S6, S7, S8, S9) for training, (S11) for validation, and (S5) for testing.
  ### 3DPW:
  * 3DPW is a dataset in both indoor and outdoor environments captured by mobile devices, which covers more diverse and complex human motions. 
  * We adpot the 23-joint pose.
## TRAIN: ## 
  ### Human3.6M:    
    python train_mixer_h36m.py --input_n 10 --output_n 25 
  ### 3DPW:
    python main_3dpw_3d.py --input_n 10 --output_n 25
## TEST: ##
  ### Human3.6M:
    python test_mixer_h36m.py --input_n 10 --output_n 25 
  ### 3DPW:
    python test_mixer_3dpw.py --input_n 10 --output_n 25
