# CSCI6962-Final-Project
- This repository contains all of the code and results produced for the final project in CSCI6962.
- This project proposes a variant of MESHGRPAHNETS' encode-process-decode architecture by replacing the Processor componenet with a Graph Convolutional Network(GCN) and changing from ReLU to Swish activation function in the MLPs in the encoder and Decoder.
- We used code from two repositories to produce these results:
    -  We first used the TensorFlow1 code from the orginal MESHGRAPHNETS paper (https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) to reproduce the results for the CylinderFlow Dataset. We edited the run_model.py file to more accurately track the training loss, and validation loss, as well as store these losses in a file. See the Original_meshgraphnets folder for more details about installation and how we reproduced the results.
    -  The second repository we used was a rewrite of MESHGRPAHNETS in PyTorch that similarly produced results for CylinderFlow (https://github.com/echowve/meshGraphNets_pytorch). We modified the model code to replace the processor with a GCN, and also the train.py file to track and calculate the training and validation loss
 
Modifications made to the PyTorch version of MESHGRAPHNETS:
  - model/model.py
     - Exchanged the MLPs in the processor, with GCN
  - videos/
     - Produced results for our version of the model
  - results/
     - Generated Rollouts to be used for rendering reuslts
  - loss/
     - Added text files for the training and validation loss, and their respective plots
  - train.py
     - Modified the code to run for less epochs (resource contraints) an to track training loss as well as validation loss, and save that output to a file
 
Installation Instructions for the PyTorch Version of MESHGRAPHNETS:
  - Initial Installation for torch 1.13 for CUDA 1.17
     - conda create --name torch113 python=3.8
     - conda install pytorch==1.13.1 torchvision==0.14.1
     - torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
     - pip install torch_geometric
     - pip install pyg_lib torch_scatter torch_sparse torch_cluster
     - torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
     - pip install networkx
  - Additional Installs
     - pip install h5py==3.6.0 matplotlib==3.4.3 numpy==1.21.1 opencv_python==4.5.4.58 Pillow==9.1.0 tqdm==4.62.3
   
To Train: 
  - Download the Dataset (See Original_meshgraphents README)
  - python parse_tfrecord.py
  - python train.py

To Test:
  - python rollout.py
  - python render_results.py



## BASELINE RESULTS
https://github.com/SohaYusuf/CSCI6962-Final-Project/assets/60673567/92f6bf36-ad23-4349-8007-6345d725aee8

Training vs Validation Loss Curves:
![train_vs_valid_loss_avg](https://github.com/SohaYusuf/CSCI6962-Final-Project/assets/60673567/eef74d43-fdff-4dfe-9841-d2e49632c179)



