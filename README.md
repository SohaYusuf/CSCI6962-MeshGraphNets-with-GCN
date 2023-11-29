# CSCI6962-Final-Project
- This repository contains all of the code and results produced for the final project in CSCI6962.
- This project proposes a variant of MESHGRPAHNETS' encode-process-decode architecture by replacing the Processor componenet with a Graph Convolutional Network(GCN) and changing from ReLU to Swish activation function in the MLPs in the encoder and Decoder.
- We used code from two repositories to produce these results:
    -  We first used the TensorFlow1 code from the orginal MESHGRAPHNETS paper (https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) to reproduce the results for the CylinderFlow Dataset. We edited the run_model.py file to more accurately track the training loss, and validation loss, as well as store these losses in a file. See the Original_meshgraphnets folder for more details about installation and how we reproduced the results.
    -  The second repository we used was a rewrite of MESHGRPAHNETS in PyTorch that similarly produced results for CylinderFlow (https://github.com/echowve/meshGraphNets_pytorch). We modified the model code to replace the processor with a GCN, and also the train.py file to track and calculate the training and validation loss
 
Installation Instructions for the PyTorch Version of MESHGRAPHNETS:
