These are all the files from the Original MESHGRAPHNETS paper. The code is written in TensorFlow1.
Changes we made to the files:
  - plot_cfd.py
      - Edited the code to be able to save the plot as a gif

To reproduce the results as we did in this proect:
  - Install conda with python 3.9
  - Create conda virtual env with python 3.7 or 3.6
  - Install tensorflow-gpu 1.15 using conda
  - install dm-sonnet(not sonnet) using conda
  - Install matplotlib and other libraries as needed

Commands to create environment:
  - conda create --name tfl1 python=3.7
  - conda activate tfl1
  - conda install -c conda-forge tensorflow-gpu=1.15
  - pip install dm-sonnet==1.1

Check install versions:
  - conda activate tfl1
  - python --version
  - python -c "import tensorflow; print(tensorflow.__version__)"
  - python -c "import sonnet; print(sonnet.__version__)"

Download the Dataset:
  - mkdir -p ${DATA}
  - bash meshgraphnets/download_dataset.sh cylinder_flow ${DATA}

Train the Model:
  - python -m meshgraphnets.run_model --mode=train --model=cfd \
    --checkpoint_dir=${DATA}/chk --dataset_dir=${DATA}/cylinder_flow

Generate some Trajectory Rollouts:
  - python -m meshgraphnets.run_model --mode=eval --model=cfd \
    --checkpoint_dir=${DATA}/chk --dataset_dir=${DATA}/cylinder_flow \
    --rollout_path=${DATA}/rollout_cylinder.pkl

Plotting a Trajectory:
  - python -m meshgraphnets.plot_cfd --rollout_path=${DATA}/rollout_cylinder.pkl
