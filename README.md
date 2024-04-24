# Spiking Network Simulation

This repository contains a Jupyter Notebook that simulates a spiking neural network (SNN) using PyTorch. The network is trained and tested on the Spiking Heidelberg Digits (SHD) dataset.

## Files

- `spiking-network.ipynb`: Jupyter Notebook containing the code for setting up, training, and evaluating the spiking neural network.
- `utils.py`: Python utility file for downloading and preparing the SHD dataset.

There is also a folder labeled `training_outputs` where loss graphs and test accuracies for each hyperparameter configuration are saved. Files from a given execution must be moved out of this folder prior to performing a new execution; this will not be done automatically.

## Environment Setup

To run the notebook, you need to set up a Python environment with the necessary dependencies. This project uses PyTorch, among other libraries.

### Dependencies

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- h5py

You can install these dependencies via pip:

    pip install numpy matplotlib seaborn h5py torch torchvision

### Dataset

The SHD dataset will be automatically downloaded and prepared by the `utils.py` script when you run the notebook.

## Execution Instructions

1. Clone this repository to your local machine.
2. Ensure that you have Jupyter Notebook or JupyterLab installed. If not, you can install it using:

    pip install jupyterlab

3. Navigate to the repository directory and start JupyterLab or Notebook:

    jupyter lab
    #### or
    jupyter notebook

4. Open `spiking-network.ipynb` in Jupyter and run the cells sequentially.

## Notebook Structure

The notebook includes the following key components:

- **Data Loading and Preparation**: The notebook begins with loading the SHD dataset using utility functions defined in `utils.py`.
- **Network Configuration**: Parameters such as the number of neurons, learning rate, and other hyperparameters are set up.
- **Model Definition**: Definition of the spiking neural network model, including forward and backward passes.
- **Training Loop**: Code for training the SNN on the training dataset.
- **Evaluation**: Functions to evaluate the model's performance on the test set.
- **Visualization**: Code to visualize training loss and spike trains.

Ensure your Python environment is properly set up and all dependencies are installed before running the notebook to avoid runtime errors.
