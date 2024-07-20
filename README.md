# TranSpec

## Installation environment

The required packages and their versions can be found in requirements.txt

## Training a model

This section describes how to preprocess data to make it accessible to the model, train the model, run inference, and compute accuracy.

### Data preprocessing

We take the first 5000 molecules of the QM9S dataset as an example. First, we prepare two data files in the examples/data folder of figshare(https://doi.org/10.6084/m9.figshare.26333170), which are the IR spectra data file and the corresponding SMILES file. Find the path of the folder where qm9sdata_processing-examples.py is currently located in Anaconda Prompt, put the two data files and qm9sdata_processing-examples.py into the same path, and then execute the preprocessing operation with the following code.


```python
python qm9sdata_processing-examples.py
```

### Training

Through preprocessing, we obtained the infrared spectral data that went into the model under the current path. Training set, validation set and test set files were generated in the examples/dataset folder of figshare. Here, we set the epoch of the training process to 5 (actually the epoch is 200) and use the following code to perform the operation of training the model in the current path.


```python
python train-examples.py
```

### Inference

Models will generate optimal_qm9s_ircnn_examples.pt file in the current path after the model passes the training, then we will load this model parameter file into the evaluation code with a maximum limit of 20 candidate molecules and only the first 10 molecules will be tested. Finally, the labeled SMILES file and the SMILES file of the candidate molecules will be generated, and finally the accuracy will be calculated to produce the results. Execute the following code for evaluation.


```python
python evaluate-examples.py
```
