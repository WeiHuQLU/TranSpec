# TranSpec

## Installation environment

The required operating system, GPU specifications, software packages, their versions, and download instructions can be found in the requirements.txt file.Setting up the environment takes approximately 10 minutes.

## Training a model

This section describes how to preprocess data to make it accessible to the model, train the model, run inference, and compute accuracy.

### Data preprocessing

We take the first 5000 molecules of the QM9S dataset as an example. First, we prepare two data files in the examples/data folder of figshare(https://figshare.com/s/1581c92625dc3803f983), which are the IR spectra data file and the corresponding SMILES file. Find the path of the folder where qm9sdata_processing-examples.py is currently located in Anaconda Prompt, put the two data files and qm9sdata_processing-examples.py into the same path, and then execute the preprocessing operation with the following code.(If you see the error ModuleNotFoundError: No module named 'util', please copy util.py from the src folder into your current working directory.)


```python
python qm9sdata_processing-examples.py
```

### Training

Through preprocessing, we obtained the infrared spectral data that went into the model under the current path. Training set, validation set and test set files were generated in the examples/dataset folder of figshare. Here, we set the epoch of the training process to 5 (actually the epoch is 200) and use the following code to perform the operation of training the model in the current path.Running the example training code takes around 15 minutes.(If you see the error ModuleNotFoundError: No module named 'model', please copy model.py from the src folder into your current working directory.)


```python
python train-examples.py
```

### Inference

Models will generate optimal_qm9s_ircnn_examples.pt file in the current path after the model passes the training, then we will load this model parameter file into the evaluation code with a maximum limit of 20 candidate molecules and only the first 10 molecules will be tested. Finally, the labeled SMILES file and the SMILES file of the candidate molecules will be generated, and finally the accuracy will be calculated to produce the results. Execute the following code for evaluation.Running the evaluation code takes about 5 minutes.


```python
python evaluate-examples.py
```
