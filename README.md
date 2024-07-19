# TranSpec

## Installation environment

The required packages and their versions can be found in requirements.txt

## Training a model

This section describes how to preprocess data to make it accessible to the model, train the model, run inference, and compute accuracy.

### Data preprocessing

We took the first 5000 molecules of the QM9S dataset as an example for interception. First, we preprocessed 2 data files in the examples/data folder. In Anaconda Prompt, find the path to the folder where qm9sdata_processing-examples.py is currently located, and then use the following code to perform the preprocessing operation.


```python
python qm9sdata_processing-examples.py
```

### Training

Through preprocessing, we obtained the infrared spectral data that went into the model. And the training set, validation set, and test set files were generated in the examples/dataset folder. Here, we set the epoch to 5 during the training process, (in reality epoch is 200) and use the following code to perform the operation of training the model under the current path.


```python
python train-examples.py
```

### Inference

The model generates an optimal_qm9s_ircnn_examples.pt file in the examples folder after it passes the training, we then load this model parameter file into the evaluation code with a maximum limit of 20 candidates and only the first 10 molecules are tested. Eventually, the labeled SMILES file and the SMILES file of the candidates will be generated, and finally the accuracy will be calculated to give the results.Execute the following code for evaluation.


```python
python evaluate-examples.py
```
