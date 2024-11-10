# System requirements

The required operating system, GPU specifications, software packages, their versions, and download instructions can be found in the requirements.txt file.  Detailed information is provided below.

## All software dependencies and operating systems (including version numbers)

Python and system requirements 

python==3.9.0  # Requires Python version 3.9.0

os: Windows 11  # Compatible with Windows 11

cuda: 11.8+  # Requires NVIDIA GPU with CUDA version 11.8 or higher

## Versions the software has been tested on

Python dependencies

torch==2.0.0+cu118

numpy==1.24.1

pandas==2.1.3

rdkit==2022.9.5

scipy==1.11.4

scikit-learn==1.5.2

torch-cluster==1.6.1+pt20cu118

torch-scatter==2.1.1+pt20cu118

torch-sparse==0.6.17+pt20cu118

torch-spline-conv==1.2.2+pt20cu118

torch-geometric==2.4.0

## Any required non-standard hardware

None

# Installation guide

## Instructions

Note: To download PyTorch, use the following command:

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

The following packages are dependencies required for torch-geometric==2.4.0:

torch-cluster==1.6.1+pt20cu118

torch-scatter==2.1.1+pt20cu118

torch-sparse==0.6.17+pt20cu118

torch-spline-conv==1.2.2+pt20cu118

These dependencies must be downloaded offline from the following link: 

https://data.pyg.org/whl/torch-2.0.0%2Bcu118.html 

After downloading the corresponding versions, navigate to the directory where the files are located and install them using:

pip install <downloaded_package.whl>

Finally, install torch-geometric with the following command:

pip install torch-geometric==2.4.0

Other libraries can be installed using pip install as usual.

## Typical install time on a "normal" desktop computer

Setting up the environment takes approximately 10 minutes.

# Demo

## Step 1. Data preprocessing

### Instructions to run on data

We take the first 5000 molecules of the QM9S dataset as an example. First, we prepare two data files in the examples/data folder of figshare(https://figshare.com/s/1581c92625dc3803f983), which are the IR spectra data file and the corresponding SMILES file. Find the path of the folder where qm9sdata_processing-examples.py is currently located in Anaconda Prompt, put the two data files and qm9sdata_processing-examples.py into the same path, and then execute the preprocessing operation with the following code.

```python
python qm9sdata_processing-examples.py
```

### Expected output

qm9s_train_irdata-examples.pt

qm9s_valid_irdata-examples.pt

qm9s_test_irdata-examples.pt

### Expected run time for demo on a "normal" desktop computer

It takes about 2 minutes.

## Step 2. Training

### Instructions to run on data

Through preprocessing, we obtained the infrared spectral data that went into the model under the current path. Training set, validation set and test set files were generated in the examples/dataset folder of figshare. Here, we set the epoch of the training process to 5 (actually the epoch is 200) and use the following code to perform the operation of training the model in the current path.

```python
python train-examples.py
```

### Expected output

optimal_qm9s_ircnn_examples.pt

### Expected run time for demo on a "normal" desktop computer

It takes about 5 minutes.

## Step 3. Inference

### Instructions to run on data

Models will generate optimal_qm9s_ircnn_examples.pt file in the current path after the model passes the training, then we will load this model parameter file into the evaluation code with a maximum limit of 20 candidate molecules and only the first 10 molecules will be tested. Finally, the labeled SMILES file and the SMILES file of the candidate molecules will be generated, and finally the accuracy will be calculated to produce the results. 

```python
python evaluate-examples.py
```

### Expected output

0.01_candidate_ircnn-examples.csv

0.01_label_ircnn-examples.csv

### Expected run time for demo on a "normal" desktop computer

It takes about 1 minutes.

# Instructions for use

## How to run the software on your data

Setp 1. Process the dataset, dividing into training, validation, and test sets.

```python
python qm9sdata_processing.py
```

```python
python NISTdata_processing.py
```

Setp 2. Train TranSpec on dataset.

```python
python train.py
```

Setp 3. Perform model fusion.

```python
python model_fusion.py
```

Setp 4. Train SpecGNN, and re-rank the potential SMILES candidates.

```python
python SpecGNN.py
```

Setp 5. Filter the SMILES candidates using molecular mass.

```python
python mass_screening.py
```

## Reproduction instructions

Setp 1. Utilize data_processing.py script to devide the .CSV files (the data of the infrared and Raman spectra) and the .TXT files (the corresponding SMILES) into training, validation, and test sets. The proportion of the training, validation, and test sets are 80%, 10% and 10%, respectively. Here the .CSV files and the .TXT files could be the QM9S IR, QM9S Raman and experiment IR data. The data augmentation can be applied to the training set during preprocessing by setting the keyword of "shift，scale or quantize" to "Ture" in data_processing.py script. The final output are three .PT files that corresponds to the training, validation, and test sets.

Setp 2. Use the train.py script to train TranSpec on the prepared dataset outputted from Setp 1, resulting in a .PT file written with the ML parameters. Then use evaluate.py script to evaluate the accuracy of TranSpec on the test set, providing accuracy metrics and a .CSV file that contains all SMILES candidates.

Setp 3. Use the model_fusion.py script to perform model fusion, making the correct SMILES rank higher.

Setp 4. Use SpecGNN.py script and an experimental IR spectra dataset to train SpecGNN, realizing the direct molecular spectra simulation based solely on SMILES. Then utilize SpecGNN to simulated all molecules in experimental IR test dataset. Based on the spectra similarities, re-rank the potential SMILES candidates.

Setp 5. Apply the mass_screening.py script to further filter the SMILES candidates, making the correct SMILES rank higher.

Setp 6. Evaluate the model's ability to recognize functional groups using the functional_groups.py script.

Setp 7. Use the recognizing_isomer.py script to assess the model's capability in distinguishing isomers and homologous.










