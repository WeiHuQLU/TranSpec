import torch
from torch.utils.data import DataLoader
import csv
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit import Chem
import os
import sys
current_dir = os.path.dirname(os.path.abspath('__file__'))
src_dir = os.path.join(current_dir, '../src')
if src_dir not in sys.path:
    sys.path.append(src_dir)
from model import Model
from util import collate_fn
from search_methods import threshold_value_search

RDLogger.DisableLog('rdApp.*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Dictionary for SMILES tokens
dic = {'PAD': 0, 'BOS': 1, 'EOS': 2, 'C': 3, 'N': 4, 'O': 5,
        'F': 6, '1': 7, '2': 8, '3': 9, '4': 10, '5': 11, '=': 12, '#': 13, '(': 14, ')': 15}
dic_2 = dict(zip(dic.values(), dic.keys()))

# Device configuration
device = torch.device("cuda:0")

# Load test data and model
test_data = torch.load(r"dataset\qm9s_test_irdata-examples.pt")
test_loader = DataLoader(test_data, batch_size=1, collate_fn=collate_fn, pin_memory=True, shuffle=False)
print(len(test_data))

# Model parameters
vocab_size = len(dic)
d_model = 512
en_layers = 6
de_layers = 6
en_head = 8
de_head = 8
en_dim_feed = 2048
de_dim_feed = 2048
dropout = 0.1
max_len = 100

# Initialize model
model = Model(d_model,
          en_layers,
          de_layers,
          en_head,
          de_head,
          en_dim_feed,
          de_dim_feed,
          dropout,
          max_len,
          vocab_size,
          bias=True,
          use_cnn=True,
          use_mlp=False,
          input_channels=1,
          reshape_size=10)

# Load the trained model weights
model.load_state_dict(torch.load(r"optimal_qm9s_ircnn_examples.pt", map_location=device))
model.eval()

def evaluate_model(model, test_loader, dic, dic_2, threshold_value, candidate_limit, output_prefix):
    """
    Evaluates the model using threshold value search and writes the results to CSV files.
    
    Args:
        model: Trained model.
        test_loader: DataLoader for the test data.
        dic: Dictionary mapping tokens to indices.
        dic_2: Dictionary mapping indices to tokens.
        threshold_value: Probability threshold for candidate selection.
        candidate_limit: Maximum number of candidate SMILES strings.
        output_prefix: Prefix for output file names.
    
    Returns:
        accuracy: Top-1 accuracy of the model on the test data.
    """
    accuracy = 0
    a = 0
    
    label_file = f"{output_prefix}_label_ircnn-examples.csv"
    candidate_file = f"{output_prefix}_candidate_ircnn-examples.csv"

    with open(label_file, 'w', newline='') as label_csvfile, open(candidate_file, 'w', newline='') as candidate_csvfile:
        label_writer = csv.writer(label_csvfile)
        candidate_writer = csv.writer(candidate_csvfile)
        
        with torch.no_grad():
            for batch_index, (input_x, de_input, pad_mask, sub_mask, label_idx) in enumerate(test_loader):
                candidate_smiles, _ = threshold_value_search(model, input_x, start_symbol=1, dic_length=len(dic),
                                              threshold_value=threshold_value, candidate_limit=candidate_limit, dic=dic_2, 
                                              use_cnn=True, use_mlp=False, input_channels=1)
                
                de_input = de_input.squeeze(0)[1:].tolist()
                label_smiles = "".join([dic_2.get(i) for i in de_input])
                label_ = AllChem.MolToSmiles(AllChem.MolFromSmiles(label_smiles))

                valid_candidates = [AllChem.MolToSmiles(AllChem.MolFromSmiles(candidate)) for candidate in candidate_smiles if AllChem.MolFromSmiles(candidate)]

                label_writer.writerow([label_])
                candidate_writer.writerow(valid_candidates)

                print(f"{batch_index + 1}---valid_candidates_num: {len(valid_candidates)}")

                for idx, pre_ in enumerate(valid_candidates):
                    if pre_ == label_:
                        print("Correct prediction")
                        accuracy += 1
                        break

                print(label_)
                print("-----------------------------------")
                
                a += 1
                if a == 10:
                    break
            
    print("Correct count:", accuracy)
#     print("Accuracy:", accuracy / len(test_loader))
    return accuracy / len(test_loader)

def calculate_top_n_accuracy(predicted_smiles_file, label_smiles_file, top_n_values):
    """
    Calculates the top-N accuracy for the model predictions.
    
    Args:
        predicted_smiles_file: CSV file containing multiple columns of predicted SMILES strings.
        label_smiles_file: CSV file containing the true SMILES strings.
        top_n_values: List of top-N values to calculate accuracy for.
    
    Returns:
        accuracies: Dictionary mapping top-N values to accuracy percentages.
    """
    def smiles_to_mol(smiles):
        """ Convert a SMILES string to an RDKit Mol object. """
        return Chem.MolFromSmiles(smiles)

    def mol_to_smiles(mol):
        """ Convert an RDKit Mol object to a canonical SMILES string. """
        return Chem.MolToSmiles(mol) if mol is not None else None

    def compare_smiles(smiles1, smiles2):
        """ Compare two SMILES strings for equality. """
        mol1 = smiles_to_mol(smiles1)
        mol2 = smiles_to_mol(smiles2)
        if mol1 is None or mol2 is None:
            return False
        return mol_to_smiles(mol1) == mol_to_smiles(mol2)

    # Read predicted SMILES
    with open(predicted_smiles_file, 'r') as multi_file:
        multi_reader = csv.reader(multi_file)
        multi_smiles_list = [row for row in multi_reader]

    # Read true SMILES
    with open(label_smiles_file, 'r') as label_file:
        label_reader = csv.reader(label_file)
        label_smiles_list = [row[0] for row in label_reader]

    # Ensure the number of predictions matches the number of labels
    min_length = min(len(label_smiles_list), len(multi_smiles_list))

    def calculate_accuracy(predicted_smiles, label_smiles, top_n):
        correct_count = 0
        for i in range(min_length):
            label_smile = label_smiles[i]
            top_n_preds = predicted_smiles[i][:top_n]
            for pred_smile in top_n_preds:
                if compare_smiles(label_smile, pred_smile):
                    correct_count += 1
                    break
        print(correct_count)
        return correct_count / min_length

    accuracies = {}
    for top_n in top_n_values:
        accuracy = calculate_accuracy(multi_smiles_list, label_smiles_list, top_n)
        accuracies[top_n] = accuracy

    for top_n, accuracy in accuracies.items():
        print(f"Top-{top_n} accuracy: {accuracy:.2%}")
    
    return accuracies

# Run the evaluation
evaluate_model(model, test_loader, dic, dic_2, threshold_value=0.01, candidate_limit=20, output_prefix="0.01")

# Calculate and print top-N accuracies
calculate_top_n_accuracy("0.01_candidate_ircnn-examples.csv", "0.01_label_ircnn-examples.csv", [1, 3, 5, 10])
