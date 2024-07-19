import os
import csv
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit import RDLogger

# Disable RDKit logging and TensorFlow messages
RDLogger.DisableLog('rdApp.*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Function to fill each row to 500 columns
def fill_to_500_columns(file_path, output_path):
    with open(file_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            items = [item.strip() for item in line.strip().split(',')]
            filled_items = items + [''] * (500 - len(items))
            outfile.write(','.join(filled_items) + '\n')

# Fill the required files to 500 columns
file_paths = [
    r"0.01candidate_nistircnn.csv",
    r"0.01candidate_nistircnn-quantize.csv",
    r"0.01candidate_nistircnn-scale.csv",
    r"0.01candidate_nistircnn-shift.csv",
    r"0.01candidate_nistirmlp.csv",
    r"0.01candidate_nistirmlp-qunatize.csv",
    r"0.01candidate_nistirmlp-scale.csv",
    r"0.01candidate_nistirmlp-shift.csv"
]

output_paths = [
    '0.01nist-ircnn.csv',
    '0.01nist-ircnn-quantize.csv',
    '0.01nist-ircnn-scale.csv',
    '0.01nist-ircnn-shift.csv',
    '0.01nist-irmlp.csv',
    '0.01nist-irmlp-quantize.csv',
    '0.01nist-irmlp-scale.csv',
    '0.01nist-irmlp-shift.csv',
]

for file_path, output_path in zip(file_paths, output_paths):
    fill_to_500_columns(file_path, output_path)

# Load filled CSV files into dataframes
submissions = [pd.read_csv(output_path, dtype=str, header=None) for output_path in output_paths]

# Load one of the filled files as the base for the ensemble
ensemble = pd.read_csv('0.01nist-ircnn-no.csv', dtype=str, header=None)

# Number of rows to process
num_rows = ensemble.shape[0]

# Process each row
for i in range(num_rows):
    print(i)
    vote_counts = {}
    
    # Process each submission file
    for sub in submissions:
        for idx, smile in enumerate(sub.iloc[i]):
            mol = AllChem.MolFromSmiles(str(smile))
            if mol is not None:
                if smile in vote_counts:
                    vote_counts[smile] += 200 - idx
                else:
                    vote_counts[smile] = 200 - idx
                
                # Add weights based on rank
                if idx == 0:
                    vote_counts[smile] += 60
                if idx < 3:
                    vote_counts[smile] += 50
                if idx < 5:
                    vote_counts[smile] += 40
                if idx < 10:
                    vote_counts[smile] += 30
                if idx < 20:
                    vote_counts[smile] += 20
                if idx < 50:
                    vote_counts[smile] += 10
    
    # Sort votes and select top 100 SMILES
    sorted_votes = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Update ensemble with top 100 SMILES
    for rank, (smile, _) in enumerate(sorted_votes[:100]):
        ensemble.iloc[i, rank] = smile

# Save ensemble to CSV
ensemble.to_csv('0.01nist-ir-cnn+mlp-8model.csv', index=False, header=False)

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

# Calculate and print top-N accuracies
calculate_top_n_accuracy("0.01nist-ir-cnn+mlp-8model.csv", "0.01label_nistircnn.csv", [1, 3, 5, 10])

