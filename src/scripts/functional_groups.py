import csv
from rdkit import Chem

def extract_functional_groups_with_smarts(smiles, smarts):
    """
    Extract functional groups from a SMILES string based on a SMARTS pattern.
    
    :param smiles: SMILES string
    :param smarts: SMARTS pattern
    :return: List of functional group matches
    """
    groups = []
    mol = Chem.MolFromSmiles(smiles)
    matcher = Chem.MolFromSmarts(smarts)
    functional_groups = mol.GetSubstructMatches(matcher)
    for i in range(len(functional_groups)):
        groups.append(i + 1)
    return groups

def main():
    # Define the list of SMARTS patterns to match
    smarts_patterns = [
        '[CX4]',               
        '[$([CX3]=[CX3])]',
        '[$([CX2]#C)]',
        '[$([cX3](:*):*),$([cX2+](:*):*)]',
        '[#6][OX2H]',
        '[CX3H1](=O)[#6]',
        '[#6][CX3](=O)[#6]',
        '[#6][CX3](=O)[OX2H0][#6]',
        '[OD2]([#6])[#6]',
        '[NX3;!$(NC=O)]',
        '[NX1]#[CX2]',
        '[NX3][CX3](=[OX1])[#6]'
    ]

    # Open label SMILES file and predicted SMILES file
    with open(r"0.01label_nistircnn.csv", 'r') as label_file, open(r"0.01nist-ir-cnn+mlp-16model.csv", 'r') as predicted_file:
        # Read CSV files
        label_reader = csv.reader(label_file)
        predicted_reader = csv.reader(predicted_file)

        # Initialize counters
        total_samples = 0
        correct_predictions = 0

        # Initialize confusion matrices for each SMARTS pattern
        confusion_matrices = {pattern: {'True Positive': 0, 'False Positive': 0, 'False Negative': 0} for pattern in smarts_patterns}

        # Compare SMILES line by line and update confusion matrices
        for label_row, predicted_row in zip(label_reader, predicted_reader):
            label_smiles = label_row[0]
            predicted_smiles = predicted_row[0] if predicted_row else ''  # Handle empty rows
            if not predicted_smiles:
                continue  # Skip empty rows

            total_samples += 1

            for pattern in smarts_patterns:
                label_groups = extract_functional_groups_with_smarts(label_smiles, pattern)
                predicted_groups = extract_functional_groups_with_smarts(predicted_smiles, pattern)

                # Update confusion matrix
                for group in predicted_groups:
                    if group in label_groups:
                        confusion_matrices[pattern]['True Positive'] += 1
                    else:
                        confusion_matrices[pattern]['False Positive'] += 1

                for group in label_groups:
                    if group not in predicted_groups:
                        confusion_matrices[pattern]['False Negative'] += 1

        # Calculate and print precision, recall, and F1 Score for each SMARTS pattern
        for pattern in smarts_patterns:
            tp = confusion_matrices[pattern]['True Positive']
            fp = confusion_matrices[pattern]['False Positive']
            fn = confusion_matrices[pattern]['False Negative']

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"Pattern: {pattern}")
            print(f"  Precision: {precision}")
            print(f"  Recall: {recall}")
            print(f"  F1 Score: {f1_score}")

if __name__ == '__main__':
    main()



import csv
from rdkit import Chem

# Function to extract functional groups from a SMILES string based on a SMARTS pattern
def extract_functional_groups_with_smarts(smiles, smarts):
    """
    Extract functional groups from a SMILES string based on a SMARTS pattern.
    
    :param smiles: SMILES string
    :param smarts: SMARTS pattern
    :return: List of functional group matches
    """
    groups = []
    mol = Chem.MolFromSmiles(smiles)
    matcher = Chem.MolFromSmarts(smarts)
    functional_groups = mol.GetSubstructMatches(matcher)
    for i in range(len(functional_groups)):
        groups.append(i + 1)
    return groups

# Define the list of SMARTS patterns to match
smarts_patterns = [
        '[CX4]',               
        '[$([CX3]=[CX3])]',
        '[$([CX2]#C)]',
        '[$([cX3](:*):*),$([cX2+](:*):*)]',
        '[#6][OX2H]',
        '[CX3H1](=O)[#6]',
        '[#6][CX3](=O)[#6]',
        '[#6][CX3](=O)[OX2H0][#6]',
        '[OD2]([#6])[#6]',
        '[NX3;!$(NC=O)]',
        '[NX1]#[CX2]',
        '[NX3][CX3](=[OX1])[#6]'
    ]

# Open label SMILES file and predicted SMILES file
with open(r"0.01label_nistircnn.csv", 'r') as label_file, open(r"0.01nist-ir-cnn+mlp-16model.csv", 'r') as predicted_file:
    # Read CSV files
    label_reader = csv.reader(label_file)
    predicted_reader = csv.reader(predicted_file)
    
    # Initialize counters
    total_samples = 0
    correct_predictions = 0

    # Initialize accuracy dictionary for each SMARTS pattern
    accuracy_dict = {pattern: {'total_samples': 0, 'correct_predictions': 0} for pattern in smarts_patterns}

    # Compare SMILES line by line and update accuracy dictionary
    for label_row, predicted_row in zip(label_reader, predicted_reader):
        label_smiles = label_row[0]
        predicted_smiles = predicted_row[0] if predicted_row else ''  # Handle empty rows
        if not predicted_smiles:
            continue  # Skip empty rows

        for pattern in smarts_patterns:
            label_groups = extract_functional_groups_with_smarts(label_smiles, pattern)
            predicted_groups = extract_functional_groups_with_smarts(predicted_smiles, pattern)
            
            # Update counters
            if not label_groups:
                continue  # Skip if both lists are empty

            accuracy_dict[pattern]['total_samples'] += 1
            if label_groups == predicted_groups:
                accuracy_dict[pattern]['correct_predictions'] += 1

    # Calculate and print accuracy for each SMARTS pattern
    for pattern in smarts_patterns:
        total_samples = accuracy_dict[pattern]['total_samples']
        correct_predictions = accuracy_dict[pattern]['correct_predictions']
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        print(f"Pattern: {pattern}")
        print(f"  Correct Predictions: {correct_predictions}")
        print(f"  Total Samples: {total_samples}")
        print(f"  Accuracy: {accuracy}")

        

from rdkit import Chem

def count_molecules_with_smarts(filename, smarts_list):
    """
    Count the number of molecules in a file that match given SMARTS patterns.

    :param filename: Path to the file containing SMILES strings
    :param smarts_list: List of SMARTS patterns to search for
    :return: Dictionary with SMARTS patterns as keys and counts as values
    """
    counts = {smarts: 0 for smarts in smarts_list}
    with open(filename, 'r') as file:
        for line in file:
            smiles = line.strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                for smarts in smarts_list:
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
                        counts[smarts] += 1
    return counts

def main():
    filename = r"NIST_SMILES.txt"  # Replace with your txt file path
    smarts_list = [
        '[CX4]',               # Example SMARTS patterns
        '[$([CX3]=[CX3])]',
        '[$([CX2]#C)]',
        '[$([cX3](:*):*),$([cX2+](:*):*)]',
        '[#6][OX2H]',
        '[CX3H1](=O)[#6]',
        '[#6][CX3](=O)[#6]',
        '[#6][CX3](=O)[OX2H0][#6]',
        '[OD2]([#6])[#6]',
        '[NX3;!$(NC=O)]',
        '[NX1]#[CX2]',
        '[NX3][CX3](=[OX1])[#6]'
    ]  # Add all 12 SMARTS patterns here

    counts = count_molecules_with_smarts(filename, smarts_list)
    for smarts, count in counts.items():
        print(f'SMARTS: {smarts} -> Count: {count}')

if __name__ == '__main__':
    main()

