import csv
from collections import defaultdict
from rdkit import Chem
from itertools import combinations

# Function to convert SMILES to chemical formula
def smiles_to_formula(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_count = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atom_count[symbol] = atom_count.get(symbol, 0) + 1

    for atom in mol.GetAtoms():
        total_hydrogens = atom.GetTotalNumHs()
        if total_hydrogens > 0:
            atom_count['H'] = atom_count.get('H', 0) + total_hydrogens

    formula = ''.join(f"{symbol}{count if count > 1 else ''}" for symbol, count in sorted(atom_count.items()))
    return formula

# Read CSV file and store SMILES and row numbers in a dictionary
def read_smiles_from_csv(file_path):
    smiles_dict = defaultdict(list)
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            smiles = row[0]
            formula = smiles_to_formula(smiles)
            if formula is not None:
                smiles_dict[formula].append(idx + 1)
    return smiles_dict

# Generate pairwise combinations of indices
def generate_combinations(smiles_dict):
    result = [indices for indices in smiles_dict.values() if len(indices) > 1]
    combined_list = [list(combination) for sublist in result for combination in combinations(sublist, 2)]
    return combined_list

# Write prediction results to a file
def write_prediction_results(label_file_path, predicted_file_path, output_file_path, top_n):
    with open(label_file_path, 'r') as label_file, open(predicted_file_path, 'r') as predicted_file:
        label_reader = csv.reader(label_file)
        predicted_reader = csv.reader(predicted_file)
        
        label_smiles = [row[0] for row in label_reader]
        predicted_smiles = [row[:top_n] for row in predicted_reader]
        min_length = min(len(label_smiles), len(predicted_smiles))

        with open(output_file_path, 'w') as result_file:
            for i in range(min_length):
                label_smile = label_smiles[i]
                pred_smiles = [s.strip() for s in predicted_smiles[i]]
                result_file.write("Correct prediction\n" if label_smile in pred_smiles else "Error prediction\n")

# Update result list with prediction results
def update_result_list(result_list, prediction_file_path):
    with open(prediction_file_path, 'r') as result_file:
        results = [result.strip() for result in result_file.readlines()]
    
    updated_result_list = []
    for sublist in result_list:
        updated_sublist = ["Correct prediction" if results[row - 1] == "Correct prediction" else "Error prediction" for row in sublist]
        updated_result_list.append(updated_sublist)
    return updated_result_list

# Calculate accuracy of predictions
def calculate_accuracy(result_list, updated_result_list):
    correct_count = sum(all(result == "Correct prediction" for result in sublist) for sublist in updated_result_list)
    accuracy = correct_count / len(result_list)
    return correct_count, accuracy

# Main function to execute the above steps
def main():
    label_file_path = r"0.01label_nistircnn.csv"
    predicted_file_path = r"0.01nist-ir-cnn+mlp-16model.csv"
    
    smiles_dict = read_smiles_from_csv(label_file_path)
    combined_list = generate_combinations(smiles_dict)
    
    for top_n, output_file_path in zip([1, 3, 5, 10], ["prediction_results-top1.txt", "prediction_results-top3.txt", "prediction_results-top5.txt", "prediction_results-top10.txt"]):
        write_prediction_results(label_file_path, predicted_file_path, output_file_path, top_n)
        updated_result_list = update_result_list(combined_list, output_file_path)
        correct_count, accuracy = calculate_accuracy(combined_list, updated_result_list)
        print(f"Top {top_n} - Correct count: {correct_count}, Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
