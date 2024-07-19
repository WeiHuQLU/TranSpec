import csv
from rdkit import Chem
from rdkit.Chem import Descriptors

def calculate_exact_mass(input_csv_path, output_csv_path, multi_column=False):
    with open(input_csv_path, 'r') as infile, open(output_csv_path, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            row = [cell for cell in row if cell] if multi_column else row
            exact_masses = []
            for smiles in row:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    exact_masses.append(f"{Descriptors.ExactMolWt(mol):.4f}")
                else:
                    exact_masses.append('Invalid SMILES')
            writer.writerow(exact_masses if multi_column else exact_masses[:1])

calculate_exact_mass(r"0.01label_nistircnn.csv", "0.01label_nistircnn-mass.csv")
calculate_exact_mass(r"0.01nist-ir-cnn+mlp-16model.csv", "0.01nist-ir-cnn+mlp-16model-mass.csv", multi_column=True)

def match_indices(single_mass_path, multi_mass_path, output_path):
    with open(single_mass_path, 'r') as mass_file, open(multi_mass_path, 'r') as multi_mass_file, open(output_path, 'w', newline='') as output_file:
        single_mass_list = [row[0] for row in csv.reader(mass_file)]
        multi_mass_list = [row for row in csv.reader(multi_mass_file)]
        writer = csv.writer(output_file)
        
        for i, (single_mass, multi_mass_row) in enumerate(zip(single_mass_list, multi_mass_list)):
            matched_indices = [str(index) for index, mass in enumerate(multi_mass_row) if mass == single_mass]
            writer.writerow([i] + matched_indices)

match_indices("0.01label_nistircnn-mass.csv", "0.01nist-ir-cnn+mlp-16model-mass.csv", "matched_indices.csv")

def extract_matched_smiles(multi_smiles_path, matched_indices_path, output_path):
    with open(multi_smiles_path, 'r') as multi_file, open(matched_indices_path, 'r') as matched_file, open(output_path, 'w', newline='') as output_file:
        multi_smiles_list = [row for row in csv.reader(multi_file)]
        matched_list = [row for row in csv.reader(matched_file)]
        writer = csv.writer(output_file)
        
        for row_index, matched_row in enumerate(matched_list):
            new_row = [row_index]
            if matched_row and len(matched_row) > 1:
                for col_index_str in matched_row[1:]:
                    if col_index_str:
                        col_index = int(col_index_str)
                        if col_index < len(multi_smiles_list[row_index]):
                            new_row.append(multi_smiles_list[row_index][col_index])
            writer.writerow(new_row)

extract_matched_smiles("0.01nist-ir-cnn+mlp-16model.csv", "matched_indices.csv", "matched_smiles.csv")

def calculate_top_n_accuracy(predicted_smiles_path, label_smiles_path, top_n_values):
    def compare_smiles(smiles1, smiles2):
        mol1, mol2 = Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)
        return Chem.MolToSmiles(mol1) == Chem.MolToSmiles(mol2) if mol1 and mol2 else False
    
    with open(predicted_smiles_path, 'r') as multi_file, open(label_smiles_path, 'r') as label_file:
        multi_smiles_list = [row[1:] for row in csv.reader(multi_file)]
        label_smiles_list = [row[0] for row in csv.reader(label_file)]
        
    min_length = min(len(label_smiles_list), len(multi_smiles_list))
    
    accuracies = {}
    for top_n in top_n_values:
        correct_count = 0
        for i in range(min_length):
            label_smile = label_smiles_list[i]
            top_n_preds = multi_smiles_list[i][:top_n]
            if any(compare_smiles(label_smile, pred_smile) for pred_smile in top_n_preds):
                correct_count += 1
        accuracies[top_n] = correct_count / min_length
        print(f"Top-{top_n} accuracy: {accuracies[top_n]:.2%}")

calculate_top_n_accuracy("matched_smiles.csv", "0.01label_nistircnn.csv", [1, 3, 5, 10])
