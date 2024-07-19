from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

# Function to convert a SMILES string to an RDKit molecule
def smiles_to_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol

# Function to calculate Tanimoto similarity between two SMILES strings
def calculate_tanimoto_similarity(smiles1, smiles2):
    mol1 = smiles_to_molecule(smiles1)
    mol2 = smiles_to_molecule(smiles2)
    if mol1 is None or mol2 is None:
        return None
    
    # Calculate Morgan fingerprints
    fp1 = AllChem.GetMorganFingerprint(mol1, 2)
    fp2 = AllChem.GetMorganFingerprint(mol2, 2)
    
    # Calculate Tanimoto similarity
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return similarity

# File paths for predicted SMILES and label SMILES
predicted_smiles_file = r"0.01nist-ir-cnn+mlp-16model.csv"
label_smiles_file = r"0.01label_nistircnn.csv"

# Initialize interval counter
interval_counts = [0] * 20

with open(predicted_smiles_file, 'r') as predicted_file, open(label_smiles_file, 'r') as label_file:
    # Compare predicted and label SMILES line by line
    for predicted_line, label_line in zip(predicted_file, label_file):
        predicted_smiles_list = predicted_line.strip().split(',')[:10]  # Get the top 10 predicted SMILES
        label_smiles = label_line.strip().split(',')[0]  # Get the label SMILES
        
        # Calculate similarity for each predicted SMILES with the label SMILES and update counter
        for predicted_smiles in predicted_smiles_list:
            # Skip empty predicted SMILES
            if not predicted_smiles.strip():
                continue
            similarity = calculate_tanimoto_similarity(predicted_smiles, label_smiles)

            # Count the similarity value in the appropriate interval
            if similarity is not None:
                if 0 <= similarity <= 0.05:
                    interval_counts[0] += 1
                elif 0.05 < similarity <= 0.1:
                    interval_counts[1] += 1
                elif 0.1 < similarity <= 0.15:
                    interval_counts[2] += 1
                elif 0.15 < similarity <= 0.2:
                    interval_counts[3] += 1
                elif 0.2 < similarity <= 0.25:
                    interval_counts[4] += 1
                elif 0.25 < similarity <= 0.3:
                    interval_counts[5] += 1
                elif 0.3 < similarity <= 0.35:
                    interval_counts[6] += 1
                elif 0.35 < similarity <= 0.4:
                    interval_counts[7] += 1
                elif 0.4 < similarity <= 0.45:
                    interval_counts[8] += 1
                elif 0.45 < similarity <= 0.5:
                    interval_counts[9] += 1
                elif 0.5 < similarity <= 0.55:
                    interval_counts[10] += 1
                elif 0.55 < similarity <= 0.6:
                    interval_counts[11] += 1
                elif 0.6 < similarity <= 0.65:
                    interval_counts[12] += 1
                elif 0.65 < similarity <= 0.7:
                    interval_counts[13] += 1
                elif 0.7 < similarity <= 0.75:
                    interval_counts[14] += 1
                elif 0.75 < similarity <= 0.8:
                    interval_counts[15] += 1
                elif 0.8 < similarity <= 0.85:
                    interval_counts[16] += 1
                elif 0.85 < similarity <= 0.9:
                    interval_counts[17] += 1
                elif 0.9 < similarity <= 0.95:
                    interval_counts[18] += 1
                elif 0.95 < similarity < 1:
                    interval_counts[19] += 1

# Calculate the ratio for each interval
total_samples = sum(interval_counts)
print(total_samples)
print(interval_counts)
interval_ratios = [count / total_samples for count in interval_counts]

# Print the ratio for each interval
for i, ratio in enumerate(interval_ratios):
    print(f"Similarity Ratio in Interval {i+1}: {ratio:.2%}")
