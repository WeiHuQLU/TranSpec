import torch
import csv
from util import MyDataSet
from rdkit.Chem import AllChem
from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
import numpy as np

def read_smiles_from_file(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file.readlines()]

def find_matching_lines(file_b_smiles, file_a_smiles):
    return [index for index, smile in enumerate(file_b_smiles) if smile in file_a_smiles]

def smi_tokenizer(smi):
    import re
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    tokens = re.findall(pattern, smi)
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

def normalize_spectrum(spectrum_data):
    scaler = MinMaxScaler(feature_range=(0, 100))
    spectrum_data_2d = spectrum_data.view(-1, spectrum_data.size(-1))
    normalized_data_2d = scaler.fit_transform(spectrum_data_2d.numpy())
    return torch.tensor(normalized_data_2d.reshape(spectrum_data.size()))

def process_ir_data(path, matching_lines):
    with open(path, "r") as ir_csv_file:
        ir_data = [list(map(float, row[1:])) for row in csv.reader(ir_csv_file)][1:]
    ir_data = torch.tensor([ir_data[i] for i in range(len(ir_data)) if i not in matching_lines])
    ir_data = ir_data.unsqueeze(1)
    ir_data = ir_data[:, :, 414:2883]

    x_original = torch.linspace(552, 3844, steps=2469)
    x_resampled = torch.linspace(552, 3844, steps=3000)

    resampled_data = []
    for spectrum in ir_data:
        f = interp1d(x_original, spectrum.squeeze(0), kind='cubic')
        spectrum_resampled = torch.tensor(f(x_resampled))
        spectrum_resampled = torch.clamp(spectrum_resampled, max=40 * torch.mean(spectrum_resampled))
        resampled_data.append(spectrum_resampled)

    return normalize_spectrum(torch.stack(resampled_data))

def augment_data(data, shift=None, scale=None, quantize=None):
    if shift:
        shift_value = torch.randint(-29, 31, (1,)).item()
        data = torch.roll(data, shifts=shift_value, dims=-1)
    if scale:
        scale_value = torch.rand(1).item() * 0.01 - 0.005 + 1
        data *= scale_value
    if quantize:
        quant_value = torch.randint(100, 300, (1,)).item()
        data = torch.round(data * quant_value) / quant_value
    return data

def loader(path1, path2, dictionary=None, shift=False, scale=False, quantize=False, seed=None):
    if seed:
        torch.manual_seed(seed)
        
    smiles_chars, smiles_split = [], []
    with open(path1, "r") as file:
        lines = file.readlines()
    for smile in lines:
        smile = smile.strip()
        can_mol = AllChem.MolFromSmiles(smile)
        Chem.Kekulize(can_mol)
        h_smiles = AllChem.MolToSmiles(can_mol, kekuleSmiles=True)
        tokens = smi_tokenizer(h_smiles).split(" ")
        tokens = ["BOS"] + tokens + ["EOS"]
        smiles_split.append(tokens)
        smiles_chars.extend([char for char in tokens if char not in smiles_chars])
    
    if dictionary is None:
        dic = {char: idx for idx, char in enumerate(["PAD", "BOS", "EOS"] + smiles_chars)}
    else:
        dic = dictionary
        
    atoms_smiles_idxs = [[dic[char] for char in smile] for smile in smiles_split]
    
    file_a_smiles = read_smiles_from_file(r"NIST_SMILES.txt")
    file_b_smiles = read_smiles_from_file(r"qm9s_SMILES.txt")
    matching_lines = find_matching_lines(file_b_smiles, file_a_smiles)
    
    ir_data_normalized = process_ir_data(path2, matching_lines)
    atoms_smiles_idxs = [atoms_smiles_idxs[i] for i in range(len(atoms_smiles_idxs)) if i not in matching_lines]
    
    train_spectrum, valid_spectrum, test_spectrum = [], [], []
    train_smiles_idxs, valid_smiles_idxs, test_smiles_idxs = [], [], []

    for i in range(len(ir_data_normalized)):
        if i % 10 < 8:
            spectrum = ir_data_normalized[i]
            spectrum = augment_data(spectrum, shift=shift, scale=scale, quantize=quantize)
            train_spectrum.append(spectrum.tolist())
            train_smiles_idxs.append(atoms_smiles_idxs[i])
        elif i % 10 == 8:
            valid_spectrum.append(ir_data_normalized[i].tolist())
            valid_smiles_idxs.append(atoms_smiles_idxs[i])
        else:
            test_spectrum.append(ir_data_normalized[i].tolist())
            test_smiles_idxs.append(atoms_smiles_idxs[i])

    print(f"Training samples: {len(train_smiles_idxs)}")
    print(f"Validation samples: {len(valid_smiles_idxs)}")
    print(f"Testing samples: {len(test_smiles_idxs)}")

    train_data = MyDataSet(train_spectrum, train_smiles_idxs)
    if shift or scale or quantize:
        file_suffix = []
        if shift:
            file_suffix.append("shift")
        if scale:
            file_suffix.append("scale")
        if quantize:
            file_suffix.append("quantize")

        file_suffix = "-".join(file_suffix)
        torch.save(train_data, f"qm9s_train_irdata_{file_suffix}_552-3844_del_same_molecule_as_NIST.pt")
    else:
        torch.save(train_data, "qm9s_train_irdata_552-3844_del_same_molecule_as_NIST.pt")
    
    if not (shift or scale or quantize):
        valid_data = MyDataSet(valid_spectrum, valid_smiles_idxs)
        torch.save(valid_data, "qm9s_valid_irdata_552-3844_del_same_molecule_as_NIST.pt")

    return dic

path1="qm9s_SMILES.txt"
path2="qm9s_irdata.csv"
loader(path1, path2, dictionary = {'PAD': 0, 'BOS': 1, 'EOS': 2, 'C': 3, 'N': 4, 'O': 5,
        'F': 6, 'Cl': 7, 'Br': 8, 'S': 9, '1': 10, '2': 11, '3': 12, '4': 13, '5': 14, '=': 15, '#': 16, '(': 17, ')': 18})



import torch
import csv
from util import MyDataSet
from rdkit.Chem import AllChem
from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import re

def smi_tokenizer(smi):  # smiles--->tokens
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

def shift_data(data, shift_range=(-29, 31)):
    shift = torch.randint(shift_range[0], shift_range[1], (1,))
    return torch.roll(data, shifts=int(shift), dims=-1)

def scale_data(data, scale_range=(0.995, 1.005)):
    scale = torch.rand(1) * 0.01 - 0.005 + 1
    return data * scale

def quantize_data(data, quant_range=(100, 300)):
    quant = torch.randint(quant_range[0], quant_range[1], (1,))
    return torch.round(data * quant) / quant

def normalize_spectrum(spectrum_data):
    scaler = MinMaxScaler(feature_range=(0, 100))
    spectrum_data_2d = spectrum_data.view(-1, spectrum_data.size(-1))
    normalized_data_2d = scaler.fit_transform(spectrum_data_2d.numpy())
    return torch.tensor(normalized_data_2d.reshape(spectrum_data.size()))

def loader(path1, path2, dictionary=None, shift=None, scale=None, quantize=None, seed=None):
    if seed:
        torch.manual_seed(seed)

    smiles_chars = []
    smiles_split = []

    with open(path1, "r") as file:
        lines = file.readlines()
    b, e = "BOS", "EOS"
    for smile in lines:
        d = []
        smiles = smile.strip()
        can_mol = AllChem.MolFromSmiles(smiles)
        Chem.Kekulize(can_mol)
        h_smiles = AllChem.MolToSmiles(can_mol, kekuleSmiles=True)
        tokens = smi_tokenizer(h_smiles).split(" ")
        for char in tokens:
            d.append(char)
            if char not in smiles_chars:
                smiles_chars.append(char)
        d.append(e)
        d.insert(0, b)
        smiles_split.append(d)

    if dictionary is None:
        dic = {"PAD": 0, "BOS": 1, "EOS": 2}
        for index, i in enumerate(smiles_chars):
            dic[i] = index + 3
    else:
        dic = dictionary

    atoms_smiles_idxs = [[dic.get(i) for i in pad_smile_split] for pad_smile_split in smiles_split]
    del smiles_split, smiles_chars

    ir_data = []
    with open(path2, "r") as ir_csv_file:
        ir_csv_reader_lines = csv.reader(ir_csv_file)
        for i_list in ir_csv_reader_lines:
            ir_list = [float(i) for i in i_list]
            ir_data.append(ir_list)
            del i_list

    ir_data = torch.tensor(ir_data)
    ir_data = ir_data.repeat(1, 1, 1).view(-1, 1, ir_data.size(-1))
    ir_data_normalized = normalize_spectrum(ir_data)
    del ir_data

    train_spectrum = []
    train_smiles_idxs = []
    valid_spectrum = []
    valid_smiles_idxs = []
    test_spectrum = []
    test_smiles_idxs = []

    for i in range(len(ir_data_normalized)):
        if i % 10 < 8: 
            if shift:
                ir_data_normalized[i] = shift_data(ir_data_normalized[i])
            if scale:
                ir_data_normalized[i] = scale_data(ir_data_normalized[i])
            if quantize:
                ir_data_normalized[i] = quantize_data(ir_data_normalized[i])
            train_spectrum.append(ir_data_normalized[i].tolist())
            train_smiles_idxs.append(atoms_smiles_idxs[i])
        elif i % 10 == 8: 
            valid_spectrum.append(ir_data_normalized[i].tolist())
            valid_smiles_idxs.append(atoms_smiles_idxs[i])
        else: 
            test_spectrum.append(ir_data_normalized[i].tolist())
            test_smiles_idxs.append(atoms_smiles_idxs[i])

    del ir_data_normalized, atoms_smiles_idxs

    train_data = MyDataSet(train_spectrum, train_smiles_idxs)
    if shift or scale or quantize:
        file_suffix = []
        if shift:
            file_suffix.append("shift")
        if scale:
            file_suffix.append("scale")
        if quantize:
            file_suffix.append("quantize")
        file_suffix = "-".join(file_suffix)
        torch.save(train_data, f"NIST_train_irdata_{file_suffix}.pt")
    else:
        torch.save(train_data, "NIST_train_irdata.pt")

    if not (shift or scale or quantize):
        valid_data = MyDataSet(valid_spectrum, valid_smiles_idxs)
        torch.save(valid_data, "NIST_valid_irdata.pt")
        del valid_data
        
        test_data = MyDataSet(test_spectrum, test_smiles_idxs)
        torch.save(test_data, "NIST_test_irdata.pt")
        del test_data

    return dic

path1=r"NIST_SMILES.txt"
path2=r"NIST_irdata.csv"
loader(path1, path2, dictionary = {'PAD': 0, 'BOS': 1, 'EOS': 2, 'C': 3, 'N': 4, 'O': 5,
        'F': 6, 'Cl': 7, 'Br': 8, 'S': 9, '1': 10, '2': 11, '3': 12, '4': 13, '5': 14, '=': 15, '#': 16, '(': 17, ')': 18})