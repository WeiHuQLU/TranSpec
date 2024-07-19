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
            ir_list = [float(i) for i in i_list[1:]]
            ir_data.append(ir_list)
            del i_list
            
    ir_data = ir_data[1:]
    ir_data = torch.tensor(ir_data).repeat(1, 1, 1).view(-1, 1, ir_data.size(-1))
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
        torch.save(train_data, f"qm9s_train_irdata-{file_suffix}.pt")
    else:
        torch.save(train_data, "qm9s_train_irdata.pt")

    if not (shift or scale or quantize):
        valid_data = MyDataSet(valid_spectrum, valid_smiles_idxs)
        torch.save(valid_data, "qm9s_valid_irdata.pt")
        del valid_data
        
        test_data = MyDataSet(test_spectrum, test_smiles_idxs)
        torch.save(test_data, "qm9s_test_irdata.pt")
        del test_data

    return dic

path1=r"qm9s_SMILES.txt"
path2=r"qm9s_irdata.csv"
loader(path1, path2, dictionary = {'PAD': 0, 'BOS': 1, 'EOS': 2, 'C': 3, 'N': 4, 'O': 5,
        'F': 6, '1': 7, '2': 8, '3': 9, '4': 10, '5': 11, '=': 12, '#': 13, '(': 14, ')': 15})


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

def loader(path1, path2, path3, dictionary=None):
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
            ir_list = [float(i) for i in i_list[1:]]
            ir_data.append(ir_list)
            del i_list

    rm_data = []
    with open(path3, "r") as rm_csv_file:
        rm_csv_reader_lines = csv.reader(rm_csv_file)  # 按行读取，每一行是一个分子的展宽数据
        for i_list in rm_csv_reader_lines:
            rm_list = [float(i) for i in i_list[1:]]
            rm_data.append(rm_list)
            del i_list
            
    ir_data = ir_data[1:]
    ir_data = torch.tensor(ir_data).repeat(1, 1, 1).view(-1, 1, ir_data.size(-1))
    ir_data_normalized = normalize_spectrum(ir_data)
    del ir_data
    
    rm_data = rm_data[1:]
    rm_data = torch.tensor(rm_data).repeat(1, 1, 1).view(-1, 1, rm_data.size(-1))
    rm_data_normalized = normalize_spectrum(rm_data)
    del rm_data
    
    spectrum_data = torch.concat([rm_data_normalized, ir_data_normalized], dim=1)   
    
    train_spectrum = []
    train_smiles_idxs = []
    valid_spectrum = []
    valid_smiles_idxs = []
    test_spectrum = []
    test_smiles_idxs = []

    for i in range(len(ir_data_normalized)):
        if i % 10 < 8: 
            train_spectrum.append(spectrum_data[i].tolist())
            train_smiles_idxs.append(atoms_smiles_idxs[i])
        elif i % 10 == 8: 
            valid_spectrum.append(spectrum_data[i].tolist())
            valid_smiles_idxs.append(atoms_smiles_idxs[i])
        else: 
            test_spectrum.append(spectrum_data[i].tolist())
            test_smiles_idxs.append(atoms_smiles_idxs[i])

    del ir_data_normalized, rm_data_normalized, atoms_smiles_idxs

    train_data = MyDataSet(train_spectrum, train_smiles_idxs)
    torch.save(train_data, f"qm9s_train_ir+ramandata.pt")
    del train_data
    
    valid_data = MyDataSet(valid_spectrum, valid_smiles_idxs)
    torch.save(valid_data, "qm9s_valid_ir+ramandata.pt")
    del valid_data

    test_data = MyDataSet(test_spectrum, test_smiles_idxs)
    torch.save(test_data, "qm9s_test_ir+ramandata.pt")
    del test_data

    return dic

path1=r"qm9s_SMILES.txt"
path2=r"qm9s_irdata.csv"
path3=r"qm9s_ramandata.csv"
loader(path1, path2, path3, dictionary = {'PAD': 0, 'BOS': 1, 'EOS': 2, 'C': 3, 'N': 4, 'O': 5,
        'F': 6, '1': 7, '2': 8, '3': 9, '4': 10, '5': 11, '=': 12, '#': 13, '(': 14, ')': 15})


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

def loader(path1, path2, path3, dictionary=None, shift=None, scale=None, quantize=None, seed=None):
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
            ir_list = [float(i) for i in i_list[1:]]
            ir_data.append(ir_list)
            del i_list

    rm_data = []
    with open(path3, "r") as rm_csv_file:
        rm_csv_reader_lines = csv.reader(rm_csv_file)  # 按行读取，每一行是一个分子的展宽数据
        for i_list in rm_csv_reader_lines:
            rm_list = [float(i) for i in i_list[1:]]
            rm_data.append(rm_list)
            del i_list
            
    ir_data = ir_data[1:]            
    ir_data = torch.tensor(ir_data).repeat(1, 1, 1).view(-1, 1, ir_data.size(-1))
    ir_data_normalized = normalize_spectrum(ir_data)
    for i in range(len(ir_data_normalized)):
        if shift:
            ir_data_normalized[i] = shift_data(ir_data_normalized[i])                

        if scale:   
            ir_data_normalized[i] = scale_data(ir_data_normalized[i])  

        if quantize:
            ir_data_normalized[i] = quantize_data(ir_data_normalized[i]) 
    del ir_data
    
    rm_data = rm_data[1:]
    rm_data = torch.tensor(rm_data).repeat(1, 1, 1).view(-1, 1, rm_data.size(-1))
    rm_data_normalized = normalize_spectrum(rm_data)
    for i in range(len(rm_data_normalized)):
        if shift:
            rm_data_normalized[i] = shift_data(rm_data_normalized[i])                

        if scale:   
            rm_data_normalized[i] = scale_data(rm_data_normalized[i])  

        if quantize:
            rm_data_normalized[i] = quantize_data(rm_data_normalized[i]) 
    del rm_data            

    spectrum_data = torch.concat([rm_data_normalized, ir_data_normalized], dim=1)
    
    train_spectrum = []
    train_smiles_idxs = []

    for i in range(len(spectrum_data)):
        if i % 10 < 8: 
            train_spectrum.append(spectrum_data[i].tolist())
            train_smiles_idxs.append(atoms_smiles_idxs[i])
    del ir_data_normalized,rm_data_normalized,atoms_smiles_idxs

    train_data = MyDataSet(train_spectrum, train_smiles_idxs)
    
    file_suffix = []
    if shift:
        file_suffix.append("shift")
    if scale:
        file_suffix.append("scale")
    if quantize:
        file_suffix.append("quantize")
    file_suffix = "-".join(file_suffix)
    torch.save(train_data, f"qm9s_train_ir+ramandata-{file_suffix}.pt")
        
    return dic

path1=r"qm9s_SMILES.txt"
path2=r"qm9s_irdata.csv"
path3=r"qm9s_ramandata.csv"
loader(path1, path2, path3, dictionary = {'PAD': 0, 'BOS': 1, 'EOS': 2, 'C': 3, 'N': 4, 'O': 5,
        'F': 6, '1': 7, '2': 8, '3': 9, '4': 10, '5': 11, '=': 12, '#': 13, '(': 14, ')': 15}, shift=True, seed=42)