from torch import nn
from torch.utils.data import Dataset
import torch
from torch.autograd import Variable
import math
import numpy as np

# Define the positional encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Create a tensor to hold the positional encodings
        pe = torch.zeros(max_len, d_model)
        # Create a tensor to hold the positions
        position = torch.arange(0, max_len).unsqueeze(1)
        # Create a tensor for the div_term calculation
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a new dimension to pe
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encodings to the input tensor
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

# Define a custom dataset class
class MyDataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)

# Define a custom loss class
class Loss(nn.Module):
    def __init__(self, model):
        super(Loss, self).__init__()
        self.model = model
        self.type_loss = nn.CrossEntropyLoss()

    def forward(self, en, de_1, tgt_mask, tgt_key_padding_mask, label1):
        # Forward pass through the model
        pre_type = self.model(en, de_1, tgt_mask, tgt_key_padding_mask)
        # Compute the cross-entropy loss
        type_loss = self.type_loss(pre_type.view(-1, pre_type.size(-1)), label1.view(-1))
        return type_loss, pre_type.detach()

# Define a custom accuracy calculation class
class TypeAccuracy(nn.Module):
    def __init__(self):
        super(TypeAccuracy, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pre_types, labels_type):
        pre_types = self.softmax(pre_types)
        correct_smile = []
        error_smile = []
        accuracy = 0
        for i in range(len(pre_types)):
            # Get the predicted and true labels
            a = torch.argmax(pre_types[i], dim=-1)
            b = labels_type[i]
            # Check if the prediction is correct
            if a.equal(b):
                accuracy += 1
                correct_smile.append([a.tolist(), b.tolist()])
            else:
                error_smile.append([a.tolist(), b.tolist()])

        return accuracy, correct_smile, error_smile

# Define a function to create padding masks
def padding_mask(sentence, pad_idx=0):
    mask = (sentence == pad_idx)
    return mask

# Define a function to create subsequent masks
def subsequent_mask(size):
    attn_shape = (size, size)
    subsequent_mask_ = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask_) == 1

# Define a collate function for the data loader
def collate_fn(batch_data):
    en_input = []
    de_input = []
    label_idx = []
    batch_length = []

    # Extract inputs and labels from the batch
    for j, k in batch_data:
        en_input.append(j)
        de_input.append(k[:-1])
        label_idx.append(k[1:])
        batch_length.append(len(k[1:]))

    input_x = torch.tensor(en_input)
    batch_max = max(batch_length)
    de_in = []

    # Pad the decoder input sequences
    for smile_id in de_input:
        smile_id = smile_id + [0] * (batch_max - len(smile_id))
        de_in.append(smile_id)

    de_in = torch.tensor(de_in)
    pad_mask = padding_mask(de_in, pad_idx=0)
    sub_mask = subsequent_mask(batch_max)

    labels = []
    # Pad the label sequences
    for label in label_idx:
        label = label + [0] * (batch_max - len(label))
        labels.append(label)

    labels = torch.tensor(labels)
    return input_x, de_in, pad_mask, sub_mask, labels
