import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
import time
import os
from model import Model
from util import Loss, TypeAccuracy, collate_fn

# Suppress TensorFlow log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dic = {'PAD': 0, 'BOS': 1, 'EOS': 2, 'C': 3, 'N': 4, 'O': 5,
        'F': 6, '1': 7, '2': 8, '3': 9, '4': 10, '5': 11, '=': 12, '#': 13, '(': 14, ')': 15}

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Load datasets
train_data = torch.load(r"qm9s_train_irdata.pt")
valid_data = torch.load(r"qm9s_valid_irdata.pt")
train_length = len(train_data)
valid_length = len(valid_data)

print(f"Training samples: {train_length}")
print(f"Validation samples: {valid_length}")

# Create data loaders
train_loader = DataLoader(train_data, batch_size=32, collate_fn=collate_fn, pin_memory=True, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, collate_fn=collate_fn, pin_memory=True, shuffle=True)

# Define model hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 200
vocab_size = len(dic)  # Set according to the actual dictionary size
d_model = 512
en_layers = 6
de_layers = 6
en_head = 8
de_head = 8
en_dim_feed = 2048
de_dim_feed = 2048
dropout = 0.1
max_len = 100


# Initialize the model
model = Model(d_model, en_layers, de_layers, en_head, de_head, en_dim_feed, de_dim_feed, dropout, max_len, vocab_size,
              bias=True, use_cnn=True, use_mlp=False, input_channels=1, reshape_size=10)

# pretrained_params = torch.load(r"optimal_qm9s_ircnn_no-augmentation.pt")
# model.load_state_dict(pretrained_params)

model.to(device)

# Initialize loss function and accuracy calculation
loss_fn = Loss(model)
loss_fn.to(device)
type_accuracy = TypeAccuracy()
type_accuracy.to(device)

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.99), weight_decay=5e-4)

# Define training function
def train_one_epoch(model, loader, optimizer, loss_fn, accuracy_fn, device):
    model.train()
    total_loss = 0
    total_accuracy = 0
    for batch_index, (input_x, de_input, pad_mask, sub_mask, label_idx) in enumerate(loader):
        input_x, de_input, pad_mask, sub_mask, label_idx = input_x.to(device), de_input.to(device), pad_mask.to(device), sub_mask.to(device), label_idx.to(device)

        optimizer.zero_grad()  # Reset gradients
        loss, pred_types = loss_fn(input_x, de_input, sub_mask, pad_mask, label_idx)  # Compute loss and predictions
        loss.backward()  # Backpropagate
        optimizer.step()  # Update model parameters

        total_loss += loss.item()
        accuracy = accuracy_fn(pred_types, label_idx)[0]  # Calculate accuracy
        total_accuracy += accuracy

        if (batch_index + 1) % 10 == 0:
            print(f"Step [{batch_index+1}/{len(loader)}], Loss: {total_loss / (batch_index+1):.4f}")

    avg_loss = total_loss / len(loader)  # Average loss for the epoch
    avg_accuracy = total_accuracy / len(loader.dataset)  # Average accuracy for the epoch
    return avg_loss, avg_accuracy

# Define validation function
def validate(model, loader, loss_fn, accuracy_fn, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # Disable gradient calculation for validation
        for input_x, de_input, pad_mask, sub_mask, label_idx in loader:
            input_x, de_input, pad_mask, sub_mask, label_idx = input_x.to(device), de_input.to(device), pad_mask.to(device), sub_mask.to(device), label_idx.to(device)

            loss, pred_types = loss_fn(input_x, de_input, sub_mask, pad_mask, label_idx)  # Compute loss and predictions
            total_loss += loss.item()
            accuracy, _, _ = accuracy_fn(pred_types, label_idx)  # Calculate accuracy
            total_accuracy += accuracy

    avg_loss = total_loss / len(loader)  # Average loss for the epoch
    avg_accuracy = total_accuracy / len(loader.dataset)  # Average accuracy for the epoch
    return avg_loss, avg_accuracy

# Main function
def main():
    max_accuracy = 0
    best_model = None
    counter = 0
    best_valid_loss = float('inf')

    for epoch in range(epochs):
        start_time = time.time()  # Record start time
        
        # Train for one epoch
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, loss_fn, type_accuracy, device)
        
        # Validate for one epoch
        valid_loss, valid_accuracy = validate(model, valid_loader, loss_fn, type_accuracy, device)
        
        epoch_time = time.time() - start_time  # Calculate epoch duration
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, Valid Accuracy: {valid_accuracy:.4f}, Epoch Time: {epoch_time:.2f}s")
        
        # Save the best model
        if valid_accuracy > max_accuracy:
            max_accuracy = valid_accuracy
            best_model = copy.deepcopy(model)
        
        # Early stopping mechanism
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            counter = 0
        else:
            counter += 1
            if counter >= 10:
                print("Early stopping...")
                break
    
    # Save the best model
    torch.save(best_model.state_dict(), "optimal_qm9s_ircnn_no-augmentation.pt")

if __name__ == "__main__":
    main()
