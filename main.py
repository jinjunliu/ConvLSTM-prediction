import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dataset import load_data, split_data
from ConvLSTM_pytorch.convlstm import ConvLSTM
import torch.nn as nn


def main():
    # Load data
    sst_data = load_data()  # should return a xr.dataarray of shape (1590, 180, 360)

    # Split data into training and testing sets
    train_data, test_data = split_data(sst_data)  # should return two torch tensors
    train_data = torch.from_numpy(train_data).float() # convert to torch tensor
    test_data = torch.from_numpy(test_data).float() # convert to torch tensor

    # replace missing values with 0
    train_data = torch.nan_to_num(train_data, nan=0.0)
    test_data = torch.nan_to_num(test_data, nan=0.0)

    # Create data loaders
    train_dataset = TensorDataset(train_data)
    test_dataset = TensorDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Create ConvLSTM model
    input_dim = 1  # single channel input
    hidden_dim = [64, 64, 128]  # arbitrary choice, you can change
    kernel_size = (3, 3)  # arbitrary choice, you can change
    num_layers = 3  # arbitrary choice, you can change
    model = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True, return_all_layers=False)

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # mean squared error loss
    optimizer = optim.Adam(model.parameters())  # Adam optimizer

    # Training loop
    for epoch in range(100):  # arbitrary choice, you can change
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[0].unsqueeze(2)  # add an extra dimension for the channel

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

        # print statistics
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    print('Finished Training')

    # Testing loop
    with torch.no_grad():
        for data in test_loader:
            inputs = data[0].unsqueeze(2)  # add an extra dimension for the channel
            predicted, _ = model(inputs)

            # Here you can do something with the predictions, e.g. save them to a file
            print(predicted)

        
def create_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        inout_seq.append(train_seq)
    return torch.stack(inout_seq)

# Create sequences
sequence_length = 24  # 2 years
sst_data = create_sequences(sst_data, sequence_length)

# Split data into training and testing sets
train_data, test_data = split_data(sst_data)  # should return two torch tensors

# Create data loaders
train_dataset = TensorDataset(train_data)
test_dataset = TensorDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)



if __name__ == "__main__":
    main()