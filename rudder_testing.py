
"""
This file implements the algorithm RUDDER as desceribed here: https://ml-jku.github.io/rudder/#:~:text=Delayed%20rewards%20are%20very%20common,vii

We are first looking to find a theory of the case that RUDDER will actually improve a complex task of ours.
We design a simulation where it is possible to get a "perfect" score, however
We need to get some sequence samples with variance in the returns
"""
import numpy as np
import torch
import tqdm
import os
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from widis_lstm_tools.nn import LSTMLayer
import matplotlib
import matplotlib.pyplot


"""
This is a LSTM that takes in either actions or states and tries to predict two things (total return, and reward contribution)

total return is the total reward that this run will get from here on out (somewhat unknown)
reward contribution is the amount that this specific action or state contributes (in terms of reward value)
"""
class Net(torch.nn.Module):
    """
    We are going to
    """
    def __init__(self, in_features, n_lstm):
        super(Net, self).__init__()

        # This will create an LSTM layer where we will feed the "concatenate"
        self.lstm1 = LSTMLayer(
            in_features=in_features, out_features=n_lstm, inputformat='NLC',
            # cell input: initialize weights to forward inputs with xavier, disable connections to recurrent inputs
            w_ci=(torch.nn.init.xavier_normal_, False),
            # input gate: disable connections to forward inputs, initialize weights to recurrent inputs with xavier
            w_ig=(False, torch.nn.init.xavier_normal_),
            # output gate: disable all connection (=no forget gate) and disable bias
            w_og=False, b_og=False,
            # forget gate: disable all connection (=no forget gate) and disable bias
            w_fg=False, b_fg=False,
            # LSTM output activation is set to identity function
            a_out=lambda x: x
        )

        # After the LSTM layer, we add a fully connected output layer
        self.fc_out = torch.nn.Linear(n_lstm, 1)

    def forward(self, tensor):
        # Process input sequence by LSTM
        lstm_out, *_ = self.lstm1(tensor,
                                  return_all_seq_pos=True  # return predictions for all sequence positions
                                  )
        net_out = self.fc_out(lstm_out)
        return net_out
class SequencesDataset(Dataset):
    def __init__(self, pickle_dir, transform=None):
        """
        Args:
            pickle_dir (string): Directory with all the pickle files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pickle_dir = pickle_dir
        self.pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.pkl')]
        self.transform = transform

    def __len__(self):
        return len(self.pickle_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load the pickle file
        file_path = os.path.join(self.pickle_dir, self.pickle_files[idx])
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            # Convert to torch tensor

        data = data.drop(columns=['state'])
        actions = np.asarray(data['action'],dtype=int)
        actions_onehot = np.identity(55, dtype=np.float32)[actions]

        return torch.tensor(actions_onehot), torch.tensor(data['reward'])
def lossfunction(predictions, rewards):
    returns = rewards.sum(dim=1)
    # Main task: predicting return at last timestep
    main_loss = torch.mean(predictions[:, -1] - returns) ** 2
    # Auxiliary task: predicting final return at every timestep ([..., None] is for correct broadcasting)
    aux_loss = torch.mean(predictions[:, :] - returns[..., None]) ** 2
    # Combine losses
    loss = main_loss + aux_loss * 0.5
    return loss

if __name__ == "__main__":

    # Create Network

    # Create an instance of the dataset
    pickle_dataset = SequencesDataset(pickle_dir='sequences',transform=None)

    # Create a DataLoader
    data_loader = DataLoader(pickle_dataset, batch_size=4, shuffle=True, num_workers=2)

    device = 'mps'
    n_actions = 55
    net = Net(in_features=n_actions, n_lstm=16)
    _ = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    update = 0
    n_updates = 1000
    running_loss = 100
    progressbar = tqdm.tqdm(total=n_updates)
    while update < n_updates:
        for data in data_loader:
            # Get samples
            actions, rewards = data
            actions, rewards = actions.to(device), rewards.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Get outputs for network
            outputs = net(tensor=actions)

            # Calculate loss, do backward pass, and update
            loss = lossfunction(outputs[..., 0], rewards)
            loss.backward()
            running_loss = running_loss * 0.99 + loss * 0.01
            optimizer.step()
            update += 1
            progressbar.set_description(f"Loss: {running_loss:8.4f}")
            progressbar.update(1)

    progressbar.close()
    # Load 2 samples
    a0, r0 = pickle_dataset.__getitem__(3)
    a1, r1 = pickle_dataset.__getitem__(15)

    # Apply our reward redistribution model to the samples
    actions = torch.stack([torch.Tensor(a0).to(device), torch.Tensor(a1).to(device)], dim=0)
    rewards = torch.stack([torch.Tensor(r0).to(device), torch.Tensor(r1).to(device)], dim=0)
    predictions = net(tensor=actions.to(device))[..., 0]

    # Use the differences of predictions as redistributed reward
    redistributed_reward = predictions[:, 1:] - predictions[:, :-1]

    # For the first timestep we will take (0-predictions[:, :1]) as redistributed reward
    redistributed_reward = torch.cat([predictions[:, :1], redistributed_reward], dim=1)
    # Calculate prediction error
    returns = rewards.sum(dim=1)
    predicted_returns = redistributed_reward.sum(dim=1)
    prediction_error = returns - predicted_returns

    # Distribute correction for prediction error equally over all sequence positions
    redistributed_reward += prediction_error[:, None] / redistributed_reward.shape[1]

    redistributed_reward = redistributed_reward.cpu().detach().numpy()
    rr0, rr1 = redistributed_reward[0], redistributed_reward[1]

    fig, axes = plt.subplots(4, 2, figsize=(8, 6), dpi=100)
    axes[1, 0].plot(a0.argmax(-1))
    axes[1, 1].plot(a1.argmax(-1))
    axes[1, 0].xaxis.grid(True)
    axes[1, 1].xaxis.grid(True)
    axes[1, 0].set_title('actions (sample 1)')
    axes[1, 1].set_title('actions (sample 2)')
    axes[1, 0].set_xlabel('time (environment steps)')
    axes[1, 1].set_xlabel('time (environment steps)')

    axes[2, 0].plot(r0)
    axes[2, 1].plot(r1)
    axes[2, 0].xaxis.grid(True)
    axes[2, 1].xaxis.grid(True)
    axes[2, 0].set_title('original rewards (sample 1)')
    axes[2, 1].set_title('original rewards (sample 2)')
    axes[2, 0].set_xlabel('time (environment steps)')
    axes[2, 1].set_xlabel('time (environment steps)')

    axes[3, 0].plot(rr0)
    axes[3, 1].plot(rr1)
    axes[3, 0].xaxis.grid(True)
    axes[3, 1].xaxis.grid(True)
    axes[3, 0].set_title('redistributed rewards (sample 1)')
    axes[3, 1].set_title('redistributed rewards (sample 2)')
    axes[3, 0].set_xlabel('time (environment steps)')
    axes[3, 1].set_xlabel('time (environment steps)')

    fig.tight_layout()
    plt.show()