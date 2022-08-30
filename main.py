import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn import metrics
from torch import optim
from torch.utils.data import DataLoader

from dataset.disprot_dataset import DisprotDataset, Sequence, collate_fn
from dataset.utils import PadRightTo


class Net(nn.Module):
    def __init__(self, in_size, in_features, out_size):
        super().__init__()

        def conv_out_len(layer, length_in):
            return (length_in + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // \
                   layer.stride[0] + 1

        self.conv1 = nn.Conv1d(in_features, 50, kernel_size=21, stride=1, padding=10)
        self.conv2 = nn.Conv1d(self.conv1.out_channels, 30, kernel_size=11, stride=1, padding=5)
        self.conv3 = nn.Conv1d(self.conv2.out_channels, 20, kernel_size=7, stride=1, padding=3)
        self.conv4 = nn.Conv1d(self.conv3.out_channels, 1, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        x = x.flatten(start_dim=1)
        return x


def trim_padding_and_flat(sequences: List[Sequence], pred):
    all_target = np.array([])
    all_trimmed_pred = np.array([])
    for i, seq in enumerate(sequences):
        tmp_pred = pred[i][:len(seq)].cpu().detach().numpy()
        all_target = np.concatenate([all_target, seq.clean_target])
        all_trimmed_pred = np.concatenate([all_trimmed_pred, tmp_pred])
    return all_target, all_trimmed_pred


def batch_auc(sequences: List[Sequence], pred):
    target, pred = trim_padding_and_flat(sequences, pred)
    fpr, tpr, thresholds = metrics.roc_curve(target, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


def plot_auc_and_loss(train_losses, test_losses, test_aucs, epoch, title="AUC and Loss", output_file="default_auc_output"):
    plt.close('all')
    fig, ax1 = plt.subplots(figsize=(8.5, 7.5))

    x_test = np.arange(1, epoch + 2)
    train_losses = train_losses.reshape(-1, 4).mean(axis=1)
    x_train = np.linspace(0, epoch + 1, len(train_losses))

    ax1.plot(x_train, train_losses, color='slategrey', linewidth=1, label='Train Loss')
    ax1.plot(x_test, test_losses, color='dodgerblue', marker='o', linewidth=2, label='Test Loss')
    max_ticks = 22
    ax1.set_xticks(np.linspace(0, epoch + 2, max_ticks, dtype=int))
    ax1.tick_params(axis='y', color='slategrey', labelcolor='slategrey')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ax2.plot(x_test, test_aucs, color='orange', marker='o', linewidth=2, label='Test AUC')
    ax2.tick_params(axis='y', color='orange', labelcolor='orange')
    ax2.set_yticks(np.linspace(0, 1, 11))
    ax2.set_ylabel('AUC')
    # Set the minimum y-axis value to 0.0 and maximum y-axis value to 1.0 (AUC is between 0.0 and 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, which='major', axis='y', linestyle='dotted')

    plt.title(title)
    fig.legend(ncol=1, bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax1.transAxes)

    plt.tight_layout()
    plt.savefig(f"{output_file}.png")
    #plt.show()


# Function that get the results from the model on the test set and plot the ROC curve
def plot_roc_curve(model, data_loader, device, set='Test', outfile="default_roc_curve"):
    model.eval()
    all_output, all_target = np.array([]), np.array([])
    with torch.no_grad():
        for sequences, data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            target, output = trim_padding_and_flat(sequences, output)
            all_target = np.concatenate([all_target, target])
            all_output = np.concatenate([all_output, output])

    fpr, tpr, thresholds = metrics.roc_curve(all_target, all_output, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    r = np.linspace(0, 1, 1000)
    fs = np.mean(np.array(np.meshgrid(r, r)).T.reshape(-1, 2), axis=1).reshape(1000, 1000)
    cs = ax.contour(r[::-1], r, fs, levels=np.linspace(0.1, 1, 10), colors='silver', alpha=0.7, linewidths=1,
                    linestyles='--')
    ax.clabel(cs, inline=True, fmt='%.1f', fontsize=10, manual=[(l, 1 - l) for l in cs.levels[:-1]])
    ax.plot(fpr, tpr, color='orange', linewidth=1, label=f'{set} AUC = %0.3f' % auc)
    ax.plot([0, 1], [0, 1], color='k', linestyle='--')
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    plt.legend(loc='lower right')
    plt.title(f'ROC Curve for {set} Set')
    plt.savefig(f"{outfile}.png")
    #plt.show()


# To get the loss we cut the output and target to the length of the sequence, removing the padding.
# This helps the network to focus on the actual sequence and not the padding.
def get_loss(sequences, output, criterion) -> torch.Tensor:
    loss = 0.0
    # Cycle through the sequences and accumulate the loss, removing the padding
    for i, seq in enumerate(sequences):
        seq_loss = criterion(output[i][:len(seq)], torch.tensor(seq.clean_target, device=device, dtype=torch.float))
        loss += seq_loss
    # Return the average loss over the sequences of the batch
    return loss / len(sequences)


def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    losses = np.array([])
    for batch_idx, (sequences, data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = get_loss(sequences, output, criterion)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        running_loss += loss.item() * data.size(0)
        losses = np.append(losses, [loss.item()])
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{:4d}/{} ({:2.0f}%)] Loss: {:.3f}'.format(
                    epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss, losses


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    test_auc = 0
    with torch.no_grad():
        for sequences, data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += get_loss(sequences, output, criterion).cpu()
            test_auc += batch_auc(sequences, output) * data.size(0)
    test_loss /= len(test_loader)
    test_auc /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, AUC: {:.4f}\n'.format(test_loss, test_auc))
    return test_loss, test_auc


def predict_one_sequence(model, sequence: Sequence, device):
    model.eval()
    data = sequence.data.reshape(1, n_features, -1).to(device)
    output = model(data)
    _, output = trim_padding_and_flat([sequence], output)
    return output


if __name__ == '__main__':
    use_pssm = True
    n_features = 21 if use_pssm else 1
    train_epochs = 50

    # Performance tuning
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.benchmark = True
    ######

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Load the data
    train_data = pd.read_json(os.path.join("data/dataset/disorder_train.json"), orient='records', dtype=False)
    test_data = pd.read_json(os.path.join("data/dataset/disorder_test.json"), orient='records', dtype=False)
    # Defining the dataset
    train_disorder = DisprotDataset(data=train_data, feature_root='data/features',
                                    pssm=use_pssm, transform=PadRightTo(4000), target_transform=PadRightTo(4000))
    test_disorder = DisprotDataset(data=test_data, feature_root='data/features',
                                   pssm=use_pssm, transform=PadRightTo(4000), target_transform=PadRightTo(4000))
    # Defining the dataloader for the training set and the test set
    train_loader = DataLoader(train_disorder, batch_size=50, shuffle=True, num_workers=8, collate_fn=collate_fn,
                              pin_memory=True)
    test_loader = DataLoader(test_disorder, batch_size=50, shuffle=True, num_workers=8, collate_fn=collate_fn,
                             pin_memory=True)

    # Instantiate the model
    net = Net(in_size=4000, in_features=n_features, out_size=4000).to(device)
    # net = nn.DataParallel(net).to(device)

    # Define the loss function and the optimizer
    criterion = nn.MSELoss(reduction='mean')

    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.000005)

    all_train_loss, all_test_loss, all_test_aucs = np.array([]), np.array([]), np.array([])
    for epoch in range(train_epochs):
        _, losses = train(net, train_loader, optimizer, criterion, device, epoch)
        all_train_loss = np.concatenate((all_train_loss, losses))

        test_loss, test_auc = test(net, test_loader, criterion, device)
        all_test_loss = np.append(all_test_loss, [test_loss])
        all_test_aucs = np.append(all_test_aucs, [test_auc])

        if epoch % 10 == 0:
            plot_auc_and_loss(all_train_loss, all_test_loss, all_test_aucs, epoch, output_file=f"epoch{epoch}_auc_and_loss")

    plot_roc_curve(net, test_loader, device, outfile="roc_curve_base")
    plot_roc_curve(net, train_loader, device, set='Train', outfile="roc_curve_train_base")

    sequence: Sequence = test_disorder[0]
    prediction = predict_one_sequence(net, sequence, device)
    for idx, (aa, pred) in enumerate(zip(sequence.sequence, prediction)):
        print(f'{idx + 1:3d}\t{aa}\t{pred:.3f}')
