import gc
import os
import random
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn import metrics
from torch import optim
from torch.utils.data import DataLoader
import timeit

from dataset.disprot_dataset import DisprotDataset, Sequence, collate_fn
from dataset.utils import PadRightTo
from net import Net

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

def save_auc_and_loss(train_losses, test_losses, test_aucs, epoch, outfile, outfile_train):
    x_test = np.arange(1, epoch + 2)
    train_losses = train_losses.reshape(-1, 4).mean(axis=1)
    x_train = np.linspace(0, epoch + 1, len(train_losses))
    with open(outfile,'w') as outfmt:
        for i in range(len(test_losses)):
            outfmt.write(f"{x_test[i]}\t{test_losses[i]}\t{test_aucs[i]}\n")
    with open(outfile_train, 'w') as outfmt:
        for i in range(len(x_train)):
            outfmt.write(f"{x_train[i]}\t{train_losses[i]}\n")

def plot_auc_and_loss(train_losses, test_losses, test_aucs, epoch, title="AUC and Loss"):
    plt.close('all')
    fig, ax1 = plt.subplots(figsize=(8.5, 7.5))
    x_test = np.arange(1, epoch + 2)
    train_losses = train_losses.reshape(-1, 4).mean(axis=1)
    x_train = np.linspace(0, epoch + 1, len(train_losses))
    ax1.plot(x_train, train_losses, color='slategrey', linewidth=1, label='Train Loss')
    if len(test_losses) == 0:
        ax1.plot(x_test, test_losses, color='dodgerblue', marker='o', linewidth=2, label='Test Loss')
    max_ticks = 22
    ax1.set_xticks(np.linspace(0, epoch + 2, max_ticks, dtype=int))
    ax1.tick_params(axis='y', color='slategrey', labelcolor='slategrey')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_yscale('log')
    ax2 = ax1.twinx()
    if len(test_aucs) == 0:
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
    plt.savefig('AUC_and_LOSS.png')


# Function that get the results from the model on the test set and plot the ROC curve
def plot_roc_curve(model, data_loader, device, set='Test'):
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
    plt.savefig(f"ROC_CURVE_{set}.png")
    return auc


# To get the loss we cut the output and target to the length of the sequence, removing the padding.
# This helps the network to focus on the actual sequence and not the padding.
def get_loss(sequences, output, criterion, device) -> torch.Tensor:
    loss = 0.0
    # Cycle through the sequences and accumulate the loss, removing the padding
    for i, seq in enumerate(sequences):
        seq_loss = criterion(
            output[i][:len(seq)],
            torch.tensor(seq.clean_target, device=device, dtype=torch.float)
        )
        loss += seq_loss
    # Return the average loss over the sequences of the batch
    return loss / len(sequences)

def train(model, train_loader, optimizer, criterion, device, epoch):
    t0 = timeit.default_timer()
    model.train()
    t1 = timeit.default_timer()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    t2 = timeit.default_timer()
    losses = np.array([])
    print(f"    Train: {t1 - t0}")
    print(f"    Zero grad: {t2 - t1}")
    for batch_idx, (sequences, data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = get_loss(sequences, output, criterion, device)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        running_loss += loss.item() * data.size(0)
        losses = np.append(losses, [loss.item()])
        if batch_idx % 10 == 0:
            print('    Train Epoch: {} [{:4d}/{} ({:2.0f}%)] Loss: {:.3f}'.format(
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
            test_loss += get_loss(sequences, output, criterion, device).cpu()
            test_auc += batch_auc(sequences, output) * data.size(0)
    test_loss /= len(test_loader)
    test_auc /= len(test_loader.dataset)
    print('    Test set: Average loss: {:.4f}, AUC: {:.4f}\n'.format(test_loss, test_auc))
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
    train_epochs = 100

    # Performance tuning
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.benchmark = True
    ######

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Load the data
    train_data = pd.read_json(
        os.path.join(
            "data",
            "dataset",
            "disorder_train.json"
        ),
        orient='records',
        dtype=False
    )
    test_data = pd.read_json(
        os.path.join(
            "data",
            "dataset",
            "disorder_test.json"
        ),
        orient='records',
        dtype=False
    )
    # Defining the dataset
    train_disorder = DisprotDataset(
        data=train_data,
        feature_root='data/features',
        pssm=use_pssm,
        transform=PadRightTo(4000),
        target_transform=PadRightTo(4000)
    )
    test_disorder = DisprotDataset(
        data=test_data,
        feature_root='data/features',
        pssm=use_pssm,
        transform=PadRightTo(4000),
        target_transform=PadRightTo(4000)
    )
    # Defining the dataloader for the training set and the test set
    train_loader = DataLoader(
        train_disorder,
        batch_size=50,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
        pin_memory_device=device.type
    )
    test_loader = DataLoader(
        test_disorder,
        batch_size=50,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
        pin_memory_device=device.type
    )

    train_aucs = []
    test_aucs = []
    seed = int(timeit.default_timer())
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Instantiate the model
    net = Net(in_size=4000, in_features=n_features, out_size=4000).to(device)
    net = nn.DataParallel(net).to(device)

    # Define the loss function and the optimizer
    criterion = nn.MSELoss(reduction='mean').cuda()

    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.000005)

    all_train_loss, all_test_loss, all_test_aucs = np.array([]), np.array([]), np.array([])
    for epoch in range(train_epochs):
        print(f"Epoch {epoch+1}")
        t0 = timeit.default_timer()
        _, losses = train(net, train_loader, optimizer, criterion, device, epoch)
        t1 = timeit.default_timer()
        print(f"  Train time: {t1-t0}")
        all_train_loss = np.concatenate((all_train_loss, losses))
        t2 = timeit.default_timer()
        print(f"  Concat: {t2-t1}")
        test_loss, test_auc = test(net, test_loader, criterion, device)
        t3 = timeit.default_timer()
        print(f"  Test: {t3-t2}")
        all_test_loss = np.append(all_test_loss, [test_loss])
        all_test_aucs = np.append(all_test_aucs, [test_auc])
        t4 = timeit.default_timer()
        print(f"  Appends: {t4-t3}")

        # if epoch % 10 == 0:
        #     plot_auc_and_loss(all_train_loss, all_test_loss, all_test_aucs, epoch)

    plot_auc_and_loss(all_train_loss, all_test_loss, all_test_aucs, epoch)
    auc_test = plot_roc_curve(net, test_loader, device)
    auc_train = plot_roc_curve(net, train_loader, device, set='Train')
    save_auc_and_loss(
        all_train_loss,
        all_test_loss,
        all_test_aucs,
        epoch,
        "loss_and_auc_data.txt",
        "train_loss_and_auc_data.txt"
    )
    train_aucs.append(auc_train)
    test_aucs.append(auc_test)
    net = None
    gc.collect()
    torch.cuda.empty_cache()

    with open("auc_summary.txt", 'a', encoding='utf-8') as f_out:
        for tr, te in zip(train_aucs, test_aucs):
            f_out.write(f"{tr}\t{te}\n")
    # sequence: Sequence = test_disorder[0]
    # prediction = predict_one_sequence(net, sequence, device)
    # for idx, (aa, pred) in enumerate(zip(sequence.sequence, prediction)):
    #     print(f'{idx + 1:3d}\t{aa}\t{pred:.3f}')
