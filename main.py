import numpy as np
from tqdm import tqdm  # Displays a progress bar

import random
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader, random_split


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = self.pool(self.activation(self.conv3(x)))
        x = x.view(-1, 256 * 8 * 8)
        x = self.dropout1(x)
        x = self.activation(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def train(model, train_loader, val_loader, num_epoch=10, lambda_val=0.1, training_round=1, epoch=0):
    model.train()

    for i in range(num_epoch):
        train_running_loss = []
        for batch, (labelA, labelP) in tqdm(train_loader):
            batch = batch.to(device)
            labelA = labelA.to(device)
            labelP = labelP.to(device)
            optimizer.zero_grad()

            pred = model(batch)

            labelA = labelA.unsqueeze(1).float()
            loss_A = criterion(pred[:, 0], labelA)

            labelP = labelP.unsqueeze(1).float()
            loss_P = criterion(pred[:, 1], labelP)

            if training_round == 1:
                loss = loss_A + lambda_val * loss_P
            elif training_round == 2:
                loss = -1 * lambda_val * loss_P
            else:
                loss = loss_A

            train_running_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        print("Epoch {} train loss:{}".format(i + 1 + epoch, np.mean(train_running_loss)))

        val_running_loss = []
        with torch.no_grad():
            model.eval()
            for batch, (labelA, labelP) in tqdm(val_loader):
                batch = batch.to(device)
                labelA = labelA.to(device)
                labelP = labelP.to(device)

                pred = model(batch)

                labelA = labelA.unsqueeze(1).float()
                loss_A = criterion(pred[:, 0], labelA)

                labelP = labelP.unsqueeze(1).float()
                loss_P = criterion(pred[:, 1], labelP)

                if training_round == 1:
                    loss = loss_A + lambda_val * loss_P
                elif training_round == 2:
                    loss = -1 * lambda_val * loss_P
                else:
                    loss = loss_A

                val_running_loss.append(loss.item())

            print("Epoch {} val loss:{}".format(i + 1 + epoch, np.mean(val_running_loss)))
            model.train()


def evaluate(model, loader, evalA=True):
    model.eval()
    total = 0
    correct = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    with torch.no_grad():
        for batch, (labelA, labelP) in tqdm(loader):
            batch = batch.to(device)
            labelA = labelA.to(device)
            pred = model(batch)
            predA = pred[:, 0]
            predP = pred[:, 1]

            if evalA:
                pred = predA
            else:
                pred = predP

            pred = pred.view(-1, 1)
            predictions = pred.clamp(0, 1).round().view(-1)
            correct += (predictions == labelA.squeeze()).sum().item()
            true_positives += ((predictions >= 1) & (labelA.squeeze() == 1)).sum().item()
            true_negatives += ((predictions <= 0) & (labelA.squeeze() == 0)).sum().item()
            false_positives += ((predictions >= 1) & (labelA.squeeze() == 0)).sum().item()
            false_negatives += ((predictions <= 0) & (labelA.squeeze() == 1)).sum().item()
            total += len(pred)
    acc = correct / total
    print("Evaluation accuracy: {}".format(acc))
    print("True positives: {}, True negatives: {}, False positives: {}, False negatives: {}".format(true_positives,
                                                                                                    true_negatives,
                                                                                                    false_positives,
                                                                                                    false_negatives))
    return acc, true_positives, true_negatives, false_positives, false_negatives


showPlot = False

# Set a random seed for reproducibility
random.seed(13)
num_epoch = 5
scale = 1000


# Load the dataset and train, val, test splits
print("Loading datasets...")
CELEBA_transform = transforms.Compose([
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
])

CELEBA_dataset = datasets.CelebA('.', download=True, transform=CELEBA_transform, target_type="attr")

# Create a random subset of the dataset with 1000 images
subset_indices = random.sample(range(len(CELEBA_dataset)), scale)
CELEBA_subset = Subset(CELEBA_dataset, subset_indices)

train_val_size = int(len(CELEBA_subset) * 0.9)
test_size = len(CELEBA_subset) - train_val_size
celeba_trainval_core, celeba_test_core = random_split(CELEBA_subset, [train_val_size, test_size])
train_val_size = len(celeba_trainval_core)
train_size = int(train_val_size * 0.9)
val_size = train_val_size - train_size
celeba_train_core, celeba_val_core = random_split(celeba_trainval_core, [train_size, val_size])

# Test 1: 9 | 20
# Test 2: 37 | 13
# Test 3: 22 | 20
# Test 4: 14 | 13

# 9  Blond Hair
# 13 Chubby
# 14 Double Chin
# 20 Male
# 22 Mustache
# 37 Wearing Necklace
tests = [(9, 20), (37, 13), (22, 20), (14, 13)]

# 0
# 0.001
# 0.10
# 0.21
# 0.53
# 1
lambdas = [0, 0.001, 0.1, 0.21, 0.53, 1]
for test in tests:
    for lambda_val in lambdas:
        attribute_index = test[0]
        protected_index = test[1]

        # Get only the label corresponding to the chosen attribute
        celeba_train = [(data, (torch.narrow(scores, 0, attribute_index, 1), torch.narrow(scores, 0, protected_index, 1))) for
                        data, scores in celeba_train_core]
        celeba_val = [(data, (torch.narrow(scores, 0, attribute_index, 1), torch.narrow(scores, 0, protected_index, 1))) for
                      data, scores in celeba_val_core]
        celeba_test = [(data, (torch.narrow(scores, 0, attribute_index, 1), torch.narrow(scores, 0, protected_index, 1))) for
                       data, scores in celeba_test_core]

        # Create dataloaders
        trainloader = DataLoader(celeba_train, batch_size=128, shuffle=True)
        valloader = DataLoader(celeba_val, batch_size=128, shuffle=True)
        testloader = DataLoader(celeba_test, batch_size=128, shuffle=True)

        # Get the name of the chosen attribute
        attribute_names = CELEBA_dataset.attr_names
        attribute_name = attribute_names[attribute_index]
        protected_attribute_name = attribute_names[protected_index]

        print(f"Using attribute: {attribute_name}")
        print(f"Protecting attribute: {protected_attribute_name}")

        if showPlot:
            # Create a figure with a 4x4 grid of subplots
            fig, axs = plt.subplots(4, 4)

            # Loop over the first 16 images in the training set
            for i in range(16):
                # Get the i-th image and its label
                image, label = celeba_train[i]
                # Convert the image from a PyTorch tensor to a numpy array
                image = image.numpy().transpose((1, 2, 0))
                # Plot the image in the i-th subplot
                axs[i // 4, i % 4].imshow(image)
                # Set the title of the subplot to the label
                axs[i // 4, i % 4].set_title(str(label[0].item()) + ", " + str(label[1].item()))

            # Adjust the spacing between subplots and show the figure
            plt.subplots_adjust(hspace=0.4)
            plt.show()

        print("Done!")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Network().to(device)
        criterion = nn.MSELoss()  # Use binary cross-entropy loss for binary classification
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        print("Training...")
        for i in range(num_epoch):
            train(model, celeba_train, celeba_val, 1, lambda_val, 1, i)
            if lambda_val > 0:
                train(model, celeba_train, celeba_val, 1, lambda_val, 2, i)
        print("=====")
        # print("Evaluate on validation set...")
        # evaluate(model, celeba_val)
        # print("Evaluate on validation set for protected...")
        # evaluate(model, celeba_val, False)
        # print("Evaluate on test set")
        # evaluate(model, testloader)
        # print("Evaluate on test set for protected...")
        # evaluate(model, testloader, False)

        lambda_val_str = str(lambda_val).replace('.', '')  # Convert to string and remove decimal point
        torch.save(model.state_dict(), "modelA" + str(attribute_index) + "P" + str(protected_index) + "L" + lambda_val_str + ".pt")

