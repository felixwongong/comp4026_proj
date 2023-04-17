import numpy as np
import torch
import torch.utils.data
import torchvision
import torchxrayvision as xrv
import matplotlib.pyplot as plt
from PIL import Image
import imgaug.augmenters as iaa
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

# https://github.com/ieee8023/covid-chestxray-dataset

width = height = 224

model = models.resnet50(pretrained=True)

# original label
'''
00 = {str} 'Aspergillosis'
01 = {str} 'Aspiration'
02 = {str} 'Bacterial'
03 = {str} 'COVID-19'
04 = {str} 'Chlamydophila'
05 = {str} 'Fungal'
06 = {str} 'H1N1'
07 = {str} 'Herpes '
08 = {str} 'Influenza'
09 = {str} 'Klebsiella'
10 = {str} 'Legionella'
11 = {str} 'Lipoid'
12 = {str} 'MERS-CoV'
13 = {str} 'MRSA'
14 = {str} 'Mycoplasma'
15 = {str} 'No Finding'
16 = {str} 'Nocardia'
17 = {str} 'Pneumocystis'
18 = {str} 'Pneumonia'
19 = {str} 'SARS'
20 = {str} 'Staphylococcus'
21 = {str} 'Streptococcus'
22 = {str} 'Tuberculosis'
23 = {str} 'Varicella'
24 = {str} 'Viral'
'''

labels_map = {
    0: 'Aspergillosis',
    1: 'Aspiration',
    2: 'Bacterial',
    3: 'COVID-19',
    4: 'Chlamydophila',
    5: 'Fungal',
    6: 'H1N1',
    7: 'Herpes',
    8: 'Influenza',
    9: 'Klebsiella',
    10: 'Legionella',
    11: 'Lipoid',
    12: 'MERS-CoV',
    13: 'MRSA',
    14: 'Mycoplasma',
    15: 'No Finding',
    16: 'Nocardia',
    17: 'Pneumocystis',
    18: 'Pneumonia',
    19: 'SARS',
    20: 'Staphylococcus',
    21: 'Streptococcus',
    22: 'Tuberculosis',
    23: 'Varicella',
    24: 'Viral'
}

def grayscale_to_rgb(p):
    print(p)
    return p
    # return [p[0], p[0], p[0]]

transform = torchvision.transforms.Compose(
    [
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ]
)

dataset = xrv.datasets.COVID19_Dataset(
    imgpath="covid-chestxray-dataset-master/images",
    csvpath="covid-chestxray-dataset-master/metadata.csv",
    transform=transform,
)

# convert dataset.labels to 0 and 1
# 0 = no covid
# # 1 = covid
# def preprocess_dataset(dataset):
#     new_dataset = []
#     for i in range(len(dataset)):
#         label = dataset.labels[i][3]
#         new_dataset.append((dataset[i]['img'], label))
#
#     return dataset


# dataset = preprocess_dataset(dataset)

print(f'Number of images: {len(dataset)}')


def visualize_dataset(dataset):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        item = dataset[sample_idx]
        img = item['img']
        label = np.where(item['lab'] == 1)[0][0]
        img = np.transpose(img, (1, 2, 0))
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


visualize_dataset(dataset)

num_of_classes = 2


def split_dataset(dataset, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    training_data, testing_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    training_data = training_data  # type: torch.utils.data.Subset
    testing_data = testing_data  # type: torch.utils.data.Subset

    print(training_data.dataset)

    training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=1, shuffle=True)
    testing_data_loader = torch.utils.data.DataLoader(testing_data, batch_size=1, shuffle=True)

    return training_data_loader, testing_data_loader


print(f'Number of images: {len(dataset)}')

training_data_loader, testing_data_loader = split_dataset(dataset, 0.8)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    for item in dataloader:
        # # Compute prediction error
        x = item['img']
        y = item['lab']

        print(x)
        print(y)

        pred = model(x)
        loss = loss_fn(pred, y)
        #
        # # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return model


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
train_loop(training_data_loader, model, loss_fn, optimizer)

# print(f'Accuracy of the network on the 10000 test images: {evaluate(model, testing_data_loader)} %')
