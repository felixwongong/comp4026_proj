import numpy as np
import torch
import torchxrayvision as xrv
import matplotlib.pyplot as plt
from PIL import Image
import imgaug.augmenters as iaa
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, Dataset
import torch.nn as nn
import torch.optim as optim

# https://github.com/ieee8023/covid-chestxray-dataset

width = height = 224

augment_seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(
        rotate=(-20, 20),
        scale=(0.8, 1.2),
    ),
    iaa.Multiply((0.8, 1.2)),
    iaa.GaussianBlur(sigma=(0, 2.0)),
], random_order=True)




def to_255(img):
    return (img * 255)


def augment_image(image, n_augments=1):
    images_aug = [image] * n_augments
    images_aug = augment_seq.augment_images(images_aug)
    return images_aug


'''
Normalized image pixel to 0-1, original is -1024 to 1024
'''


def norm_pixel_values(img):
    return np.clip((img / 1024 + 1.) * 0.5, 0, 1)


def grayscale_to_rgb(p):
    return p.expand(3, -1, -1)


def resize(input):
    pil_img = Image.fromarray(input)
    resized_phil_img = pil_img.resize((width, height), Image.ANTIALIAS)
    return np.array(resized_phil_img)


def preprocess_for_net(image, mean, std):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
        grayscale_to_rgb,
    ])
    return preprocess(image)


def tensor_to_ndarray(img_tensor, std, mean):
    img = img_tensor.cpu().numpy().transpose((1, 2, 0))
    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    return img



class CovidDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        data = self.data_list[index]
        img = data["img"]
        lab = torch.tensor(data["lab"], dtype=torch.long)

        return img, lab

    def __len__(self):
        return len(self.data_list)


def main():
    dataset = xrv.datasets.COVID19_Dataset(imgpath="covid-chestxray-dataset-master/images",
                                           csvpath="covid-chestxray-dataset-master/metadata.csv")
    ds, mean, std = preprocessing(dataset)

    train_index, test_index = train_test_split(range(len(ds)), test_size=0.2, random_state=42)
    train_data = Subset(ds, train_index)
    test_data = Subset(ds, test_index)

    train_ds = CovidDataset(train_data.dataset)
    test_ds = CovidDataset(test_data.dataset)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    resnet = models.resnet50(pretrained=True)

    resnet.fc = nn.Linear(resnet.fc.in_features, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            img, lab = data
            img, lab = img.to(device), lab.to(device)

            optimizer.zero_grad()

            outputs = resnet(img)
            loss = criterion(outputs, lab)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss}")

    resnet.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img, lab = data
            img, lab = img.to(device), lab.to(device)
            outputs = resnet(img)
            _, predicted = torch.max(outputs.data, 1)

            for j in range(img.size(0)):
                # Display the image
                plt.imshow(tensor_to_ndarray(img[j], std=std, mean=mean))
                plt.title(f'Label: {lab[j].item()}, Predicted: {predicted[j].item()}')
                plt.axis('off')
                plt.show()


'''
return list of dictionary
{
    "lab": 1 or 0 for covid or non-covid case,
    "img": tenor image object for putting into network
}
'''
def preprocessing(dataset):
    new_ds = []
    covid_cnt = 0
    # Labelling
    for i in range(len(dataset)):
        img = dataset[i]["img"][0]
        normalized = norm_pixel_values(img)
        print(i)
        ds = dataset[i]
        new_ds.append({
            "idx": ds["idx"],
            "img": normalized,
            "lab": ds["lab"][3].astype(np.int8)
        })

        if ds["lab"][3] == 1:
            covid_cnt += 1
    non_covid_cnt = len(dataset) - covid_cnt
    dataset = None  # to gc
    # Data augmentation
    augment_ds = []
    offset = covid_cnt - non_covid_cnt
    ratio = -(covid_cnt // -non_covid_cnt)
    covid_cnt = non_covid_cnt = 0  # for recalculation
    for i in range(len(new_ds)):
        augmented = []
        if new_ds[i]["lab"] == 1:
            augmented = augment_image(new_ds[i]["img"])
            covid_cnt += len(augmented)
        else:
            if offset > 0:
                augmented = augment_image(new_ds[i]["img"], ratio)
                offset -= (len(augmented) - 1)
            else:
                augmented = augment_image(new_ds[i]["img"])
            non_covid_cnt += len(augmented)

        for j in range(len(augmented)):
            augmented[j] = {
                "lab": new_ds[i]["lab"],
                "img": augmented[j]
            }

        for new_augment in augmented:
            augment_ds.append(new_augment)
    augmented = None
    print(f"Augmented: Covid: {covid_cnt}, others: {non_covid_cnt}")
    # resize to ensure all shape are the same for computing mean & std
    for i in range(len(augment_ds)):
        augment_ds[i]["img"] = resize(augment_ds[i]["img"])
    all_images = [d["img"] for d in augment_ds]
    stack = np.stack(all_images, axis=0)
    mean = np.mean(stack)
    std = np.std(stack)
    print(f"{mean}, {std}")
    for i in range(len(augment_ds)):
        augment_ds[i]["img"] = preprocess_for_net(augment_ds[i]["img"], mean, std)
    # tmp = tensor_to_ndarray(augment_ds[0]["img"], mean=mean, std=std)
    #
    # # Display the image using plt.imshow()
    # plt.imshow(tmp, cmap='gray')
    # plt.show()
    print("Preprocessing end")
    return augment_ds, mean, std


if __name__ == "__main__":
    main()
