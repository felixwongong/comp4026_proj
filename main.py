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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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





def train(model, train_loader, optimizer, criterion, epoches):
    """
        Enables CUDA if available to speed up training
    """
    epoch_losses = []

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    for epoch in range(epoches):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            img, lab = data
            img, lab = img.to(device), lab.to(device)

            optimizer.zero_grad()

            outputs = model(img)
            loss = criterion(outputs, lab)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_losses.append(running_loss)
        print(f"Epoch {epoch + 1}, Loss: {running_loss}")

    model_children = list(model.children())


    """
    the following visualization fetched from
    https://debuggercafe.com/visualizing-filters-and-feature-maps-in-convolutional-neural-networks-using-pytorch/
    """
    model_weights = []
    counter = 0
    conv_layers = []

    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)

    plt.figure(figsize=(20, 17))
    for i, filter in enumerate(model_weights[0]):
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter[0, :, :].cpu().detach(), cmap='gray')
        plt.axis('off')
        plt.savefig('./outputs/filter.png')
    plt.show()

    img, lab = None, None
    for i, data in enumerate(train_loader, 0):
        img, lab = data
        img, lab = img.to(device), lab.to(device)
        break

    results = [conv_layers[0](img)]

    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        results.append(conv_layers[i](results[-1]))
    # make a copy of the `results`
    outputs = results

    for num_layer in range(len(outputs)):
        plt.figure(figsize=(30, 30))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print(layer_viz.size())
        for i, filter in enumerate(layer_viz):
            if i == 64:
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter.cpu(), cmap='gray')
            plt.axis("off")
        print(f"Saving layer {num_layer} feature maps...")
        plt.savefig(f"./outputs/layer_{num_layer}.png")
        plt.show()
        plt.close()

    model.eval()

    # plot loss over epoch
    plt.plot(epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # plot activation
    for name, act in activation.items():
        print(act.shape)
        plt.imshow(act[0, 0, :, :].cpu().numpy())
        plt.title(name)
        plt.show()


def test(model, test_loader, std, mean):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img, lab = data
            img, lab = img.to(device), lab.to(device)
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)

            # print("size:", img.size(0))
            figure = plt.figure(figsize=(4, 8))
            num_of_images = min(img.size(0), 32)
            for index in range(1, num_of_images + 1):
                figure.add_subplot(4, 8, index)
                plt.axis('off')
                plt.imshow(tensor_to_ndarray(img[index - 1], std=std, mean=mean))
                plt.title(f'{lab[index - 1].item()}/{predicted[index - 1].item()}',
                          color=("green" if lab[index - 1].item() == predicted[index - 1].item() else "red"))

                if lab[index - 1].item() == 1:
                    if predicted[index - 1].item() == 1:
                        true_positive += 1
                    else:
                        false_negative += 1
                else:
                    if predicted[index - 1].item() == 0:
                        true_negative += 1
                    else:
                        false_positive += 1
            plt.show()

    return true_positive, true_negative, false_positive, false_negative


def main():
    dataset = xrv.datasets.COVID19_Dataset(
        imgpath="covid-chestxray-dataset-master/images",
        csvpath="covid-chestxray-dataset-master/metadata.csv",
        transform=transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            # xrv.datasets.XRayResizer(224),
        ])
    )
    ds, mean, std = preprocessing(dataset)

    train_index, test_index = train_test_split(range(len(ds)), test_size=0.2, random_state=42)
    train_data = Subset(ds, train_index)
    test_data = Subset(ds, test_index)

    train_ds = CovidDataset(train_data)
    test_ds = CovidDataset(test_data)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    resnet = models.resnet50(pretrained=True)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)

    if torch.cuda.is_available():
        resnet = resnet.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 11

    train(resnet, train_loader, optimizer, criterion, num_epochs)
    tp, tn, fp, fn = test(resnet, test_loader, std, mean)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = 2 * tp / (2 * tp + fp + fn)

    # plot confusion matrix in a table
    plt.figure(figsize=(5, 5))
    plt.imshow([[tp, fp], [fn, tn]])
    plt.xticks([0, 1], ["Positive", "Negative"])
    plt.yticks([0, 1], ["Positive", "Negative"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # plot table of accuracy, precision, sensitivity, specificity, f1
    plt.figure(figsize=(5, 5))
    plt.plot([accuracy, precision, sensitivity, specificity, f1])
    plt.xticks([0, 1, 2, 3, 4], ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1"])
    plt.show()

    print("------------")
    print("True Positive:", tp)
    print("True Negative:", tn)
    print("False Positive:", fp)
    print("False Negative:", fn)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("F1:", f1)


if __name__ == "__main__":
    main()
