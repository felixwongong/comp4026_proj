import numpy as np
import torchxrayvision as xrv
import matplotlib.pyplot as plt
from PIL import Image
import imgaug.augmenters as iaa
import torchvision.models as models
import torchvision.transforms as transforms

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

resnet = models.resnet50(pretrained=True)
resnet.eval()


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


def resize(input):
    pil_img = Image.fromarray(input)
    resized_phil_img = pil_img.resize((width, height), Image.ANTIALIAS)
    return np.array(resized_phil_img)


def preprocess_for_net(image, mean, std):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])
    return preprocess(image).unsqueeze(0)


def tensor_to_ndarray(img, std, mean):
    image_tensor = img.squeeze(0)
    image_tensor = image_tensor * std + mean
    return image_tensor.numpy().squeeze()


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


def main():
    dataset = xrv.datasets.COVID19_Dataset(imgpath="covid-chestxray-dataset-master/images",
                                           csvpath="covid-chestxray-dataset-master/metadata.csv")
    ds = preprocessing(dataset)
    print(ds)


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

        ds = dataset[i]
        new_ds.append({
            "idx": ds["idx"],
            "img": normalized,
            "lab": ds["lab"][3]
        })

        if ds["lab"][3] == 1:
            covid_cnt += 1
    non_covid_cnt = len(dataset) - covid_cnt
    dataset = None  # to gc
    # Data augmentation
    augment_ds = []
    offset = covid_cnt - non_covid_cnt
    ratio = -(covid_cnt // -non_covid_cnt)
    print(f"ratio is {ratio}")
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
    return augment_ds


if __name__ == "__main__":
    main()
