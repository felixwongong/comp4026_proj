import numpy as np
import torchxrayvision as xrv
import matplotlib.pyplot as plt
from PIL import Image
import imgaug.augmenters as iaa

# https://github.com/ieee8023/covid-chestxray-dataset

augment_seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(
        rotate=(-20, 20),
        scale=(0.8, 1.2),
    ),
    iaa.Multiply((0.8, 1.2)),
    iaa.GaussianBlur(sigma=(0, 2.0)),
], random_order=True)



def augment_image(image, n_augments = 1):
    images_aug = [image] * n_augments
    images_aug = augment_seq.augment_images(images_aug)
    return images_aug

def norm_pixel_values(img):
    return np.clip((img / 1024 + 1.) * 0.5 * 255, 0, 255).astype(np.uint8)

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
    dataset = xrv.datasets.COVID19_Dataset(imgpath="covid-chestxray-dataset-master/images", csvpath="covid-chestxray-dataset-master/metadata.csv")

    new_ds = []
    covid_cnt = 0

    # Labelling
    for i in range(len(dataset)):
        img = dataset[i]["img"][0]
        resized = norm_pixel_values(img)

        ds = dataset[i]
        new_ds.append({
            "idx": ds["idx"],
            "img": resized,
            "lab": ds["lab"][3]
        })

        if ds["lab"][3] == 1:
            covid_cnt += 1

    non_covid_cnt = len(dataset) - covid_cnt
    dataset = None   # to gc

    # Data augmentation
    augment_ds = []
    offset = covid_cnt - non_covid_cnt
    ratio = -(covid_cnt // -non_covid_cnt)
    print(f"ratio is {ratio}")

    covid_cnt = non_covid_cnt = 0       # for recalculation
    for i in range(len(new_ds)):
        augmented = []
        if new_ds[i]["lab"] == 1:
            augmented = augment_image(new_ds[i]["img"])
            covid_cnt += len(augmented)

            # test augmented images
            if i == 1:
                plt.imshow(new_ds[i]["img"], cmap="gray")
                plt.show()
                for i in range(len(augmented)):
                    plt.imshow(augmented[i], cmap="gray")
                    plt.show()

        else:
            if offset > 0:
                augmented = augment_image(new_ds[i]["img"], ratio)
                offset -= (len(augmented) - 1)
            else:
                augmented = augment_image(new_ds[i]["img"])
            non_covid_cnt += len(augmented)
        augment_ds.append(augmented)

    print(f"Augmented: Covid: {covid_cnt}, others: {non_covid_cnt}")

if __name__ == "__main__":
    main()