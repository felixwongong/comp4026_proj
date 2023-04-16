import cv2 as cv
import numpy as np
import torchxrayvision as xrv
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from PIL import Image
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

# https://github.com/ieee8023/covid-chestxray-dataset

width = 200
height = 200
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
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    images_aug = [image] * n_augments
    images_aug = augment_seq.augment_images(images_aug)
    return images_aug


def resize(input):
    pil_img = Image.fromarray(input)
    resized_phil_img = pil_img.resize((width, height), Image.ANTIALIAS)
    return np.array(resized_phil_img)



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
    img = dataset[2]["img"][0]

    new_ds = []

    covid_cnt = 0

    # Labelling
    for i in range(len(dataset)):
        resized = resize(img)
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
                cv.imshow("original", new_ds[i]["img"])
                cv.waitKey(0)
                cv.destroyAllWindows()
                for i in range(len(augmented)):
                    cv.imshow(f"augmented-{i}", augmented[i])
                    cv.waitKey(0)
                    cv.destroyAllWindows()
        else:
            if offset > 0:
                augmented = augment_image(new_ds[i]["img"], ratio)
                offset -= (len(augmented) - 1)
            else:
                augmented = augment_image(new_ds[i]["img"])
            non_covid_cnt += len(augmented)
        augment_ds.append(augmented)

    print(f"Covid: {covid_cnt}, others: {non_covid_cnt}")
    # print(type(img))
    # plt.show()
    # csv = pd.DataFrame(dataset.csv)
    # print(tabulate(csv.head(), headers='keys'))

if __name__ == "__main__":
    main()