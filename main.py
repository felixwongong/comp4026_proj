import cv2 as cv
import numpy as np
import torchxrayvision as xrv
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from PIL import Image

# https://github.com/ieee8023/covid-chestxray-dataset

width = 200
height = 200

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

    for i in range(len(dataset)):
        resized = resize(img)
        ds = dataset[i]
        new_ds.append({
            "idx": ds["idx"],
            "img": resized,
            "lab": ds["lab"][3]
        })
        # plt.imshow(resized, cmap="gray")
        # plt.show()

    print(new_ds[0])

    # print(type(img))
    # plt.show()
    # csv = pd.DataFrame(dataset.csv)
    # print(tabulate(csv.head(), headers='keys'))

if __name__ == "__main__":
    main()