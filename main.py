import cv2 as cv
import torchxrayvision as xrv
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

def main():
    dataset = xrv.datasets.COVID19_Dataset(imgpath="covid-chestxray-dataset-master/images", csvpath="covid-chestxray-dataset-master/metadata.csv")
    img = dataset[2]["img"][0];

    plt.imshow(img, cmap="gray")
    cv.waitKey(0)
    cv.destroyAllWindows()
    plt.show()

    csv = pd.DataFrame(dataset.csv)
    print(tabulate(csv.head(), headers='keys'))

if __name__ == "__main__":
    main()