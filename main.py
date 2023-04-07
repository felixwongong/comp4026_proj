import cv2 as cv
import torchxrayvision as xrv
import matplotlib.pyplot as plt

def main():
    d = xrv.datasets.COVID19_Dataset(imgpath="covid-chestxray-dataset-master/images", csvpath="covid-chestxray-dataset-master/metadata.csv")
    img = d[2]["img"][0];

    plt.imshow(img, cmap="gray")
    cv.waitKey(0)
    cv.destroyAllWindows()
    plt.show()

if __name__ == "__main__":
    main()