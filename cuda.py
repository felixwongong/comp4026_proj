import torch


if __name__ == "__main__":
    print("cuda" if torch.cuda.is_available() else "cpu")