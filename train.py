import os

from dataset.dataset import VideoDataset
from modeling.model import Model
from torch.utils.data import random_split


def train():
    dataset = VideoDataset(label_file="data/metadata.json")

    input_size = (17, 2)  # 17 keypoints with x and y coordinates
    hidden_size = 128
    num_layers = 2
    num_classes = 2

    # model = ShopliftingModel(
    #     input_size=input_size,
    #     hidden_size=hidden_size,
    #     num_layers=num_layers,
    #     num_classes=num_classes
    # )
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    model = Model(num_classes=num_classes, device="cpu")
    model.train_epochs(10, train_dataset=train_dataset, eval_dataset=test_dataset, eval_every=2)


if __name__ == "__main__":
    train()