import os

from data.dataset import VideoFolder
from modeling.model import ShopliftingModel

def train():
    dataset = VideoFolder("dataset/", num_frames=64)

    input_size = 17 * 2  # 17 keypoints with x and y coordinates
    hidden_size = 128
    num_layers = 2
    num_classes = 2

    model = ShopliftingModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes
    )

    model.train_epochs(10, dataset=dataset)


if __name__ == "__main__":
    train()