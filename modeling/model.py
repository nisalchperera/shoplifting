import torch

from torch.nn import Module
from ultralytics import YOLO

from inception_time.model import InceptionTime


class ShopliftingModel(Module):
    def __init__(self, model_version=10, model_size="m", **kwargs) -> None:
        super().__init__(**kwargs)

        self.pose = YOLO(f"yolov{model_version}{model_size}-pose.yaml")
        self.inceptiontime = InceptionTime()

    def forward(self, image):

        x = self.pose(image)

        ## preprocess x

        y = self.inceptiontime.predict(x)

        return y
    
    def train(self, images, y, learning_rate, batch_size, epochs, verbose=True, save_dir="models/"):
        x = self.pose(images)

        ## preprocess x

        model = self.inceptiontime(x=x, y=y, learning_rate=learning_rate, batch_size=batch_size, epochs=epochs, verbose=verbose)
        