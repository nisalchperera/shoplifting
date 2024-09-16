import cv2
import torch
import json

import numpy as np

from collections import defaultdict

from torchvision import transforms
from torchvision.datasets import DatasetFolder
from typing import Any, Callable, Optional, Tuple

from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

import joblib
import pandas as pd
import numpy as np


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')  # don't have GPU
    return device


class CustomDataset(Dataset):
    def __init__(self, json_path):
        super().__init__()
        
        self.read_json(json_path)
        
    def read_json(self, path):
        with open(path, "r+") as reader:
            data = json.load(reader)

        self.videos = defaultdict(list)

        for video_data in data["images"]:
            for frame_data in data["annotations"]:
                if video_data["id"] == frame_data["video_id"]:
                    self.videos[video_data["id"]].append(frame_data)

        for video_id, frames in self.videos.items():
            frames = sorted(frames, key=lambda d: d['frame_id'])
            self.videos[video_id] = frames

        self.classes = [cls["category"] for cls in sorted(data["categories"], key=lambda d: d["id"])]

    def num_classes(self):
        return len(self.classes)
        
    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video_id = list(self.videos.keys())[index]
        x = np.array([keypnts["keypoints"] for keypnts in self.videos[video_id]], dtype=np.uint8)
        y = list(set([keypnts["category_id"] - 1 for keypnts in self.videos[video_id]]))[0]
        
        if x.shape[0] > 64:
            idx = np.linspace(0, x.shape[0], 64, dtype=np.uint8, endpoint=False)
            x = x[idx]

        if x.shape[0] < 64:
            pad_width = ((0, 64 - x.shape[0]), (0, 0), (0, 0))
            x = np.pad(x, pad_width, mode='constant', constant_values=0)

        return torch.from_numpy(x), torch.tensor(y)


VIDEO_EXTENSIONS = (".mp4")


def video_loader(path, num_frames):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    interval = total_frames // num_frames
    
    frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 640))
            frames.append(frame)
    
    cap.release()

    return frames


class VideoFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = video_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        num_frames: int = 128
    ):
        super().__init__(
            root,
            loader,
            VIDEO_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples
        # self.skip = skip
        self.num_frames = num_frames
        if not transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0,0,0], [1,1,1])
            ])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path, self.num_frames)
        if self.transform is not None:
            for i in range(len(sample)):
                sample[i] = self.transform(sample[i])
            sample = torch.stack(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target