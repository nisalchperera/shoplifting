import os
import cv2
import json
import torch

import numpy as np

from ultralytics import YOLO
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

model = YOLO("models/yolov8m.pt")

def extract_frames(video, num_frames):
    
    # Get total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the step size to evenly space frame extraction
    step = total_frames // num_frames
    
    frames = []
    for i in range(num_frames):
        # Set the frame position
        video.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        
        # Read the frame
        ret, frame = video.read()
        if ret:
            result = model.predict(frame, classes=[0])[0]
            boxes = result.boxes.xyxy.int().cpu().numpy()
            _areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            # if _areas.size == 0:
            #     frame = cv2.resize(frame, (224, 448))
            if _areas.size != 0:
                _idx = np.argmax(_areas)
                boxes = boxes[_idx.astype(int)]
                frame = frame[boxes[1]:boxes[3], boxes[0]:boxes[2]]

            frame = cv2.resize(frame, (224, 448))
            frames.append(frame)
    
    # Release the video object
    video.release()
    
    return frames

class VideoDataset(Dataset):
    def __init__(self, label_file, video_dir="./", num_frames=64, transform=None):
        self.video_dir = video_dir
        self.num_frames = num_frames

        if transform:
            self.transform = transform
        else:
            self.transform = VideoTransform()
        
        # Read labels from file
        with open(label_file, 'r') as f:
            metadata = json.load(f)

        self.load_data(metadata)

    def load_data(self, data):
        self.video_paths = [os.path.join(self.video_dir, d["path"]) for d in data["video"]]
        self.labels = [d["category"] for d in data["video"]]

        self.label2id = {label: i for i, label in enumerate(data["categories"])}
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video clip
        cap = cv2.VideoCapture(video_path)
        frames = extract_frames(cap, self.num_frames)
        
        # If video is shorter than clip_length, loop the last frame
        while len(frames) < self.num_frames:
            frames.extend(frames[-4:])

        # make sure frames has num_frames.
        if len(frames) > self.num_frames:
            frames = frames[:self.num_frames]
        
        # Apply transformations
        if self.transform:
            frames = self.transform(frames)

        # Convert to numpy array
        if isinstance(frames, list):
            frames = np.array(frames)
        
        return frames, self.label2id[label]

class VideoTransform:
    def __init__(self, size=(448, 224)):
        self.size = size
    
    def __call__(self, clip):
        # Resize
        clip = [cv2.resize(frame, self.size) for frame in clip]
        clip = np.array(clip)
        
        # Convert to torch tensor
        clip = torch.from_numpy(clip).permute(0, 3, 1, 2).float()
        
        # Normalize
        clip = clip / 255.0

        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
        
        for i in range(clip.shape[0]):
            clip[i] = transform(clip[i])
        
        clip = clip.permute(1, 0, 2, 3)

        return clip

def create_video_dataloader(video_dir, label_file, batch_size=16, num_frames=64, num_workers=4, shuffle=True):
    # Define transformations
    transform = VideoTransform(size=(448, 224))
    
    # Create dataset
    dataset = VideoDataset(video_dir, label_file, num_frames, transform)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                            num_workers=num_workers, pin_memory=True)
    
    return dataloader