import os
import cv2
import json
import numpy as np

from glob import glob
from datetime import datetime
from collections import deque, defaultdict, Counter

from ultralytics import YOLO


model = YOLO("models/yolov8m.pt")


def extract_frames(video, num_frames):
    
    # Get total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the step size to evenly space frame extraction
    step = total_frames // num_frames
    
    frames = []
    idx = []
    for i in range(num_frames):
        # Set the frame position
        video.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        
        # Read the frame
        ret, frame = video.read()
        if ret:
            frames.append(frame)
            idx.append(i * step)
    
    # Release the video object
    video.release()
    
    return frames, idx

def video_metadata(video_path):

    video_folder = ".".join(video_path.split(".")[:-1])
    os.makedirs(video_folder)
    cap = cv2.VideoCapture(video_path)

    frames, idx = extract_frames(cap, 64)

    for frame_num, frame in zip(idx, frames):
        result = model.predict(frame, classes=[0], verbose=False)[0]
        boxes = result.boxes.xyxy.int().cpu().numpy()
        _areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        # if _areas.size == 0:
        #     frame = cv2.resize(frame, (224, 448))
        if _areas.size != 0:
            _idx = np.argmax(_areas)
            boxes = boxes[_idx.astype(int)]
            frame = frame[boxes[1]:boxes[3], boxes[0]:boxes[2]]

        frame = cv2.resize(frame, (112, 224))

        cv2.imwrite(os.path.join(video_folder, f"{frame_num}.png"), frame)

    return 


# video_list = glob("data/train/*/*.mp4")

# dataset = defaultdict(list)

# # Example usage
# for idx, video_path in enumerate(video_list):
#     video_metadata(video_path)


    # clips = create_video_clips(video_path, clip_length=16, overlap=0.5, target_size=(224, 224))

    # print(f"Number of clips created: {len(clips)}")
    # print(f"Shape of each clip: {clips[0].shape}")


def remove_from_metadata(metadata_path):
    with open(metadata_path, "r") as r:
        metadata = json.load(r)

    ids_to_remove = []
    new_meta = defaultdict(list)
    for video in metadata["video"]:
        video_folder = ".".join(video["path"].split(".")[:-1])
        if not os.path.exists(video_folder):
            ids_to_remove.append(video["id"])
        else:
            new_meta["video"].append(video)

    for md in metadata["metadata"]:
        if md["video_id"] in ids_to_remove:
            continue
        else:
            new_meta["metadata"].append(md)

    new_meta["categories"] = metadata["categories"]

    with open(metadata_path, "w") as w:
        json.dump(new_meta, w)


remove_from_metadata("data/metadata.json")