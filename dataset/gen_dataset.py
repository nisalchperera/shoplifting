import os
import cv2
import json

from datetime import datetime
from glob import glob
from ultralytics import YOLO

pose = YOLO(f"models/yolov8m-pose.pt")

classes = os.listdir("data/train")

def video_loader(path, num_frames):
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # interval = math.ceil(total_frames / num_frames)
    interval = total_frames // num_frames
    
    frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            if frame.shape[0] > frame.shape[1]:
                height = 640
                width = int(frame.shape[1] / frame.shape[0] * height)
            else:
                width = 640
                height = int(frame.shape[0] / frame.shape[1] * width)
            frame = cv2.resize(frame, (height, width))
            frames.append(frame)
    
    cap.release()

    return frames

annotations = {
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 1,
            "category": "Normal"
        },
        {
            "id": 2,
            "category": "Shoplifting"
        }
    ]
}

category2id = {
    "Normal": 1,
    "Shoplifting": 2
}

for cls in classes:
    if os.path.isdir(os.path.join("data/train", cls)):
        video_files = list(glob(f"data/train/{cls}/*.mp4"))

        for i, video_file in enumerate(video_files):
            frames = video_loader(video_file, 64)

            images = []
            video_id = datetime.timestamp(datetime.now()) + i
            annotations["images"].append({
                    "id": video_id,
                    "video_file": video_file,
                })

            for frame_no, frame in enumerate(frames):
                res = pose(frame, device="cuda")
                x = res[0].keypoints.xy

                keypoints = []

                for j in range(x.shape[0]):
                    _x = x[j]
                    if _x.sum() == 0:
                        continue
                    else:
                        keypoints.append({
                            "id": datetime.timestamp(datetime.now()) + i + j,
                            "video_id": video_id,
                            "frame_id": frame_no,
                            "keypoints": _x.int().cpu().numpy().tolist(),
                            "bbox": res[0].boxes.xyxy[j].int().cpu().numpy().tolist(),
                            "category_id": category2id[cls]
                        })
                annotations["annotations"].extend(keypoints)
                

                print(f"Frame {frame_no} of video {video_file} is completed")

with open("data/annotations.json", "w+") as r:
    json.dump(annotations, r)