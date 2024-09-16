import os
import cv2
import torch

import numpy as np

from glob import glob
from ultralytics import YOLO
from collections import defaultdict


yolo = YOLO("yolov8m.pt")
yolo_pose = YOLO("models/yolov8m-pose.pt")
classifier = torch.load("models/shoplifting_detector.pth")

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1

id2cat = {
    0: "Normal",
    1: "Shoplifting"
}

def read_video(path):
    return cv2.VideoCapture(path, cv2.CAP_FFMPEG)


def predict(data):
    data = np.array(data, dtype=np.float32)
    if data.shape[0] < 64:
        pad_width = ((0, 64 - data.shape[0]), (0, 0), (0, 0))
        data = np.pad(data, pad_width, mode='constant', constant_values=0)

    out = classifier.model(torch.from_numpy(data).unsqueeze(0).cuda())
    # out = (out > 0.5).int().item()
    out = torch.argmax(out, dim=1).item()

    return id2cat[out]


def test():
    video_files = glob("data/validate/originals/*.mp4")

    for video_file in video_files:
        video = read_video(video_file)
        width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video_writer = cv2.VideoWriter(video_file.replace("originals", "predicted"), cv2.CAP_FFMPEG, fourcc, fps, (width, height))

        keypoints = defaultdict(list)
        ret = True
        while ret:
            ret, frame = video.read()

            # detections = yolo(frame)
            detections = yolo_pose(frame)

            for detection in detections:
                bboxes = detection.boxes.xyxy.int().cpu().numpy().tolist()

                for i, bbox in enumerate(bboxes):
                    pose = detection.keypoints.xy.float().cpu().numpy().tolist()[i]
                    keypoints[i].append(pose)
                    
                    res = predict(keypoints[i])

                    color = (0, 255, 0) if res == "Normal" else (0, 0, 255)
                    frame = cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
                    frame = cv2.putText(frame, res, tuple(bbox[:2]), font, fontScale, color, 2)

                    if len(keypoints[i]) == 64:
                        keypoints[i] = []

            # cv2.imshow("Videos", frame)
            video_writer.write(frame)

        video_writer.release()
        video.release()

if __name__ == "__main__":
    test()