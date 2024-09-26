import os
import cv2
import math
import torch

import numpy as np

from glob import glob
from datetime import datetime

from ultralytics import YOLO

from dataset.dataset import VideoTransform


yolo = YOLO("models/yolov8m.pt")
classifier = torch.load("models/shoplifting_detector_3d_cnn.pth")
video_transform = VideoTransform()

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


def test(video_files=None):
    if not video_files:
        video_files = glob("data/validate/originals/*.mp4")
        
    filenames = []
    for video_file in video_files:
        print(f"Processing {video_file}")

        start = datetime.now()
        video = read_video(video_file)
        width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        h = int(video.get(cv2.CAP_PROP_FOURCC))
        codec = chr(h&0xff) + chr((h>>8)&0xff) + chr((h>>16)&0xff) + chr((h>>24)&0xff)
        
        skip_frames = math.floor(fps / 10)

        filename = video_file.replace("originals", "predicted")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video_writer = cv2.VideoWriter(filename, cv2.CAP_FFMPEG, fourcc, fps, (width, height))

        frames = []
        ret = True
        frame_num = 0
        i = 0
        while ret:
            ret, frame = video.read()
            _frame = frame

            if frame is not None or img.size != 0 and frame_num % skip_frames == 0:
                detection = yolo.predict(frame, classes=[0], verbose=False)[0]

                for boxes in detection.boxes.xyxy.int().cpu().numpy().tolist():
                # boxes = 
                    # _areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    # if _areas.size == 0:
                    #     frame = cv2.resize(frame, (224, 448))
                    # if _areas.size != 0:
                    #     _idx = np.argmax(_areas)
                    # boxes = boxes[_idx.astype(int)].tolist()
                    _frame = _frame[boxes[1]:boxes[3], boxes[0]:boxes[2]]
                    

                    _frame = cv2.resize(_frame, (112, 224))

                    frames.append(_frame)

                    if i >= 64:
                        frames.pop(0)


                    inputs = video_transform(frames.copy()).unsqueeze(0).cuda()

                    result = np.argmax(classifier(inputs).detach().cpu().numpy()).item()
                    result = id2cat[result]
                    color = (0, 255, 0) if result == "Normal" else (0, 0, 255)

                    cv2.rectangle(frame, tuple(boxes[:2]), tuple(boxes[2:]), color, 2)
                    cv2.putText(frame, result, tuple(boxes[:2]), font, fontScale, color, 2)

                # cv2.imshow("Videos", frame)

                # for _ in range(i + 1, 64):
                #     del frames[i]

                i = i + 1

            video_writer.write(frame)
            frame_num = frame_num + 1

        video_writer.release()
        video.release()
        print(f"Time to process: {(datetime.now() - start).total_seconds()}")
        filenames.append(filename)
        
    return filenames

if __name__ == "__main__":
    test()
