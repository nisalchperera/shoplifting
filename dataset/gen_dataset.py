import cv2
import json
import numpy as np

from glob import glob
from datetime import datetime
from collections import deque, defaultdict, Counter

from ultralytics import YOLO


model = YOLO("models/yolov8m.pt")

def create_video_clips(video_path, clip_length=16, overlap=0.5, target_size=(224, 224)):
    """
    Create video clips of fixed duration with overlapping windows.
    
    Args:
    video_path (str): Path to the input video file.
    clip_length (int): Number of frames in each clip.
    overlap (float): Fraction of overlap between consecutive clips (0 to 1).
    target_size (tuple): Target size for resizing frames (width, height).
    
    Returns:
    list: List of video clips, where each clip is a numpy array of shape (clip_length, height, width, channels).
    """
    cap = cv2.VideoCapture(video_path)
    frame_buffer = deque(maxlen=clip_length)
    clips = []
    
    # Calculate step size based on overlap
    step = int(clip_length * (1 - overlap))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        frame = cv2.resize(frame, target_size)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame_buffer.append(frame)
        frame_count += 1
        
        # Create a clip when buffer is full
        if len(frame_buffer) == clip_length:
            clip = np.array(frame_buffer)
            clips.append(clip)
            
            # Move the buffer forward by the step size
            for _ in range(step):
                if len(frame_buffer) > 0:
                    frame_buffer.popleft()
    
    cap.release()
    return clips

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
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames, idx = extract_frames(cap, 64)

    widths = []
    heights = []

    results = model.track(frames, classes=[0])
    for result in results:
        boxes = result.boxes.xywh.cpu()[:, 2:].tolist()
        _boxes = result.boxes.xywh.cpu().numpy()
        _idx = np.argmax(_boxes[:, 2] - _boxes[:, 0]) * (_boxes[:, 3] - _boxes[:, 1])
        _boxes = _boxes[_idx.astype(int)]
        # _boxes = np.argmax(_boxes, axis=1)
        for w, h in boxes:
            widths.append(w)
            heights.append(h)

    meta = {
        "width": width,
        "height": height,
        "fps": fps,
        "frames_count": length,
        "person_width": sum(widths) / len(widths) if len(widths) else 0,
        "person_height": sum(heights) / len(heights) if len(heights) else 0,
    }

    return meta


video_list = glob("data/train/*/*.mp4")

dataset = defaultdict(list)

# Example usage
for idx, video_path in enumerate(video_list):

    category = video_path.split("/")[-2]

    metadata = video_metadata(video_path)

    video = {
        "id": datetime.timestamp(datetime.now()) + idx,
        "path": video_path,
        "category": category
    }

    metadata["video_id"] = video["id"]

    dataset["video"].append(video)
    dataset["metadata"].append(metadata)

    # clips = create_video_clips(video_path, clip_length=16, overlap=0.5, target_size=(224, 224))

    # print(f"Number of clips created: {len(clips)}")
    # print(f"Shape of each clip: {clips[0].shape}")

frame_counts = [d['frames_count'] for d in dataset["metadata"]]

widths = [d['person_width'] for d in dataset["metadata"]]
heights = [d['person_height'] for d in dataset["metadata"]]

print(f"Min frame count: {min(frame_counts)}")
print(f"Max frame count: {max(frame_counts)}")
print(f"Average frame count: {int(sum(frame_counts) / len(dataset['metadata']))}")
print(f"Counts: {Counter(frame_counts)}")

print(f"Average person width: {sum(widths) / len(widths)}")
print(f"Average person height: {sum(heights) / len(heights)}")

with open("data/metadata.json", "w+") as writer:
    json.dump(dataset, writer)