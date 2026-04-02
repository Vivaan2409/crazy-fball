import cv2
import numpy as np
import gc
from tqdm import tqdm

def read_video(video_path, max_frames=None):
    """
    Read video with memory management
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Reading video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
        print(f"Limiting to {max_frames} frames")
    
    pbar = tqdm(total=total_frames, desc="Loading video")
    
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_count >= max_frames):
            break
            
        frames.append(frame)
        frame_count += 1
        pbar.update(1)
        
        # Clear memory every 50 frames
        if frame_count % 50 == 0:
            gc.collect()
    
    cap.release()
    pbar.close()
    
    print(f"Loaded {len(frames)} frames")
    return frames

def save_video(output_video_frames, output_video_path):
    """Save video with web compatibility"""
    if not output_video_frames:
        raise ValueError("No frames to save")
    
    height, width = output_video_frames[0].shape[:2]
    fps = 24
    
    # Try different codecs
    codecs = [('mp4v', 'mp4v'), ('XVID', 'XVID'), ('MJPG', 'MJPG')]
    writer = None
    
    for codec_name, fourcc_code in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            if writer.isOpened():
                print(f"Using codec: {codec_name}")
                break
            writer.release()
        except:
            continue
    
    if writer is None:
        writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Write frames with memory management
    print(f"Saving {len(output_video_frames)} frames...")
    for i, frame in enumerate(tqdm(output_video_frames, desc="Saving video")):
        writer.write(frame)
        if i % 50 == 0:
            gc.collect()
    
    writer.release()
    print(f"Video saved: {output_video_path}")
    
    # Clear memory
    gc.collect()

# Keep all other functions exactly the same
def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2]-bbox[0]

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)