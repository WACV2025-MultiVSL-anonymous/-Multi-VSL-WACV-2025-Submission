import os
import cv2
import numpy as np
import re
from tqdm import tqdm

def analyze_videos_in_folder(folder_path):
    video_info = []
    
    # Get the list of files in the folder
    video_files = [filename for filename in os.listdir(folder_path) if filename.endswith(('.mp4', '.avi', '.mkv', '.mov', '.flv'))]
    
    # Use tqdm to show progress
    for filename in tqdm(video_files, desc="Processing videos"):
        file_path = os.path.join(folder_path, filename)
        cap = cv2.VideoCapture(file_path)
        
        if not cap.isOpened():
            continue
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        # Extract the word index (number after the last underscore and before the extension)
        word_order_match = re.search(r'_(\d+)\.\w+$', filename)
        word_order = int(word_order_match.group(1)) if word_order_match else None
        
        video_info.append({
            "filename": filename,
            "resolution": (width, height),
            "duration": duration,
            "frame_count": frame_count,
            "word_order": word_order
        })
        
        cap.release()

    return video_info

def get_video_statistics(video_info):
    durations = [video["duration"] for video in video_info]
    word_orders = [video["word_order"] for video in video_info if video["word_order"] is not None]

    if durations:
        max_duration = max(durations)
        min_duration = min(durations)
        
        longest_video = next(video for video in video_info if video["duration"] == max_duration)
        shortest_video = next(video for video in video_info if video["duration"] == min_duration)
        
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)
        
        resolutions = [video["resolution"] for video in video_info]
        unique_resolutions, counts = np.unique(resolutions, return_counts=True)
        most_common_resolution = unique_resolutions[np.argmax(counts)]
        
        # Calculate statistics for word order (number of videos per word)
        mean_videos_per_word = np.mean(word_orders)
        std_videos_per_word = np.std(word_orders)
        
        stats = {
            "total_videos": len(video_info),
            "longest_video": longest_video["filename"],
            "longest_video_duration": max_duration,
            "shortest_video": shortest_video["filename"],
            "shortest_video_duration": min_duration,
            "mean_duration": mean_duration,
            "std_duration": std_duration,
            "most_common_resolution": most_common_resolution,
            "mean_videos_per_word": mean_videos_per_word,
            "std_videos_per_word": std_videos_per_word
        }
        
        return stats
    else:
        return None

# Example usage: Replace this with your actual folder path
folder_path = '/mnt/disk4/handsign_project/son_data/Yolo_dataset/Blur_video'
video_info = analyze_videos_in_folder(folder_path)
stats = get_video_statistics(video_info)

if stats:
    print(f"Total videos: {stats['total_videos']}")
    print(f"Longest video: {stats['longest_video']} with duration {stats['longest_video_duration']} seconds")
    print(f"Shortest video: {stats['shortest_video']} with duration {stats['shortest_video_duration']} seconds")
    print(f"Mean duration: {stats['mean_duration']} seconds")
    print(f"Standard deviation of durations: {stats['std_duration']} seconds")
    print(f"Most common resolution: {stats['most_common_resolution']}")
    print(f"Mean videos per word: {stats['mean_videos_per_word']}")
    print(f"Standard deviation of videos per word: {stats['std_videos_per_word']}")
else:
    print("No videos found in the folder.")
