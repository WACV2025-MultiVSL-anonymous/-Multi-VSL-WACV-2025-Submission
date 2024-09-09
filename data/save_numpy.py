import numpy as np
import pandas as pd
import av
from tqdm.auto import tqdm
import concurrent.futures
import os
import cv2
import math

# data = pd.read_csv("labels.csv")
data = os.listdir("/mnt/disk3/anhnct/Hand-Sign-Recognition/data/VN_SIGN/np_videos")


def crop_hand(frame,keypoints,WRIST_DELTA,SHOULDER_DIST_EPSILON,
              clip_len,missing_wrists_left,missing_wrists_right):
    left_wrist_index = 9
    left_elbow_index = 7
    right_wrist_index = 10
    right_elbow_index = 8

    # Crop out both wrists and apply transform
    left_wrist = keypoints[0:2, left_wrist_index]
    left_elbow = keypoints[0:2, left_elbow_index]

    left_hand_center = left_wrist + WRIST_DELTA * (left_wrist - left_elbow)
    left_hand_center_x = left_hand_center[0]
    left_hand_center_y = left_hand_center[1]
    shoulder_dist = np.linalg.norm(keypoints[0:2, 5] - keypoints[0:2, 6]) * SHOULDER_DIST_EPSILON
    left_hand_xmin = max(0, int(left_hand_center_x - shoulder_dist // 2))
    left_hand_xmax = min(frame.shape[1], int(left_hand_center_x + shoulder_dist // 2))
    left_hand_ymin = max(0, int(left_hand_center_y - shoulder_dist // 2))
    left_hand_ymax = min(frame.shape[0], int(left_hand_center_y + shoulder_dist // 2))

    if not np.any(left_wrist) or not np.any(
                    left_elbow) or left_hand_ymax - left_hand_ymin <= 0 or left_hand_xmax - left_hand_xmin <= 0:
                # Wrist or elbow not found -> use entire frame then
                left_hand_crop = frame
                missing_wrists_left.append(clip_len) # I tried this and achived 93% on test
                
    else:
        left_hand_crop = frame[left_hand_ymin:left_hand_ymax, left_hand_xmin:left_hand_xmax, :]
    

    right_wrist = keypoints[0:2, right_wrist_index]
    right_elbow = keypoints[0:2, right_elbow_index]
    right_hand_center = right_wrist + WRIST_DELTA * (right_wrist - right_elbow)
    right_hand_center_x = right_hand_center[0]
    right_hand_center_y = right_hand_center[1]
    right_hand_xmin = max(0, int(right_hand_center_x - shoulder_dist // 2))
    right_hand_xmax = min(frame.shape[1], int(right_hand_center_x + shoulder_dist // 2))
    right_hand_ymin = max(0, int(right_hand_center_y - shoulder_dist // 2))
    right_hand_ymax = min(frame.shape[0], int(right_hand_center_y + shoulder_dist // 2))

    if not np.any(right_wrist) or not np.any(
                right_elbow) or right_hand_ymax - right_hand_ymin <= 0 or right_hand_xmax - right_hand_xmin <= 0:
            # Wrist or elbow not found -> use entire frame then
            right_hand_crop = frame
            missing_wrists_right.append(clip_len) # I tried this and achived 93% on test
            
    else:
        right_hand_crop = frame[right_hand_ymin:right_hand_ymax, right_hand_xmin:right_hand_xmax, :]

    crops = [left_hand_crop, right_hand_crop]
   

    return crops,missing_wrists_left,missing_wrists_right


def read_video_pyav(container,new_name,name):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        frame = frame.to_ndarray(format="rgb24")
        width = 640
        height = 360
        frame = cv2.resize(frame, (width, height))
        
        frames.append(frame)
    return frames

# Hàm để xử lý mỗi hàng của DataFrame
def process_row(row):
    name = row['name']
    new_name = name.replace('.mp4','')
    
    if not os.path.exists(f'np_videos/{new_name}.npz'):
        container = av.open(f'video/{name}')
        frames = read_video_pyav(container,new_name,name)
        np.savez_compressed(f'np_videos/{new_name}.npz', frames)
    else:
        print("exist")

def convert_to_frame(name):
    clip = np.load(f"np_videos/{name}",allow_pickle=True)
    clip = clip['arr_0']
    new_name = name.replace('.npz','')
   
    os.makedirs(f'np_frames/{new_name}',exist_ok=True)
    
    for idx,frame in enumerate(clip):
        np.savez_compressed(f'np_frames/{new_name}/{idx:06d}.npz', frame)

    try:
        os.remove(f"np_videos/{name}")
        print(f"np_videos/{name} removed successfully.")
    except OSError as e:
        print(f"Error: np_videos/{name} - {e.strerror}")
    


# Số luồng (threads) bạn muốn sử dụng
num_threads = 5 # Sửa số luồng tùy ý

# Dùng ThreadPoolExecutor để chạy các hàm process_row() song song
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Duyệt qua từng hàng của DataFrame và thực thi hàm process_row() cho mỗi hàng
    futures = [executor.submit(convert_to_frame, row) for row in data]
    # Sử dụng tqdm để theo dõi tiến trình
    for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        pass