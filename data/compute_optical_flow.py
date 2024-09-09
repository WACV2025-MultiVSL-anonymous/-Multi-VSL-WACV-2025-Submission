import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import os



def compute_optical_flow(video_path, output_dir):
    # Tạo thư mục lưu trữ ảnh nếu chưa có
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_count = 1

    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    while(1):
        ret, frame2 = cap.read()
        if not ret:
            break
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        image_name = os.path.join(output_dir, f"{frame_count:06d}.jpg")
        cv2.imwrite(image_name, rgb)
        frame_count+=1
        
        prvs = next

    cap.release()



data = pd.read_csv('/mnt/disk3/anhnct/Hand-Sign-Recognition/data/VN_SIGN/label_1_200/full_data_1_200_center_ord1.csv')

for idx,row in tqdm(data.iterrows()):
    if not os.path.exists(f'/mnt/disk3/anhnct/Hand-Sign-Recognition/data/VN_SIGN/flows/{row["name"].replace(".mp4","")}'):
        compute_optical_flow(f'/mnt/disk3/anhnct/Hand-Sign-Recognition/data/VN_SIGN/video/{row["name"]}',
                    f'/mnt/disk3/anhnct/Hand-Sign-Recognition/data/VN_SIGN/flows/{row["name"].replace(".mp4","")}')
    else:
        print("exists")