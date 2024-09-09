import pandas as pd
from moviepy.editor import VideoFileClip
from tqdm.auto import tqdm
import os
import random
import numpy as np
import torch
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42)

data = pd.read_csv("full_data_700_1000.csv")

def cut_video(video,implementer,title,idx,order_id,label_id,start_time,end_time,raw_video_name):
    start_time = float(start_time)
    end_time = float(end_time)
    
   
    implementer_id = implementer.split("_")[0]
    title_ = None
    if 'right' in title:
        title_ = 'right'
    elif 'left' in title:
        title_ = 'left'
    elif 'center' in title:
        title_ = 'center'
    assert title_ is not None

    name = f"{raw_video_name}_signer{implementer_id}_{title_}_ord{int(order_id)}_{int(label_id)-1}.mp4"

    # Ghi ra video mới
    url = f"videos_400_1000/{name}"
    if os.path.exists(url):
        print(f"{url} exists" )
        return [name,label_id-1]
    
    cliped_video = video.subclip(start_time, end_time)
    cliped_video.write_videofile(url)
    print("Video ",url)
   
    
    # ffmpeg_extract_subclip(video, start_time, end_time, targetname=url)
    return [name,label_id-1]

# 0 -> 87



ids = range(10000)


for idx,(group_id, group_data) in tqdm(enumerate(data.groupby("id"))):
    processed_video_url = []
    labels = []
    errors = [] 
    if group_id in ids:
        print(group_id,os.path.exists(f"labels/label_{group_id}.csv"))
        if not os.path.exists(f"labels/label_{group_id}.csv"):
            video = VideoFileClip(group_data.iloc[0]['video_path_local'])
            processed_video_url.append(group_data.iloc[0]['video_path_local'])
            for id,(i,row) in enumerate(group_data.iterrows()):
                try:
                    # lb = cut_video(
                    #     video, row['implementer'], row['title'], id+1, row['order_action'], row['label_id'], # 1->199
                    #     row['start_time'], row['end_time'],row['title']
                    # )
                    # lb = cut_video(
                    #     video, row['implementer'], row['title'], id+1, row['order_action'], row['id_label_in_documents'], # 200->400
                    #     row['start_time'], row['end_time'],row['title']
                    # )
                    # lb = cut_video(
                    #     video, row['implementer'], row['title'], id+1, row['order_action'], row['id_label_in_documents'] + 1, # 400->700
                    #     row['start_time'], row['end_time'],row['title']
                    # )
                    lb = cut_video(
                        video, row['implementer'], row['title'], id+1, row['order_action'], row['id_label_in_documents'], # 700->1000
                        row['start_time'], row['end_time'],row['title']
                    )
                    lb.extend([row['id-2']])
                    labels.append(lb)
                except Exception as e:
                    errors.append([group_id,row['start_time'], row['end_time'],e])
            pd.DataFrame(errors,columns=['video_id','start',"end","error"]).to_csv(f"errors/errors_{group_id}.csv",index=False) 
            pd.DataFrame(processed_video_url,columns=['video_url']).to_csv(f"processed_video_url/video_{group_id}.csv",index=False)
            pd.DataFrame(labels,columns=['name','label',"video_lb_id"]).to_csv(f"labels/label_{group_id}.csv",index=False)