import shutil
import pandas as pd
import os

videos = [i for i in os.listdir("videos") if int(i.split("_")[-1].split(".")[0]) >= 400]


if __name__ == '__main__':
    for video in videos:
        shutil.copyfile(f"videos/{video}",f"/mnt/disk2/anhnct/video_400_1000/{video}")