
import numpy as np
import os
from pathlib import Path
import json

src = '/home/mtoering/data/hmdb_videos/jpg'
ann = '/home/mtoering/data/hmdb51_1.json'
outlist = 'hmdb_1.txt'
foldername = ''

with open(ann) as annotation:
    d = json.loads(annotation.read())

print(d["database"])


fout = open(outlist, 'w')

for classname in os.listdir(src):
    class_path = os.path.join(src, classname)
    for video_folder in os.listdir(class_path):
        if video_folder in d["database"]:
            if d["database"][video_folder]["subset"] == "training":

                video_path = os.path.join(class_path, video_folder)
                fname = video_path
                path = os.path.join(video_path, 'n_frames')
                file = open(path, 'r')
                fnms = file.read()
                file.close()
                outstr = fname + ' ' + str(fnms) + '\n' # Output is e.g. videos/O/5/E/v_vAqaXZuAO5E/075 3
                fout.write(outstr)

fout.close()

