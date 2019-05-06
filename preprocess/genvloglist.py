
import numpy as np
import os

src = 'manifest.txt'
outlist = 'vlog_frames_12fps.txt'
foldername = ''

file = open(src, 'r')
fout = open(outlist, 'w')

for line in file:
    line = line[:-1] # Everything exept the last item
    fname = foldername + line 
    fnms  = len(os.listdir(fname)) # Number of frames

    outstr = fname + ' ' + str(fnms) + '\n' # Output is e.g. videos/O/5/E/v_vAqaXZuAO5E/075 3
    fout.write(outstr)


file.close()
fout.close()
