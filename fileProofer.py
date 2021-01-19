# check for any failed files!

import os, glob
import pickle

folder = '/home/rainer/data/'
subfolders = glob.glob(os.path.join(folder, '*'))
subfolders.sort()

for subfolder in subfolders:
    print(subfolder)
    checkFiles = glob.glob(os.path.join(subfolder, 'analysis', '*.pkl'))
    checkFiles.sort()
    for file in checkFiles:
        try:
            with open(file, 'rb') as f:
                dummy = pickle.load(f)
        except:
            print("Failure at %s" % file)
