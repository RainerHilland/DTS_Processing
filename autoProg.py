#!/usr/bin/env python3

import  os, sys
from datetime import datetime
import time, getopt

interval = 20 # in minutes
nFiles = 96

try:
    opts, _ = getopt.getopt(sys.argv[1:], 'i:n:')
    print(opts)
    for opt, arg in opts:
        if opt == '-i':
            interval = int(arg)
            print(" -> will update ever %i minutes" % interval)
        if opt == '-n':
            nFiles = int(arg)
            print(" -> will check for %i files as target" % nFiles)
except:
    raise ValueError("invalid argument passed :/")

while True:

    os.system('clear')
    now = datetime.strftime(datetime.now(), '%H:%M')
    print("* --------------------------------------- *\n\n")
    print("* -- %i minute update %s" % (interval, now))
    #print(datetime.strftime(now, '%H:%M:%S'))
    os.system("python3 progressCheck.py -n %i" % nFiles)

    time.sleep(interval*60)

sys.exit()
