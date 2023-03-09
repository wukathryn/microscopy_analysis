import os

def getsavedirpath(parentdir, newdirname):
    savedirpath = os.path.join(parentdir, newdirname)
    if not os.path.exists(savedirpath):
        os.makedirs(savedirpath)
        print('Created a folder at the following path: ' + savedirpath)
    return savedirpath
