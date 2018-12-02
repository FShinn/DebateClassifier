#!/usr/bin/python3.5

import os
import re



def readOldFiles(root):
    """ Takes in the root path of the debate data and
    returns a data structure of the form, 
    [(topic, [(filename, stance, message), ... ]), ... ]
    This does minimal preprocessing on the text, only collapsing 
    multiple whitespaces."""
    debates = list()
    for d in os.listdir(root):
        dirpath = os.path.join(root, d)
        if os.path.isdir(dirpath):
            entries = list()
            for f in os.listdir(dirpath):
                filepath = os.path.join(dirpath, f)
                if os.path.isfile(filepath):
                    _file = open(filepath)
                    raw = _file.read().split("\n")
                    _file.close()
                    stance = raw[0].strip().replace("#stance=", "", 1)
                    message = re.sub('[ \t]+', ' ', ' '.join(raw[1:]))
                    entries.append({"filename":f, "stance":stance, "message":message})
            debates.append((d, entries))
    return debates

def readFiles(root):
    '''reads files from new debate files'''
    debates = list()
    for d in os.listdir(root):
        dirpath = os.path.join(root, d)
        if os.path.isdir(dirpath):
            entries = list()
            for f in os.listdir(dirpath):
                filepath = os.path.join(dirpath, f)
                metapath = os.path.join(dirpath, f).replace('data', 'meta')
                if os.path.isfile(filepath) and os.path.isfile(metapath) and f.endswith('.data'):
                    datafile = open(filepath,encoding="UTF-8")
                    dataraw = datafile.read().strip().split('\n')
                    datafile.close()
                    message = re.sub('[ \t]+', ' ', ' '.join(dataraw))
                    metafile = open(metapath,encoding="UTF-8")
                    metaraw = metafile.read().strip().split('\n')
                    pid = metaraw[1].split('=')[1]
                    stance = metaraw[2].split('=')[1]
                    metafile.close()
                    if pid == '-1':
                        entries.append({"filename":f, "stance":stance, "message":message, "PID":pid})
            debates.append((d, entries))
    return debates

# ------------------ TESTING --------------------
if __name__ == "__main__":
    debates = readFiles("stance")
    
    for t, es in debates:
        print(t)
        for e in es:
            print("    ", end="")
            print(e["filename"])
            print("        ", end="")
            print(e["stance"])
            print(e["PID"])
            print(e["message"])
