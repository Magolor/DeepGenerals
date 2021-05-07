import os
import re
import sys
import math
import time
import tqdm
import json
import shutil
import logging
import requests
import jsonlines
import subprocess
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ===== File Operations =====
def Folder(PATH):
    return "/".join(PATH.split('/')[:-1])+"/"
    
def File(PATH):
    return PATH.split('/')[-1]

def Prefix(PATH):
    return ".".join(PATH.split('.')[:-1])

def Suffix(PATH):
    return PATH.split('.')[-1]

def Create(PATH):
    None if os.path.exists(PATH) else os.makedirs(PATH)

def Delete(PATH):
    shutil.rmtree(PATH) if os.path.exists(PATH) else None

def Clear(PATH):
    shutil.rmtree(PATH) if os.path.exists(PATH) else None; os.makedirs(PATH)


def SaveJSON(object, FILE, jsonl=False, indent=None):
    if jsonl:
        with jsonlines.open(FILE, 'w') as f:
            for data in object:
                f.write(data)
    else:
        with open(FILE, 'w') as f:
            json.dump(object, f, indent=indent)

def PrettifyJSON(PATH):
    if PATH[-1]=='/':
        for FILE in os.listdir(PATH):
            SaveJSON(LoadJSON(PATH+FILE),PATH+FILE,indent=4)
    else:
        SaveJSON(LoadJSON(PATH),PATH,indent=4)

def LoadJSON(FILE, jsonl=False):
    if jsonl:
        with open(FILE, 'r') as f:
            return [data for data in jsonlines.Reader(f)]
    else:
        with open(FILE, 'r') as f:
            return json.load(f)

def View(something, length=4096):
    print(str(something)[:length]+" ..." if len(str(something))>length+3 else str(something))

def ViewS(something, length=4096):
    return (str(something)[:length]+" ..." if len(str(something))>length+3 else str(something))

def ViewDict(something, length=4096, limit=512):
    print("{")
    for i,item in enumerate(something.items()):
        print("\t"+str(item[0])+": "+(ViewS(item[1])+','))
        if i>=limit:
            print("\t..."); break
    print("}")

def ViewDictS(something, length=4096, limit=512):
    s = "{\n"
    for i,item in enumerate(something.items()):
        s += "\t"+str(item[0])+": "+(ViewS(item[1])+',')+"\n"
        if i>=limit:
            s += "\t...\n"; break
    s += "}\n"; return s

def ViewJSON(json_dict, length=4096):
    print(ViewS(json.dumps(json_dict,indent=4)))

def ViewJSONS(json_dict, length=4096):
    return ViewS(json.dumps(json_dict,indent=4))

# ===== Helper Functions =====
def IP():
    return requests.get('https://api.ipify.org').text

def DATE():
    return time.strftime("%Y-%m-%d",time.localtime(time.time()))

def TQDM(something, s=0, desc=None):
    if type(something) is int:
        return tqdm.trange(s,something+s,desc=desc,dynamic_ncols=True)
    else:
        return zip(tqdm.trange(s,len(list(something))+s,desc=desc,dynamic_ncols=True),list(something))

def CMD(command, wait=True):
    h = subprocess.Popen(command,shell=True); return h.wait() if wait else h

# ===== With Clause =====
class Timer():
    def __init__(self, NAME="Timer"):
        self.name = NAME
    def __enter__(self):
        self.start_time = int(time.time())
        print("[%s Start]"%self.name,time.strftime("%Y-%m-%d %H:%M:%S.",time.localtime(self.start_time))); return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = int(time.time()); interval = self.end_time - self.start_time
        print("[%s   End]"%self.name,time.strftime("%Y-%m-%d %H:%M:%S.",time.localtime(  self.end_time)))
        print("[%s Total]: {:02d}h{:02d}m{:02d}s.".format(interval//3600,interval%3600//60,interval%60))
    def tick(self):
        cur_time = int(time.time()); interval = cur_time - self.start_time
        print("[%s  Tick]"%self.name,time.strftime("%Y-%m-%d %H:%M:%S.",time.localtime(       cur_time)))
        print("[%s Sofar]: {:02d}h{:02d}m{:02d}s.".format(interval//3600,interval%3600//60,interval%60))

class Painter():
    def __init__(self, title, FILE, figsize=(16,9)):
        self.title = title; self.FILE = FILE; self.figsize=figsize
    def __enter__(self):
        fig,axe = plt.subplots(figsize=self.figsize,dpi=300); axe.set_title(self.title); self.axe=axe; return (fig,axe) 
    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.savefig(self.FILE); plt.close()

# ===== Logging =====
def PrintConsole(*args, **kwargs):
    print(*args, file=sys.stdout, **kwargs)

def PrintError(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def LineToFloats(line):
    return [float(s) for s in re.findall(r"(?<!\w)[-+]?\d*\.?\d+(?!\d)",line)]

def ERROR(something):
    return "\033[31m"+str(something)+"\033[0m"

def SUCCESS(something):
    return "\033[32m"+str(something)+"\033[0m"

def WARN(something):
    return "\033[33m"+str(something)+"\033[0m"

def Logger(PATH, level='debug', console=False):
    if level == 'debug':
        _level = logging.DEBUG
    elif level == 'info':
        _level = logging.INFO
    logger = logging.getLogger()
    logger.setLevel(_level)
    if console:
        cs = logging.StreamHandler()
        cs.setLevel(_level)
        logger.addHandler(cs)
    if PATH is not None and PATH != '':
        Create(Folder(PATH))
        file_name = PATH
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)
        logger.addHandler(fh)
    return logger