##############################################################################
'''
Provide unified interface for FileSystem, Argument,
Logging and Visualization, etc..
'''
# Author: bingliang zhang
# Update: 2019.3.19
#
##############################################################################
import logging
import pathlib
import argparse
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import pandas as pd
import time
import json
import numpy as np

##############################################################################
# Argument
class Arugument():
    def __init__(self, program_name, cwd=''):
        self.args = {}
        self.add_args({"name":program_name,"cwd":cwd})
        self.parser = argparse.ArgumentParser()

    def add_args_by_arg_parser(self, arg_parser):
        for k in vars(arg_parser).keys():
            self.add_arg(k,vars(arg_parser)[k])

    def init_arg(self, dic:dict):
        for name,value in dic.items():
            self.parser.add_argument(f'--{name}',default=value)
        args = self.parser.parse_args()
        self.add_args_by_arg_parser(args)

    def add_arg(self, name, value):
        if name in self.args:
            raise Warning('Has added the same argument twice!')
        self.args[name]=value
        self.__setattr__(name,value)
    
    def add_args(self, dic:dict):
        for name,value in dic.items():
            self.add_arg(name,value)
    
    def get_arg(self, name):
        return self.args[name]
    
    def print_args(self, logger=None):
        if logger is None:
            for (name,value) in self.args.items():
                print('{:<16} : {}'.format(name, value))
        else:
            for (name,value) in self.args.items():
                logger.info('{:<16} : {}'.format(name, value))

##############################################################################
# File System
class CanonicalFileSystem():
    def __init__(self, program_name, cwd=""):
        if cwd == "":
            self.cwd = pathlib.Path.cwd() / program_name
        else:
            self.cwd = pathlib.Path(cwd) / program_name
        if not self.cwd.exists():
            self.cwd.mkdir()
        self.log = self.cwd / 'log.txt'
        if self.log.exists():
            self.log = self.cwd/'log_aux.txt'
        self.figure = self.cwd / 'figure'
        self.checkpoint = self.cwd / 'checkpoint'
        self.data = self.cwd / 'data'
        self.new_dir = {}
        self.new_file = {}

    def get_root_path(self):
        return self.cwd

    def safe_dirpath(self, path):
        if not path.exists():
            path.mkdir()
        return path

    def get_log_filepath(self):
        return self.log

    def get_figure_dirpath(self):
        return self.safe_dirpath(self.figure)

    def get_checkpoint_dirpath(self):
        return self.safe_dirpath(self.checkpoint)

    def get_data_dirpath(self):
        return self.safe_dirpath(self.data)

    def create_new_file(self, name, relative_path):
        self.new_file[name] = self.cwd/relative_path
        return self.new_file[name]

    def get_filepath(self, name):
        return self.new_file[name]

    def create_new_dir(self, name, relative_path):
        self.new_dir[name] = self.cwd/relative_path
        return self.safe_dirpath(self.new_dir[name])

    def get_dirpath(self, name):
        return self.new_dir[name]

##############################################################################
# Log System
def get_logger(log_path = None, level='info'):
    if level == 'debug':
        _level = logging.DEBUG
    else:
        _level = logging.INFO
    if log_path is not None:
        logger = logging.getLogger()
        logger.setLevel(_level)

        cs = logging.StreamHandler()
        cs.setLevel(_level)
        logger.addHandler(cs)

        fh = logging.FileHandler(log_path, mode='w')
        fh.setLevel(_level)

        logger.addHandler(fh)
    else:
        logger = None
    return logger

##############################################################################
# Visualization (pandas)
class Plotter():
    def __init__(self, figure_dir=None):
        if figure_dir is None:
            self.path = pathlib.Path.cwd()
        else:
            self.path = pathlib.Path(figure_dir)

        self.title_font = font_manager.FontProperties(family=['Microsoft JhengHei']
                                    ,weight='bold',stretch='semi-expanded',size=40)

    def get_DataFrame(self, X,Y):
        data = pd.DataFrame(data=Y,index=X)
        return data

    def get_save_path(self, graph_name):
        return self.path/(graph_name+'.png')

    def set_basic(self, ax, xlabel, ylabel):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        #ax.set_style("fivethirtyeight")

    def show_save(self, path):
        plt.savefig(str(path), dpi = 300)
        plt.show()
        plt.close()

    def plot(self, type, X,Y,graph_name,title='',xlable='',ylabel=''):
        """type:
        'line': X: x coordinates, Y: (multiple) y coordinates (with labels)
        'scatter': X: x coordinates, Y: (multiple) y coordinated (with labels)
        'bar':  X: descriptions for items, Y: data
        'pie':  X: descriptions for items, Y: positive data
        'hist': X: descriptions for items, Y: raw data
        'kde':  X: descriptions for items, Y: raw data
        ''
        """
        data = self.get_DataFrame(X, Y)
        self.plot_DF(type,data,graph_name,title,xlable,ylabel)

    def plot_DF(self,type, data, graph_name, title='',xlable='',ylabel=''):
        plt.style.use('fivethirtyeight')
        fig = plt.figure(figsize=(14,10), dpi=90)
        ax = fig.add_subplot(1, 1, 1)
        if type == 'pie':
            fig = data.plot(kind=type, title=title, ax=ax, subplots = True, autopct = '%.f',fontsize = 25)
        else:
            fig = data.plot(kind=type, title=title, ax=ax,legend = len(data.columns)!=1,fontsize = 25)
        fig.axes.title.set_fontproperties(self.title_font)
        self.set_basic(ax, xlable, ylabel)
        self.show_save(self.get_save_path(graph_name))

    def image_check(self, imgs, labels = None, size=(2,2), save = False, graph_name = 'fig'):
        h, w = size
        for i,img in enumerate(imgs):
            plt.subplot(h, w, i + 1)
            plt.imshow(img)
            if labels is not None:
                plt.title(labels[i])
        if save:
            self.show_save(self.path/graph_name)
        else:
            plt.show()
            plt.close()


##############################################################################
# Timer
class Timer():
    def __init__(self):
        self.item = {}

    def start(self, item_id):
        if item_id not in self.item:
            self.item[item_id] = 0
        self.item[item_id] -= time.time()
        return item_id

    def end(self, item_id):
        self.item[item_id] += time.time()

    # (hour, min, second)
    def time_tuple(self, time):
        second = time % 60
        min = (time // 60) % 60
        hour = time //3600
        return (hour,min, second)

    def get_tuple(self, item_id):
        return self.time_tuple(self.item[item_id])

    def print_item(self, item_id,logger=None):
        tuple = self.time_tuple(self.item[item_id])
        if logger is None:
            print('{:<16} : {}h {}m {:.3f}s'.format(item_id,tuple[0],tuple[1],tuple[2]))
        else:
            logger.info('{:<16} : {}h {}m {:.3f}s'.format(item_id,tuple[0],tuple[1],tuple[2]))

    def print_all(self, logger = None):
        for item in self.item.keys():
            self.print_item(item,logger)

##############################################################################
# Tracker
class Tracker():
    def __init__(self, root=None):
        self.root = None if root is None else pathlib.Path(root)/'data.json'
        self.data = {}

    def track(self, name, value):
        if name not in self.data:
            self.data[name] = []
        self.data[name].append(value)

    def get_array(self, name):
        return np.array(self.data[name])

    def get(self, name):
        return self.data[name]

    def get_all(self):
        return self.data

    def save(self):
        if self.root is None:
            return
        with open(str(self.root),'w') as f:
            json.dump(self.data,f)

    def load(self, path=None):
        if path is None:
            path = self.root
        with open(str(path), 'r') as f:
            self.data = json.load(f)
        return self.data

##############################################################################
# Utils
class Utils():
    def init(self, program_name, cwd=""):
        cwd_dir = pathlib.Path(cwd)
        if not cwd_dir.exists():
            cwd_dir.mkdir()
        self.args = Arugument(program_name,cwd)
        self.fs = CanonicalFileSystem(program_name,cwd)
        self.logger = get_logger(self.fs.get_log_filepath())
        self.plotter = Plotter(self.fs.get_figure_dirpath())
        self.tracker = Tracker(self.fs.get_log_filepath().parent)
        self.timer = Timer()

    def log(self, string):
        if self.logger is None:
            print(string)
        else:
            self.logger.info(string)

    def print_args(self):
        self.args.print_args(self.logger)

    def print_time_info(self):
        self.timer.print_all(self.logger)

    def plot_time_info(self):
        self.plotter.plot('pie',self.timer.item.keys(),self.timer.item.values(),
                         'TimeInfo','Time Info','','time/s')

    def get_fs(self):
        return self.fs

    def get_args(self):
        return self.args

    def get_plotter(self):
        return self.plotter

    def get_timer(self):
        return self.timer

    def get_tracker(self):
        return self.tracker

utils = Utils()