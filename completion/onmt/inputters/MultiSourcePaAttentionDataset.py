from typing import *
from itertools import chain, starmap

import collections
import torch
import os

from seutil import LoggingUtils, IOUtils


class MultiSourcePaAttentionDataset():

    def __init__(self, data_path, mode):
        self.data_path = data_path
        self.src_types = "sim"
        self.mode = mode.lower()
        if self.mode=="valid":
            self.mode = "val"            
        self.data_dict = self.data_reader()        
        self.num_data = len(self.data_dict["src"])

    @property
    def length(self):
         return self.num_data

    def get_data(self, src_file, tgt_file):
        input_data_tensor = [[float(num) for num in l.strip().split()[:2]] for l in IOUtils.load(src_file, IOUtils.Format.txt).splitlines()]
        input_data_tensor = torch.tensor(input_data_tensor).cuda()
        target_data_tensor = [float(l.strip().split()[-1]) for l in IOUtils.load(tgt_file, IOUtils.Format.txt).splitlines()]
        target_data_tensor = torch.tensor(target_data_tensor).cuda()
        return input_data_tensor, target_data_tensor
        
    def data_reader(self):
        data_dict = dict()
        input_data, target_data = self.get_data(os.path.join(self.data_path, f"src.sim.{self.mode}.txt"),
                                                os.path.join(self.data_path, f"tgt.sim.{self.mode}.txt"))
        data_dict["src"] = input_data
        data_dict["tgt"] = target_data
        return data_dict

    def shuffle(self):
        idx = torch.randperm(self.num_data)
        self.data_dict["src"] = self.data_dict["src"][idx,:]
        self.data_dict["tgt"] = self.data_dict["tgt"][idx]
        return
        
    def batch(self, step, batch_size):
        step = step%(self.num_data//batch_size+1)
        if (step+1)*batch_size > self.num_data:
            batch_src = self.data_dict["src"][step*batch_size:self.num_data]
            batch_tgt = self.data_dict["tgt"][step*batch_size:self.num_data]
        else:
            batch_src = self.data_dict["src"][step*batch_size:(step+1)*batch_size]
            batch_tgt = self.data_dict["tgt"][step*batch_size:(step+1)*batch_size]
        return batch_src, batch_tgt
        
    
