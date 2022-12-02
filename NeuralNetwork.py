import numpy as np
import pandas as pd
import NegativePool
import VectorDict

class NeuralNetwork:
    def __init__(self, path, lr, n_epoch, limit=None, dim=100, mode='unigram-table', a=0.75, n_negative=5, sgd='adagrad', kind='input', max_size_np=1e8):
        self.n_processed = 0
        self.n_epoch = n_epoch
        self.limit = limit
        
        self.vec_dict = VectorDict.vector_dict(dim, lr, sgd, kind)
        self.ne_pool = NegativePool.negative_pool(a, mode, max_size_np, n_negative)
        
        self.load_data(path)
        
    def load_data(self, p):
        self.packets = pd.read_csv(p, dtype=str)
        self.limit = self.packets.shape[0]
        
    def preproc_packet(self, p_idx):
        srcIP = self.packets['srcIP'][p_idx]
        dstIP = self.packets['dstIP'][p_idx]
        if srcIP < dstIP:
            flow_name = srcIP+dstIP
        else:
            flow_name = dstIP+srcIP
        
        return [flow_name] + [srcIP, dstIP, self.packets['srcproto'][p_idx], self.packets['dstproto'][p_idx],
                self.packets['srcMAC'][p_idx], self.packets['dstMAC'][p_idx], self.packets['len'][p_idx],
                self.packets['IPtype'][p_idx]]
        
        
    def proc_packet(self):
        ext_packet = self.preproc_packet(self.n_processed)
        self.vec_dict.update(ext_packet)
        
        for target in ext_packet:
            for context in ext_packet:
                if target == context:
                    continue
                
                neg_samples = self.ne_pool.get()
                
                for i in range(self.n_epoch):
                    self.vec_dict.gradient_descendent(target, context, neg_samples)
        
        self.ne_pool.update(ext_packet)
        self.n_processed += 1
        
        return self.vec_dict.get(ext_packet[0])
    
    def next_packet(self):
        if self.limit <= self.n_processed:
            print(str(self.n_processed)+" processed, Neural Network: off")
            return []
        else:
            return self.proc_packet()