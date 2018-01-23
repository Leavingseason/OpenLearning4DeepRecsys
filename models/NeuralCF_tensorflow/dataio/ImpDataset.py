'''
Created on Jan 23, 2018

@author: v-lianji
'''

import numpy as np

class ImpDataset(object):
    
    def __init__(self,path):
        self.trainPosSet, self.num_users , self.num_items  = self.load_rating_file_as_set(path + "train.tsv")
        self.testPosSet, _, _ = self.load_rating_file_as_set(path + "test.tsv")
        self.testPair2NegList = self.load_negative_file_as_dict(path + "test.negative.tsv")
    
    
    def load_rating_file_as_set(self, filename):  
        num_users, num_items = 0, 0 
        res = set() 
        with open(filename, 'r') as rd:
            while True:
                line = rd.readline()  
                if not line:
                    break 
                words = line.strip().split('\t')
                u, i = int(words[0]), int(words[1])
                num_users = max(num_users, u) 
                num_items = max(num_items, i)
                key = (u,i) 
                if key not in res:
                    res.add(key)
        
        return res, num_users + 1, num_items + 1 
         
    def  load_negative_file_as_dict(self, filename):  
        res = dict() 
        with open(filename, 'r') as rd:
            while True:
                line = rd.readline() 
                if not line:
                    break 
                words = line.strip().split('\t')
                key = eval(words[0])
                if key in res :
                    continue 
                res[key] = [int(i) for i in words[1:]] 
                res[key].append(key[1])
                res[key] = np.asarray(res[key], dtype = np.int32)
                np.random.shuffle(res[key])
        return res     
            
    def make_training_instances(self, neg_k): 
        user_input, item_input, labels = [],[],[]
        for (u,i) in self.trainPosSet:
            user_input.append(u)
            item_input.append(i)
            labels.append(1.0)
            
            for _ in range(neg_k):
                j = np.random.randint(self.num_items)
                while (u,j) in self.trainPosSet:
                    j = np.random.randint(self.num_items)
                user_input.append(u)
                item_input.append(j)
                labels.append(0.0)
           
        num_inst = len(user_input) 
        user_input, item_input, labels = np.asarray(user_input, np.int32),np.asarray(item_input, np.int32),np.asarray(labels, np.float32)
        indices = np.arange(num_inst) 
        np.random.shuffle(indices)
        return user_input[indices], item_input[indices], labels[indices], num_inst
           