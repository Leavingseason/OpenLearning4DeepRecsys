'''
Created on Jan 23, 2018

@author: v-lianji
'''


from models import utils 
from dataio import ImpDataset
import math 
import tensorflow as tf 
from models.evaluation import *
from time import time
from models.BaseModel import *
from models.GMF import *
from models.MLP import *

class NeuMF(BaseModel):
    def __init__(self, args, num_users, num_items):
        BaseModel.__init__(self, args, num_users, num_items)
        self.layers = eval(args.layers) 
        self.lambda_layers = eval(args.reg_layers)
        self.num_factors = args.num_factors 
        self.model_GMF = GMF(args, num_users, num_items)
        self.model_MLP = MLP(args, num_users, num_items)
    
    def build_core_model(self, user_indices , item_indices): 
        
        vector_GMF, len_GMF, params_GMF = self.model_GMF.build_core_model(user_indices, item_indices)  
        vector_MLP, len_MLP, params_MLP = self.model_MLP.build_core_model(user_indices, item_indices)
        
        model_vector = tf.concat([vector_GMF,vector_MLP],1)
        model_len = len_GMF + len_MLP
        
        model_params = [] 
        model_params.extend(params_GMF)
        model_params.extend(params_MLP)
         
        return model_vector,model_len,model_params

    def build_model(self, user_indices = None, item_indices = None):  
        
        if not user_indices:
            user_indices =  tf.placeholder(tf.int32,[None])
        self.user_indices = user_indices
        if not item_indices:
            item_indices =  tf.placeholder(tf.int32,[None])
        self.item_indices = item_indices
        
        self.ratings = tf.placeholder(tf.float32, [None])
        
        model_vector, model_len, model_params = self.build_core_model(user_indices , item_indices)
        
        self.output, self.loss, self.error, self.raw_error, self.train_step = self.build_train_model( model_vector, model_len, self.ratings, model_params)  
        