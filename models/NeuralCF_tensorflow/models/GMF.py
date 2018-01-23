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


class GMF(BaseModel):
    def __init__(self, args, num_users, num_items):
        BaseModel.__init__(self, args, num_users, num_items)
        self.num_factors = args.num_factors 
        

    def build_core_model(self, user_indices , item_indices): 
        
        init_value = self.init_stddev  
        
        emb_user = tf.Variable(tf.truncated_normal([self.num_users, self.num_factors], stddev=init_value/math.sqrt(float(self.num_factors)), mean=0), name = 'user_embedding', dtype=tf.float32)
        emb_item = tf.Variable(tf.truncated_normal([self.num_items, self.num_factors], stddev=init_value/math.sqrt(float(self.num_factors)), mean=0), name = 'item_embedding', dtype=tf.float32)
        
        emb_user_bias = tf.concat([emb_user, tf.ones((self.num_users,1) , dtype=tf.float32)* 0.1], 1, name='user_embedding_bias')
        emb_item_bias = tf.concat([tf.ones((self.num_items,1), dtype=tf.float32)* 0.1, emb_item], 1, name='item_embedding_bias')
        
        user_feature = tf.nn.embedding_lookup(emb_user_bias, user_indices, name = 'user_feature') 
        item_feature = tf.nn.embedding_lookup(emb_item_bias, item_indices, name = 'item_feature')   
        
        product_vector = tf.multiply(user_feature , item_feature)
           
        model_params = [emb_user,emb_item]  

        return product_vector, self.num_factors+1, model_params

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
     