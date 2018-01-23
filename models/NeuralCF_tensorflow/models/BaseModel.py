'''
Created on Jan 23, 2018

@author: v-lianji
'''
from models import utils 
from dataio import ImpDataset
import tensorflow as tf 
 

class BaseModel(object):
    def __init__(self, args, num_users, num_items):
        self.num_users, self.num_items = num_users, num_items 
        self.lr = args.lr 
        self.learner = args.learner 
        self.init_stddev = args.init_stddev
        self.loss = args.loss
        self.lambda_id_emb = args.reg_id_embedding
        self.lambda_others = args.reg_others
        self.eta = args.eta
    

    def build_train_model(self, model_vector, model_len, ratings, model_params):   
        init_value = self.init_stddev 
        
        w_output = tf.Variable(tf.truncated_normal([model_len, 1], stddev=init_value, mean=0), name='w_output', dtype=tf.float32)
        b_output =  tf.Variable(tf.truncated_normal([1], stddev=init_value*0.01, mean=0), name='b_output', dtype=tf.float32)
        model_params.append(w_output)
        model_params.append(b_output)
        raw_predictions = tf.nn.xw_plus_b(model_vector, w_output, b_output, name='output')
        
        output = tf.reshape(tf.sigmoid(raw_predictions), [-1]) 
        
        with tf.name_scope('error'): 
            type_of_loss = self.loss 
            if type_of_loss == 'cross_entropy_loss':
                raw_error = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(raw_predictions, [-1]), labels=tf.reshape(self.ratings, [-1]))
                error = tf.reduce_mean(
                                   raw_error,
                                   name='error/cross_entropy_loss'
                                   )
            elif type_of_loss == 'square_loss' or type_of_loss == 'rmse':
                raw_error = tf.squared_difference(output, ratings, name='error/squared_diff')
                error = tf.reduce_mean(raw_error, name='error/mean_squared_diff')
            elif type_of_loss == 'log_loss':
                raw_error = tf.losses.log_loss(predictions=output, labels=ratings)
                error = tf.reduce_mean(raw_error, name='error/mean_log_loss')

        
            l2_norm = 0
            for par in model_params:
                l2_norm +=  tf.nn.l2_loss(par) * self.lambda_others
            r'''
            l2_norm += tf.nn.l2_loss(emb_user) * self.lambda_id_emb
            l2_norm += tf.nn.l2_loss(emb_item) * self.lambda_id_emb
            l2_norm += tf.nn.l2_loss(w_output) * self.lambda_others
            l2_norm += tf.nn.l2_loss(b_output) * self.lambda_others
            '''
            
            loss = error + l2_norm    
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) ##--
        with tf.control_dependencies(update_ops):
            type_of_opt = self.learner
            if type_of_opt == 'adadelta':  
                train_step = tf.train.AdadeltaOptimizer(self.eta).minimize(loss,var_list=model_params)#
            elif type_of_opt == 'sgd':
                train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(loss,var_list=model_params)
            elif type_of_opt =='adam':
                train_step = tf.train.AdamOptimizer(self.lr).minimize(loss, var_list=model_params)
            elif type_of_opt =='ftrl':
                train_step = tf.train.FtrlOptimizer(self.lr).minimize(loss,var_list=model_params)
            else:
                train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(loss,var_list=model_params)          
        
        return output, loss, error, raw_error, train_step
            

 