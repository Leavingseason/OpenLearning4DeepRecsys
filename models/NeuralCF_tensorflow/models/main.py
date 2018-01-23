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
from models.GMF import *
from models.MLP import *
from models.NeuMF import *



def single_run(args, dataset):
    
    model = NeuMF(args, dataset.num_users, dataset.num_items)
    model.build_model()

    sess = tf.Session() 
    init = tf.global_variables_initializer() 
    sess.run(init) 
    
    t1 = time()
    ahit, andcg = evaluate_model(sess, model, dataset, args.topk)
    best_hr, best_ndcg, best_iter = ahit, andcg, -1
    print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (ahit, andcg, time()-t1))
    
    for epoch in range(args.epochs):
        t1 = time()
        train_users, train_items, train_labels, num_inst = dataset.make_training_instances(args.num_neg_inst) 
        #print(train_labels[0:20])
        loss_per_epoch , error_per_epoch = 0, 0 
        for ite in range((num_inst-1)//args.batch_size+1):
            start_idx = ite * args.batch_size 
            end_idx = min((ite+1)*args.batch_size , num_inst) 
            cur_user_indices, cur_item_indices, cur_label = train_users[start_idx:end_idx], train_items[start_idx:end_idx],train_labels[start_idx:end_idx]
            
            _, loss, error = sess.run([model.train_step, model.loss, model.raw_error], { model.user_indices : cur_user_indices, model.item_indices : cur_item_indices, model.ratings : cur_label})
            loss_per_epoch +=loss
            error_per_epoch += error 
        error_per_epoch /= num_inst
        t2 = time()
        if epoch % args.verbose == 0:
            ahit, andcg = evaluate_model(sess, model, dataset, args.topk)
            print('epoch %d   \t[%.1f s]: HR= %.4f\tNDCG= %.4f\tloss= %.4f\terror= %.4f\t[%.1f s]' %(epoch, t2-t1, ahit, andcg, loss_per_epoch, error_per_epoch, time()-t2))
            if ahit > best_hr :
                best_hr = ahit 
                best_iter = epoch 
            if andcg > best_ndcg :
                best_ndcg = andcg 
                
    print("End. Best Epoch %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))    
        

if __name__ == '__main__':
    args = utils.parse_args() 
    
    print("runtime arguments: %s" %(args))
    
    dataset = ImpDataset.ImpDataset(args.path)
    
    print(dataset.num_users,dataset.num_items)
    
    single_run(args, dataset) 
    
    
    
    