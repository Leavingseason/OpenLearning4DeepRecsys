'''
Created on Mar 3, 2017

@author: v-lianji
'''


import tensorflow as tf
from dataio import data_reader 
import math
from time import clock
import numpy as np


def build_model(user_indices, item_indices, rank, ratings, user_cnt, item_cnt, lr, lamb, mu, init_value):
	
	
	W_user = tf.Variable(tf.truncated_normal([user_cnt, rank], stddev=init_value/math.sqrt(float(rank)), mean=0), name = 'user_embedding', dtype=tf.float32)
	W_item = tf.Variable(tf.truncated_normal([item_cnt, rank], stddev=init_value/math.sqrt(float(rank)), mean=0), name = 'item_embedding', dtype=tf.float32)
	
	W_user_bias = tf.concat([W_user, tf.ones((user_cnt,1), dtype=tf.float32)], 1, name='user_embedding_bias')
	W_item_bias = tf.concat([tf.ones((item_cnt,1), dtype=tf.float32), W_item], 1, name='item_embedding_bias')
	
	user_feature = tf.nn.embedding_lookup(W_user_bias, user_indices, name = 'user_feature')
	item_feature = tf.nn.embedding_lookup(W_item_bias, item_indices, name = 'item_feature')	
	
	
	preds = tf.add(tf.reduce_sum( tf.multiply(user_feature , item_feature) , 1), mu)
	
	square_error = tf.sqrt(tf.reduce_mean( tf.squared_difference(preds, ratings)))
	loss = square_error + lamb*(tf.reduce_mean(tf.nn.l2_loss(W_user)) + tf.reduce_mean(tf.nn.l2_loss(W_item)))
		
	tf.summary.scalar('square_error', square_error)
	tf.summary.scalar('loss', loss)
	merged_summary = tf.summary.merge_all()
	#tf.global_variables_initializer()
	train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)   # tf.train.AdadeltaOptimizer(learning_rate=lr).minimize(loss)    #

	return train_step, square_error, loss, merged_summary

def grid_search_params():

	dataset = data_reader.sparse_data_repos(10000,10005)
	dataset.load_trainging_ratings(r'data/userbook_unique_compactid_train.txt')
	dataset.load_test_ratings(r'data/userbook_unique_compactid_valid.txt')
	dataset.load_eval_ratings(r'data/userbook_unique_compactid_test.txt')
	log_file = r'logs/BMF_book.csv'
	
	wt = open(log_file,'w')
	rank = 16
	lambs=[0.00003,0.00005,0.0001]
	batch_sizes=[500]
	n_eopch=2000
	lrs=[0.1]
	init_values = [0.01]
	#mu=dataset.training_ratings_score.mean()
	mu = np.asarray(dataset.training_ratings_score, dtype=np.float32).mean() 
	wt.write('rank,lr,lamb,mu,n_eopch,batch_size,best_train_rmse,best_test_rmse,best_eval_rmse,best_epoch,init_value,minutes\n')
	for lamb in lambs:
		for lr in lrs:
			for init_value in init_values:
				for batch_size in batch_sizes:
					run_with_parameter(dataset,rank,lr,lamb,mu,n_eopch,batch_size,wt, init_value)
	wt.close()

def run_with_parameter(dataset,rank,lr,lamb,mu,n_eopch,batch_size,wt, init_value):
	start = clock()
	tf.reset_default_graph()
	best_train_rmse, best_test_rmse, best_eval_rmse, best_eopch_idx = single_run(dataset,rank,dataset.n_user,dataset.n_item,lr,lamb,mu,n_eopch,batch_size,True, init_value)
	end = clock()
	wt.write('%d,%f,%f,%f,%d,%d,%f,%f,%f,%d,%f,%f\n' %(rank,lr,lamb,mu,n_eopch,batch_size,best_train_rmse, best_test_rmse, best_eval_rmse,best_eopch_idx,init_value,(end-start)/60))
	wt.flush()




def single_run(dataset,rank,user_cnt,item_cnt,lr,lamb,mu,n_eopch,batch_size,is_eval_on, init_value):
	
	user_indices =  tf.placeholder(tf.int32,[None])
	item_indices =  tf.placeholder(tf.int32,[None])
	ratings = tf.placeholder(tf.float32, [None])	


	train_step, square_error, loss, merged_summary = build_model(user_indices, item_indices, rank, ratings, user_cnt, item_cnt, lr, lamb, mu, init_value)
	
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init) 
	
	#print(sess.run(user_embeddings))
	
	train_writer = tf.summary.FileWriter(r'logs', sess.graph)
	
	n_instances = len(dataset.training_ratings_user)

	best_train_rmse, best_test_rmse, best_eval_rmse = -1, -1, -1
	best_eopch_idx = -1 
	for ite in range(n_eopch):
		#print(ite)
		start = clock()
		for i in range(n_instances//batch_size):
			start_idx = i * batch_size 
			end_idx = start_idx + batch_size
			cur_user_indices, cur_item_indices, cur_label = dataset.training_ratings_user[start_idx:end_idx], dataset.training_ratings_item[start_idx:end_idx],dataset.training_ratings_score[start_idx:end_idx]
			
			sess.run(train_step, { user_indices : cur_user_indices, item_indices : cur_item_indices, ratings : cur_label})	
			
		error_traing = sess.run(square_error, { user_indices : dataset.training_ratings_user, item_indices : dataset.training_ratings_item, ratings : dataset.training_ratings_score})
		error_test = sess.run(square_error, { user_indices : dataset.test_ratings_user, item_indices : dataset.test_ratings_item, ratings : dataset.test_ratings_score})
		if is_eval_on:
			error_eval = sess.run(square_error, { user_indices : dataset.eval_ratings_user, item_indices : dataset.eval_ratings_item, ratings : dataset.eval_ratings_score})
		else: 
			error_eval = -1
			
		if best_test_rmse<0 or best_test_rmse>error_test:
			best_train_rmse, best_test_rmse, best_eval_rmse = error_traing,error_test, error_eval 
			best_eopch_idx = ite 
		else:
			if ite - best_eopch_idx>10:
				break 
			
		loss_traing = sess.run(loss, { user_indices : dataset.training_ratings_user, item_indices : dataset.training_ratings_item, ratings : dataset.training_ratings_score})
		#loss_test = sess.run(loss, { user_feature : test_user_feature, item_feature : test_item_feature, ratings : test_label})
		summary = sess.run(merged_summary, { user_indices : dataset.training_ratings_user, item_indices : dataset.training_ratings_item, ratings : dataset.training_ratings_score})
		train_writer.add_summary(summary, ite)
		end = clock()
		print("Iteration %d  RMSE(train): %f  RMSE(test): %f   RMSE(eval): %f   LOSS(train): %f  minutes: %f" %(ite, error_traing, error_test, error_eval, loss_traing, (end-start)/60))
		
	
	train_writer.close()
	
	return best_train_rmse, best_test_rmse, best_eval_rmse,best_eopch_idx

if __name__ == '__main__':
	
	grid_search_params()
	#run()
	pass 
