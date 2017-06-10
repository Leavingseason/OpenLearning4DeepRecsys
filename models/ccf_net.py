'''
Created on Mar 3, 2017

@author: v-lianji
'''


import tensorflow as tf
from dataio import data_reader 
from dataio import adapter
import math
from time import clock
import numpy as np 


def grid_search(infile,logfile):
	
	#default params:
	params={
		'cf_dim':16, 
		'user_attr_rank':16, 
		'item_attr_rank':16, 
		'layer_sizes':[16,8], 
		'lr':0.1, 
		'lamb':0.001, 
		'mu':4.0, 
		'n_eopch':2000 , 
		'batch_size':500, 
		'init_value ':0.01
		}
	
	dataset = data_reader.movie_lens_data_repos(infile) 
	wt = open(logfile,'w')
	
	lambs=[0.001,0.0001,0.0005,0.005]	
	lrs=[0.1,0.05]
	layer_sizes_list = [[16],[16,8]]
	init_values = [0.01,0.1]
	mu=dataset.training_ratings_score.mean() 
	
	#wt.write('cf_dim,user_attr_rank,item_attr_rank,lr,lamb,mu,n_eopch,batch_size,best_train_rmse,best_test_rmse,best_eval_rmse,best_epoch,init_value,layer_cnt,minutes\n')
	
	for lamb in lambs:
		for lr in lrs:
			for init_value in init_values:
				for layer_sizes in layer_sizes_list:
					params['lamb']=lamb
					params['lr']=lr
					params['init_value']=init_value
					params['layer_sizes']=layer_sizes
					params['mu']=mu
					run_with_parameters(dataset, params, wt)
					run_with_parameters(dataset, params, wt)
	wt.close()

def run_with_parameters(dataset, params, wt):
	
	start = clock()
	tf.reset_default_graph()
	best_train_rmse, best_test_rmse, best_eval_rmse, best_eopch_idx = single_run(dataset, params)
	end = clock()
	wt.write('%f,%f,%f,%d,%f,%s\n' %(best_train_rmse, best_test_rmse, best_eval_rmse, best_eopch_idx,(end-start)/60, str(params)))
	wt.flush()

def single_run(dataset, params):
	cf_dim, user_attr_rank, item_attr_rank, layer_sizes, lr, lamb, mu, n_eopch , batch_size, init_value =  params['cf_dim'], params['user_attr_rank'],params['item_attr_rank'],params['layer_sizes'],params['lr'],params['lamb'],params['mu'],params['n_eopch'],params['batch_size'],params['init_value']
	##  compose features from SVD
	user_cnt,user_attr_cnt = dataset.n_user, dataset.n_user_attr
	item_cnt,item_attr_cnt = dataset.n_item, dataset.n_item_attr
	
	W_user = tf.Variable(tf.truncated_normal([user_cnt, cf_dim], stddev=init_value/math.sqrt(float(cf_dim)), mean=0), name = 'user_cf_embedding', dtype=tf.float32)
	W_item = tf.Variable(tf.truncated_normal([item_cnt, cf_dim], stddev=init_value/math.sqrt(float(cf_dim)), mean=0), name = 'item_cf_embedding', dtype=tf.float32)
	
	W_user_bias = tf.concat([W_user, tf.ones((user_cnt,1), dtype=tf.float32)], 1, name='user_cf_embedding_bias')
	W_item_bias = tf.concat([tf.ones((item_cnt,1), dtype=tf.float32), W_item], 1, name='item_cf_embedding_bias')
	
	##  compose features from attributes
	user_attr_indices, user_attr_indices_values, user_attr_indices_weights =  compose_vector_for_sparse_tensor(dataset.user_attr)
	item_attr_indices, item_attr_indices_values, item_attr_indices_weights =  compose_vector_for_sparse_tensor(dataset.item_attr)
	
	user_sp_ids = tf.SparseTensor(indices=user_attr_indices, values = user_attr_indices_values, dense_shape=[user_cnt,user_attr_cnt])
	user_sp_weights = tf.SparseTensor(indices = user_attr_indices, values = user_attr_indices_weights, dense_shape=[user_cnt, user_attr_cnt])
	
	
	item_sp_ids = tf.SparseTensor(indices=item_attr_indices, values = item_attr_indices_values, dense_shape=[item_cnt,item_attr_cnt])
	item_sp_weights = tf.SparseTensor(indices = item_attr_indices, values = item_attr_indices_weights, dense_shape=[item_cnt, item_attr_cnt])
	
	W_user_attr = tf.Variable(tf.truncated_normal([user_attr_cnt, user_attr_rank], stddev=init_value/math.sqrt(float(user_attr_rank)), mean=0), name = 'user_attr_embedding',dtype=tf.float32)
	W_item_attr = tf.Variable(tf.truncated_normal([item_attr_cnt, item_attr_rank], stddev=init_value/math.sqrt(float(item_attr_rank)), mean=0), name = 'item_attr_embedding',dtype=tf.float32)
	
	user_embeddings = tf.nn.embedding_lookup_sparse(W_user_attr, user_sp_ids, user_sp_weights,  name='user_embeddings', combiner='sum')
	item_embeddings = tf.nn.embedding_lookup_sparse(W_item_attr, item_sp_ids, item_sp_weights,  name='item_embeddings', combiner='sum')
		
	
	user_indices =  tf.placeholder(tf.int32,[None])
	item_indices =  tf.placeholder(tf.int32,[None])
	ratings = tf.placeholder(tf.float32, [None])
	
	user_cf_feature = tf.nn.embedding_lookup(W_user_bias, user_indices, name = 'user_feature')
	item_cf_feature = tf.nn.embedding_lookup(W_item_bias, item_indices, name = 'item_feature')	
	
	
	user_attr_feature = tf.nn.embedding_lookup(user_embeddings, user_indices, name = 'user_feature')
	item_attr_feature = tf.nn.embedding_lookup(item_embeddings, item_indices, name = 'item_feature')
	
	#tf.summary.image('user_feautre', user_feature)
	

	train_step, square_error, loss, merged_summary = build_model(user_cf_feature, user_attr_feature, user_attr_rank, 
																item_cf_feature,item_attr_feature, item_attr_rank, 
																ratings, layer_sizes, 
																W_user, W_item,
																W_user_attr,W_item_attr,
																lamb,lr, mu)
	
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init) 
	
	#print(sess.run(user_embeddings))
	
	train_writer = tf.summary.FileWriter(r'\\mlsdata\e$\Users\v-lianji\DeepRecsys\Test\logs', sess.graph)
	
	
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
		error_eval = sess.run(square_error, { user_indices : dataset.eval_ratings_user, item_indices : dataset.eval_ratings_item, ratings : dataset.eval_ratings_score})
		
		loss_traing = sess.run(loss, { user_indices : dataset.training_ratings_user, item_indices : dataset.training_ratings_item, ratings : dataset.training_ratings_score})
		#loss_test = sess.run(loss, { user_feature : test_user_feature, item_feature : test_item_feature, ratings : test_label})
		
		
		summary = sess.run(merged_summary, { user_indices : dataset.training_ratings_user, item_indices : dataset.training_ratings_item, ratings : dataset.training_ratings_score})
		train_writer.add_summary(summary, ite)
		end = clock()
		print("Iteration %d  RMSE(train): %f  RMSE(test): %f   RMSE(eval): %f   LOSS(train): %f  minutes: %f" %(ite, error_traing, error_test, error_eval, loss_traing, (end-start)/60))
		
		if best_test_rmse<0 or best_test_rmse>error_test:
			best_train_rmse, best_test_rmse, best_eval_rmse = error_traing,error_test, error_eval 
			best_eopch_idx = ite 
		else:
			if ite - best_eopch_idx>10:
				break 
			
	
	train_writer.close()
	
	return best_train_rmse, best_test_rmse, best_eval_rmse, best_eopch_idx

def build_model(user_cf_feature, user_attr_feature, user_attr_rank, 
			item_cf_feature, item_attr_feature, item_attr_rank, 
			ratings, layer_size, 
			W_user, W_item,
			W_user_attr, W_item_attr, lamb , lr, mu ):
	
	layer_cnt = len(layer_size)
	hiddens_user = [] 
	hiddens_item = [] 
	
	hiddens_user.append(user_attr_feature)
	hiddens_item.append(item_attr_feature)
	
	b_user_list = []
	b_item_list = []
	W_user_list = [] 
	W_item_list = []
	
	for i in range(layer_cnt):
		with tf.name_scope('layer_'+str(i)):
			b_user_list.append(tf.Variable(tf.truncated_normal([layer_size[i]]),name='user_bias'))
			b_item_list.append(tf.Variable(tf.truncated_normal([layer_size[i]]),name='item_bias'))
			if i==0:
				W_user_list.append(tf.Variable(tf.truncated_normal([user_attr_rank, layer_size[i]], stddev=1/math.sqrt(float(layer_size[i])), mean=0), name = 'W_user'))
				W_item_list.append(tf.Variable(tf.truncated_normal([item_attr_rank, layer_size[i]], stddev=1/math.sqrt(float(layer_size[i])), mean=0), name= 'W_item'))
				
				user_middle = tf.matmul(user_attr_feature,W_user_list[i]) + b_user_list[i]
				item_middle = tf.matmul(item_attr_feature,W_item_list[i]) + b_item_list[i]  
				
			else:
				W_user_list.append(tf.Variable(tf.truncated_normal([layer_size[i-1], layer_size[i]], stddev=1/math.sqrt(float(layer_size[i])), mean=0), name = 'W_user'))
				W_item_list.append(tf.Variable(tf.truncated_normal([layer_size[i-1], layer_size[i]], stddev=1/math.sqrt(float(layer_size[i])), mean=0), name= 'W_item'))
				
				user_middle =tf.matmul(hiddens_user[i],W_user_list[i]) + b_user_list[i]
				item_middle =tf.matmul(hiddens_item[i],W_item_list[i]) + b_item_list[i]
			
			hiddens_user.append(tf.identity(user_middle, name = 'factor_user')) #identity ,sigmoid
			hiddens_item.append(tf.identity(item_middle, name = 'factor_item'))
		
		
	factor_user = hiddens_user[layer_cnt] 
	factor_item = hiddens_item[layer_cnt] 
	
	
	
	preds = tf.reduce_sum( tf.multiply(user_cf_feature , item_cf_feature) , 1) +  tf.reduce_sum( tf.multiply(factor_user , factor_item) , 1) + mu
	# TODO:  bound the prediction within [min_score, max_score]
	
	square_error = tf.sqrt(tf.reduce_mean( tf.squared_difference(preds, ratings)))
	loss = square_error 
	for i in range(layer_cnt):
		loss = loss + lamb*(
							tf.reduce_mean(tf.nn.l2_loss(W_user)) + tf.reduce_mean(tf.nn.l2_loss(W_item)) +
							tf.reduce_mean(tf.nn.l2_loss(W_user_attr)) + tf.reduce_mean(tf.nn.l2_loss(W_item_attr)) + 
							tf.reduce_mean(tf.nn.l2_loss(W_user_list[i])) + tf.reduce_mean(tf.nn.l2_loss(W_item_list[i])) + tf.reduce_mean(tf.nn.l2_loss(b_user_list[i])) + tf.reduce_mean(tf.nn.l2_loss(b_item_list[i]))
						)
		
	tf.summary.scalar('square_error', square_error)
	tf.summary.scalar('loss', loss)
	merged_summary = tf.summary.merge_all()
	#tf.global_variables_initializer()
	train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

	return train_step, square_error, loss, merged_summary



def compose_vector_for_sparse_tensor(entity2attr_list):
	indices = [] 
	indices_values = []
	weight_values = [] 
	
	N = len(entity2attr_list)
	for i in range(N):
		if len(entity2attr_list[i])>0:
			cnt = 0 
			for attr_pair in entity2attr_list[i]:
				#print(entity2attr_list)
				indices.append([i,cnt])
				indices_values.append(attr_pair[0])
				weight_values.append(attr_pair[1])
				cnt+=1
		else:
			indices.append([i,0])
			indices_values.append(0)
			weight_values.append(0)
	return indices, indices_values, weight_values



if __name__ == '__main__':
	
	grid_search(r'data/movielens_100k.pkl',
			r'logs/CCFNet_movielens10m.csv')
	pass 
