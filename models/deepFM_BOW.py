'''
Created on May 14, 2017

@author: v-lianji

Upon the deepFM.py, in this file, we make the model support multiple sparse values within one field.
The input format is : label field_idx:feature_idx:value ...  
For all instances, number of field is fixed. However, different instances may have different number of active feautres under each field.

Referring https://github.com/Leavingseason/OpenLearning4DeepRecsys/issues/10 
'''

import tensorflow as tf
import math
from time import clock
import numpy as np
import sys
import os
import pickle
import sys
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from datetime import datetime
import logging
import platform
import random


FIELD_COUNT =  45 #46
FEATURE_COUNT = 100000 #46

os.makedirs('logs/', exist_ok=True)
logging_filename = 'logs/' + platform.node() + '__' + datetime.utcnow().strftime('%Y-%m-%d_%H_%M_%S') + '.log'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(logging_filename)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


handler02 = logging.StreamHandler()
handler02.setLevel(logging.INFO)
handler02.setFormatter(formatter)
logger.addHandler(handler02)



def load_data_from_file_batching(file, batch_size):
    labels = []
    features = []
    qids = []
    docids = []
    cnt = 0
    with open(file, 'r') as rd:
        while True:
            line = rd.readline().strip()
            if not line:
                break
            cnt += 1
            if '#' in line:
                punc_idx = line.index('#')
            else:
                punc_idx = len(line)

            before_comment_line = line[:punc_idx].strip()
            after_comment_line = line[punc_idx + 1:].strip()

            cols = before_comment_line.split()
            label = float(cols[0])
            if label > 0:
                label = 1
            else:
                label = 0
            words = []
            for col in cols[1:]:
                if col.startswith('qid:'):
                    qids.append(col)
                else:
                    words.append(col)
            cur_feature_list = []
            for word in words:
                if not word:
                    continue
                tokens = word.split(':')

                if len(tokens[2]) <= 0:
                    tokens[2] = '0'
                cur_feature_list.append([int(tokens[0]) - 1, int(tokens[1]) - 1, float(tokens[2])])
            features.append(cur_feature_list)
            labels.append(label)
            if len(after_comment_line) > 0:
                docids.append(after_comment_line)

            if len(qids)<len(labels):
                qids.append('qid:fake')
                
            if cnt == batch_size:
                yield labels, features, qids, docids
                labels = []
                features = []
                qids = []
                docids = []
                cnt = 0
    if cnt > 0:
        yield labels, features, qids, docids


def prepare_data_4_sp(labels, features, dim):
    instance_cnt = len(labels)

    indices = []
    values = []
    values_2 = []
    shape = [instance_cnt, dim]
    field2feature_indices = []
    field2feature_values = []
    field2feature_weights = []
    filed2feature_shape = [instance_cnt * FIELD_COUNT, -1]

    lastidx = 0 
    for i in range(instance_cnt):
        m = len(features[i])
        field2features_dic = {}
        for j in range(m):
            indices.append([i, features[i][j][1]])
            values.append(features[i][j][2])
            values_2.append(features[i][j][2] * features[i][j][2])
            #feature_indices.append(features[i][j][1])
            if features[i][j][0] not in field2features_dic:
                field2features_dic[features[i][j][0]] = 0
            else:
                field2features_dic[features[i][j][0]] += 1
            cur_idx = i * FIELD_COUNT + features[i][j][0] 
            #if lastidx<cur_idx-1 or lastidx>cur_idx:
            #    print('lastidx ',lastidx, ' curidx ',cur_idx, ' fieldidx ',features[i][j][0], 'features ',features[i] )
            if lastidx<cur_idx:
                lastidx = cur_idx
            field2feature_indices.append([i * FIELD_COUNT + features[i][j][0], field2features_dic[features[i][j][0]]])
            field2feature_values.append(features[i][j][1])
            field2feature_weights.append(features[i][j][2] ) 
            if filed2feature_shape[1] < field2features_dic[features[i][j][0]]:
                filed2feature_shape[1] = field2features_dic[features[i][j][0]]
    filed2feature_shape[1] += 1

    sorted_index = sorted(range(len(field2feature_indices)), key=lambda k: (field2feature_indices[k][0],field2feature_indices[k][1]))



    res = {}
    res['indices'] = np.asarray(indices, dtype=np.int64)
    res['values'] = np.asarray(values, dtype=np.float32)
    res['values2'] = np.asarray(values_2, dtype=np.float32)
    res['shape'] = np.asarray(shape, dtype=np.int64)
    res['labels'] = np.asarray([[label] for label in labels], dtype=np.float32)
    res['field2feature_indices'] = np.asarray(field2feature_indices, dtype=np.int64)[sorted_index]
    res['field2feature_values'] = np.asarray(field2feature_values, dtype=np.int64)[sorted_index]
    res['field2feature_weights'] = np.asarray(field2feature_weights, dtype=np.float32)[sorted_index]
    res['filed2feature_shape'] = np.asarray(filed2feature_shape, dtype=np.int64)

    return res


def load_data_cache(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def pre_build_data_cache(infile, outfile, batch_size):
    wt = open(outfile, 'wb')
    for labels, features, qids, docids in load_data_from_file_batching(infile, batch_size):
        input_in_sp = prepare_data_4_sp(labels, features, FEATURE_COUNT)
        pickle.dump((input_in_sp, qids, docids), wt)
    wt.close()


def single_run(params):
 
    logger.info('\n\n')
    logger.info(params)
    logger.info('\n\n')


    pre_build_data_cache_if_need(params['train_file'], params['batch_size'], params['clean_cache'] if 'clean_cache' in params else False)
    pre_build_data_cache_if_need(params['test_file'], params['batch_size'], params['clean_cache'] if 'clean_cache' in params else False)
    
    params['train_file'] = params['train_file'].replace('.csv','.pkl').replace('.txt','.pkl')
    params['test_file'] = params['test_file'].replace('.csv','.pkl').replace('.txt','.pkl')
  
    print('start single_run')
    
    tf.reset_default_graph()

    n_epoch = params['n_epoch']
    batch_size = params['batch_size']

    _indices = tf.placeholder(tf.int64, shape=[None, 2], name='raw_indices')
    _values = tf.placeholder(tf.float32, shape=[None], name='raw_values')
    _values2 = tf.placeholder(tf.float32, shape=[None], name='raw_values_square')
    _shape = tf.placeholder(tf.int64, shape=[2], name='raw_shape')

    _field2feature_indices = tf.placeholder(tf.int64, shape=[None, 2], name='field2feature_indices')
    _field2feature_values = tf.placeholder(tf.int64, shape=[None], name='field2feature_values')
    _field2feature_weights = tf.placeholder(tf.float32, shape=[None], name='field2feature_weights')
    _field2feature_shape = tf.placeholder(tf.int64, shape=[2], name='field2feature_shape')

    _y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')


    train_step, loss, error, preds, tmp = build_model(_indices, _values, _values2, _shape
                                                                      , _field2feature_indices, _field2feature_values, _field2feature_weights, _field2feature_shape
                                                                      , _y, params)

    # auc = tf.metrics.auc(_y, preds)


    saver = tf.train.Saver()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # log_writer = tf.summary.FileWriter(params['log_path'], graph=sess.graph)

    glo_ite = 0

    last_best_auc = None
    max_stop_grow_torrelence = 50
    stop_grow_cnt = 0


    #saver.restore(sess, 'models/[500, 100]0.001-36')
    start = clock()
    for eopch in range(n_epoch):
        iteration = -1
        

        time_load_data, time_sess = 0, 0
        time_cp02 = clock()
        
        train_loss_per_epoch = 0
       
        for training_input_in_sp, qids, docids in load_data_cache(params['train_file']):
            
            #if random.random()<0.8:
            #    continue
            
            #print('training_input_in_sp=',training_input_in_sp)
            #sys.exit()
            time_cp01 = clock()
            time_load_data += time_cp01 - time_cp02
            iteration += 1
            glo_ite += 1
            _,  cur_loss = sess.run([train_step,  loss], feed_dict={
                _indices: training_input_in_sp['indices'], _values: training_input_in_sp['values'],
                _shape: training_input_in_sp['shape'], _y: training_input_in_sp['labels'],
                _values2: training_input_in_sp['values2']
                , _field2feature_indices: training_input_in_sp['field2feature_indices']
                , _field2feature_values: training_input_in_sp['field2feature_values']
                , _field2feature_weights: training_input_in_sp['field2feature_weights']
                , _field2feature_shape: training_input_in_sp['filed2feature_shape']
            })


            time_cp02 = clock()

            time_sess += time_cp02 - time_cp01

            train_loss_per_epoch += cur_loss
            

            # log_writer.add_summary(summary, glo_ite)
        end = clock()
        #print('time for eopch ', eopch, ' ', "{0:.4f}min".format((end - start) / 60.0), ' time_load_data:', "{0:.4f}".format(time_load_data), ' time_sess:',
        #      "{0:.4f}".format(time_sess), ' train_loss: ', train_loss_per_epoch, ' train_error: ', train_error_per_epoch)
        if eopch % 1 == 0:
            model_path = params['model_path'] + "/" + str(params['layer_sizes']).replace(':', '_') + str(
                params['reg_w_linear']).replace(':', '_')
            os.makedirs(model_path, exist_ok=True)
            saver.save(sess, model_path, global_step=eopch)            
            metrics=predict_test_file(preds, sess, params['test_file'], _indices, _values, _shape, _y,
                              _values2, _field2feature_indices, _field2feature_values,_field2feature_weights
                                  , _field2feature_shape, eopch, batch_size, 'test', model_path, params['output_predictions']
                                  , params)
            metrics_strs = []
            auc = 0
            for metric_name in metrics:
                metrics_strs.append('{0} is {1:.5f}'.format(metric_name, metrics[metric_name]))
                if metric_name == 'global_auc':
                    auc = metrics[metric_name]
                
            if last_best_auc is None or auc>last_best_auc:
                last_best_auc = auc 
                stop_grow_cnt = 0 
            else:
                stop_grow_cnt+=1
                
            res_str = ' ,'.join(metrics_strs) + ', at epoch {0:d}, time is {1:.4f} min, train_loss is {2:.2f}'.format(eopch, (end -start) / 60.0, train_loss_per_epoch)

            logger.info(res_str)
            start = clock()
            
            if stop_grow_cnt>max_stop_grow_torrelence:
                break 

 

def predict_test_file(preds, sess, test_file, _indices, _values, _shape, _y, _values2, _field2feature_indices, _field2feature_values,_field2feature_weights
                                  , _field2feature_shape, epoch,
                      batch_size, tag, path, output_prediction, params):
    if output_prediction:
        wt = open(path + '/deepFM_pred_' + tag + str(epoch) + '.txt', 'w')

    gt_scores = []
    pred_scores = []

    query2res = {}

    for test_input_in_sp, qids, docids in load_data_cache(test_file):

        predictios = sess.run(preds, feed_dict={
            _indices: test_input_in_sp['indices'], _values: test_input_in_sp['values'],
            _shape: test_input_in_sp['shape'], _y: test_input_in_sp['labels'], _values2: test_input_in_sp['values2'],
                _field2feature_indices: test_input_in_sp['field2feature_indices']
                , _field2feature_values: test_input_in_sp['field2feature_values']
                , _field2feature_weights: test_input_in_sp['field2feature_weights']
                , _field2feature_shape: test_input_in_sp['filed2feature_shape']
        }).reshape(-1).tolist()
        
        if output_prediction:
            for (gt, preded, qid) in zip(test_input_in_sp['labels'].reshape(-1).tolist(), predictios, qids):
                wt.write('{0:d},{1:f}\n'.format(int(gt), preded))
                gt_scores.append(gt)
                #pred_scores.append(1.0 if preded >= 0.5 else 0.0)
                pred_scores.append(preded)
        else:
            for (gt, preded, qid) in zip(test_input_in_sp['labels'].reshape(-1).tolist(), predictios, qids):
                if qid not in query2res:
                    query2res[qid] = []
                query2res[qid].append([gt, preded])

    metrics = compute_metric(query2res, params)

    if output_prediction:
        wt.close()
    return metrics


def compute_metric(query2res, params):
    result = {}

    for m in params['metrics']:
        if 'global_auc' in m['name']:
            gt_scores = []
            pred_scores = []
            for qid in query2res:
                gt_scores.extend([x[0] for x in query2res[qid]] )
                pred_scores.extend([x[1] for x in query2res[qid]] )
            #print('gt_scores ',gt_scores) 
            #print('pred_scores ',pred_scores)   
            result['global_auc'] = roc_auc_score(np.asarray(gt_scores), np.asarray(pred_scores))
        elif 'individual_auc' in m['name']:
            aucs = []
            for qid in query2res:
                gt_scores = np.asarray([x[0] for x in query2res[qid]])
                if gt_scores.min() > 0 or gt_scores.max() < 1:
                    continue
                pred_scores = [x[1] for x in query2res[qid]]
                aucs.append(roc_auc_score(gt_scores, np.asarray(pred_scores)))
            result['individual_auc'] = np.asarray(aucs).mean()
        elif 'precision' in m['name']:
            precisions = []
            for qid in query2res:
                k = min(m['k'], len(query2res[qid]))
                gt_scores = np.asarray([x[0] for x in query2res[qid]])
                pred_scores = np.asarray([x[1] for x in query2res[qid]])
                precision = gt_scores[np.argsort(pred_scores)[::-1][:k]].mean()

                precisions.append(precision)
            result['precision_at_' + str(m['k'])] = np.asarray(precisions).mean()


    return result


def build_model(_indices, _values, _values2, _shape, _field2feature_indices, _field2feature_values,_field2feature_weights, _field2feature_shape, _y, params):
    eta = tf.constant(params['eta'])
    _x = tf.SparseTensor(_indices, _values, _shape)  # m * FEATURE_COUNT sparse tensor
    _xx = tf.SparseTensor(_indices, _values2, _shape)

    model_params = []
    tmp = []

    init_value = params['init_value']
    dim = params['dim']
    layer_sizes = params['layer_sizes']

    # w_linear = tf.Variable(tf.truncated_normal([feature_cnt, 1], stddev=init_value, mean=0), name='w_linear',
    #                        dtype=tf.float32)
    w_linear = tf.Variable(tf.truncated_normal([FEATURE_COUNT, 1], stddev=init_value, mean=0),  #tf.random_uniform([FEATURE_COUNT, 1], minval=-0.05, maxval=0.05),
                        name='w_linear', dtype=tf.float32)

    bias = tf.Variable(tf.truncated_normal([1], stddev=init_value, mean=0), name='bias')
    model_params.append(bias)
    model_params.append(w_linear)
    preds = bias
    # linear part
    preds += tf.sparse_tensor_dense_matmul(_x, w_linear, name='contr_from_linear')

    w_fm = tf.Variable(tf.truncated_normal([FEATURE_COUNT, dim], stddev=init_value / math.sqrt(float(dim)), mean=0),
                           name='w_fm', dtype=tf.float32)
    model_params.append(w_fm)
    # fm order 2 interactions
    if params['is_use_fm_part']:  
        preds = preds + 0.5 * tf.reduce_sum(
            tf.pow(tf.sparse_tensor_dense_matmul(_x, w_fm), 2) - tf.sparse_tensor_dense_matmul(_xx, tf.pow(w_fm, 2)), 1,
            keep_dims=True)

    w_nn_params = []
    b_nn_params = []
    ## deep neural network  
    if params['is_use_dnn_part']:
        # w_fm_nn_input = tf.reshape(tf.gather(w_fm, _ind) * tf.expand_dims(_values, 1), [-1, FIELD_COUNT * dim])
        # print(w_fm_nn_input.shape)

        w_fm_sparseIndexs = tf.SparseTensor(_field2feature_indices, _field2feature_values, _field2feature_shape)
        w_fm_sparseWeights = tf.SparseTensor(_field2feature_indices, _field2feature_weights, _field2feature_shape)
        w_fm_nn_input_orgin = tf.nn.embedding_lookup_sparse(w_fm, w_fm_sparseIndexs,w_fm_sparseWeights,combiner="sum")
        w_fm_nn_input = tf.reshape(w_fm_nn_input_orgin, [-1, dim * FIELD_COUNT]) 


        hidden_nn_layers = []
        hidden_nn_layers.append(w_fm_nn_input)
        last_layer_size = FIELD_COUNT * dim
        layer_idx = 0
 

        for layer_size in layer_sizes:
            cur_w_nn_layer = tf.Variable(
                tf.truncated_normal([last_layer_size, layer_size], stddev=init_value / math.sqrt(float(10)), mean=0),
                name='w_nn_layer' + str(layer_idx), dtype=tf.float32)

            cur_b_nn_layer = tf.Variable(tf.truncated_normal([layer_size], stddev=init_value, mean=0), name='b_nn_layer' + str(layer_idx)) #tf.get_variable('b_nn_layer' + str(layer_idx), [layer_size], initializer=tf.constant_initializer(0.0)) 

            cur_hidden_nn_layer = tf.nn.xw_plus_b(hidden_nn_layers[layer_idx], cur_w_nn_layer, cur_b_nn_layer)
            
            if params['activations'][layer_idx]=='tanh':
                cur_hidden_nn_layer = tf.nn.tanh(cur_hidden_nn_layer)
            elif params['activations'][layer_idx]=='sigmoid':
                cur_hidden_nn_layer = tf.nn.sigmoid(cur_hidden_nn_layer)
            elif params['activations'][layer_idx]=='relu':
                cur_hidden_nn_layer = tf.nn.relu(cur_hidden_nn_layer)
            
            #cur_hidden_nn_layer = tf.matmul(hidden_nn_layers[layer_idx], cur_w_nn_layer)
            #w_nn_layers.append(cur_w_nn_layer)
            hidden_nn_layers.append(cur_hidden_nn_layer)

            layer_idx += 1
            last_layer_size = layer_size

            model_params.append(cur_w_nn_layer)
            model_params.append(cur_b_nn_layer)
            w_nn_params.append(cur_w_nn_layer)
            b_nn_params.append(cur_b_nn_layer)


        w_nn_output = tf.Variable(tf.truncated_normal([last_layer_size, 1], stddev=init_value, mean=0), name='w_nn_output',
                                  dtype=tf.float32)
        nn_output = tf.matmul(hidden_nn_layers[-1], w_nn_output)
        model_params.append(w_nn_output)
        w_nn_params.append(w_nn_output)

        preds += nn_output

    if params['loss'] == 'cross_entropy_loss': # 'loss': 'log_loss'
        error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(preds,[-1])
                                                                       , labels=tf.reshape(_y,[-1])))
    elif params['loss'] == 'square_loss':
        preds = tf.sigmoid(preds)
        error = tf.reduce_mean(tf.squared_difference(preds, _y))  
    elif params['loss'] == 'log_loss':
        preds = tf.sigmoid(preds)
        error = tf.reduce_mean(tf.losses.log_loss(predictions=preds,labels=_y))

    lambda_w_linear = tf.constant(params['reg_w_linear'], name='lambda_w_linear')
    lambda_w_fm = tf.constant(params['reg_w_fm'], name='lambda_w_fm')
    lambda_w_nn = tf.constant(params['reg_w_nn'], name='lambda_nn_fm')
    lambda_w_l1 = tf.constant(params['reg_w_l1'], name='lambda_w_l1')

    l2_norm = tf.multiply(lambda_w_linear, tf.reduce_sum(tf.pow(w_linear, 2))) 
    l2_norm += tf.multiply(lambda_w_l1, tf.reduce_sum(tf.abs(w_linear)))


    if params['is_use_fm_part'] or params['is_use_dnn_part'] or params['is_multi_level']:
        l2_norm += lambda_w_fm * tf.nn.l2_loss(w_fm)
        #l2_norm += tf.multiply(lambda_w_fm, tf.reduce_sum(tf.pow(w_fm, 2)))

    if params['is_use_dnn_part'] or params['is_multi_level']:
        for i in range(len(w_nn_params)):
            l2_norm += lambda_w_nn * tf.nn.l2_loss(w_nn_params[i])
            #l2_norm += tf.multiply(lambda_w_nn, tf.reduce_sum(tf.pow(w_nn_params[i], 2)))

        for i in range(len(b_nn_params)):
            l2_norm += lambda_w_nn * tf.nn.l2_loss(b_nn_params[i])
            #l2_norm += tf.multiply(lambda_w_nn, tf.reduce_sum(tf.pow(b_nn_params[i], 2)))



    loss = tf.add(error, l2_norm)
    if params['optimizer']=='adadelta':    
        train_step = tf.train.AdadeltaOptimizer(eta).minimize(loss,var_list=model_params)#
    elif params['optimizer']=='sgd':
        train_step = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss,var_list=model_params)
    elif params['optimizer']=='adam':
        train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss,var_list=model_params)
    elif params['optimizer']=='ftrl':
        train_step = tf.train.FtrlOptimizer(params['learning_rate']).minimize(loss,var_list=model_params)
    else:
        train_step = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss,var_list=model_params)        

    # tf.summary.scalar('square_error', error)
    # tf.summary.scalar('loss', loss)
    # tf.summary.histogram('linear_weights_hist', w_linear)
    #
    # if params['is_use_fm_part']:
    #     tf.summary.histogram('fm_weights_hist', w_fm)
    # if params['is_use_dnn_part']:
    #     for idx in range(len(w_nn_params))  :
    #         tf.summary.histogram('nn_layer'+str(idx)+'_weights', w_nn_params[idx])
    #
    # merged_summary = tf.summary.merge_all()


    #return train_step, loss, error, preds, merged_summary, tmp
    return train_step, loss, error, preds, tmp

def pre_build_data_cache_if_need(infile, batch_size, rebuild_cache):
    outfile = infile.replace('.csv','.pkl').replace('.txt','.pkl')
    if not os.path.isfile(outfile) or rebuild_cache:
        print('pre_build_data_cache for ', infile)
        pre_build_data_cache(infile, outfile, batch_size)
        print('pre_build_data_cache finished.' )

def run():
    print ('begin running')

    params = {
        'reg_w_linear': 0.0001, 'reg_w_fm':0.0001, 'reg_w_nn': 0.0001,  #0.001
        'reg_w_l1': 0.0001,
        'init_value': 0.001,
        'layer_sizes': [100,500],
        'activations':['relu','tanh'],#
        'eta': 0.1,
        'n_epoch': 5000,  # 500
        'batch_size': 256,
        'dim': 15,
        'model_path': 'models',
        'train_file':  'data/demodata.fieldwise.txt',  
        'test_file':    'data/demodata.fieldwise.txt',
        'output_predictions':False,
        'is_use_fm_part':True,
        'is_use_dnn_part':True, 
        'multi_level_num':1,
        'learning_rate':0.0001, # [0.001, 0.01]
        'loss': 'log_loss', # [cross_entropy_loss, square_loss, log_loss]
        'optimizer':'adam', # [adam, ftrl, sgd]
        'clean_cache':True,
        'metrics': [
            #{'name': 'individual_auc'},
             {'name': 'global_auc'}
            #, {'name': 'precision', 'k': 1}
            #, {'name': 'precision', 'k': 5}
            # , {'name': 'precision', 'k': 10}
        ]

    }
   

    #single_run(feature_cnt, field_cnt, params)
    grid_search( params)

def grid_search( params):
    single_run(  params)
    '''
    for i in range(0,5):      
        params['dim'] = pow(2,i)    
        
        for _ in range(3):
            single_run(  params)
    '''
           
if __name__ == '__main__':
    run()

