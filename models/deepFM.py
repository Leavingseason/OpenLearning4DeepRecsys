'''
Created on June 04, 2017

@author: v-lianji
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


def load_data_from_file_batching(file, batch_size):
    labels = []
    features = []
    cnt = 0
    with open(file, 'r') as rd:
        while True:
            line = rd.readline()
            if not line:
                break
            cnt += 1
            if '#' in line:
                punc_idx = line.index('#')
            else:
                punc_idx = len(line)
            label = float(line[0:1])
            if label>1:
                label=1
            feature_line = line[2:punc_idx]
            words = feature_line.split(' ')
            cur_feature_list = []
            for word in words:
                if not word:
                    continue
                tokens = word.split(':')

                # if tokens[0]=='4532':
                #    print('line ', cnt, ':    ',word, '    line:', line)
                if len(tokens[1]) <= 0:
                    tokens[1] = '0'
                cur_feature_list.append([int(tokens[0]) - 1, float(tokens[1])])
            features.append(cur_feature_list)
            labels.append(label)
            if cnt == batch_size:
                yield labels, features
                labels = []
                features = []
                cnt = 0
    if cnt > 0:
        yield labels, features


def prepare_data_4_sp(labels, features, dim):
    instance_cnt = len(labels)

    indices = []
    values = []
    values_2 = []
    shape = [instance_cnt, dim]
    feature_indices = []

    for i in range(instance_cnt):
        m = len(features[i])
        for j in range(m):
            indices.append([i, features[i][j][0]])
            values.append(features[i][j][1])
            values_2.append(features[i][j][1] * features[i][j][1])
            feature_indices.append(features[i][j][0])

    res = {}

    res['indices'] = np.asarray(indices, dtype=np.int64)
    res['values'] = np.asarray(values, dtype=np.float32)
    res['values2'] = np.asarray(values_2, dtype=np.float32)
    res['shape'] = np.asarray(shape, dtype=np.int64)
    res['labels'] = np.asarray([[label] for label in labels], dtype=np.float32)
    res['feature_indices'] = np.asarray(feature_indices, dtype=np.int64)

    return res


def load_data_cache(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def pre_build_data_cache(infile, outfile, feature_cnt, batch_size):
    wt = open(outfile, 'wb')
    for labels, features in load_data_from_file_batching(infile, batch_size):
        input_in_sp = prepare_data_4_sp(labels, features, feature_cnt)
        pickle.dump(input_in_sp, wt)
    wt.close()


def single_run(feature_cnt, field_cnt,  params):

    print (params)

    pre_build_data_cache_if_need(params['train_file'], feature_cnt, params['batch_size'])
    pre_build_data_cache_if_need(params['test_file'], feature_cnt, params['batch_size'])
    
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

    _y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')
    _ind = tf.placeholder(tf.int64, shape=[None])

    train_step, loss, error, preds, merged_summary, tmp = build_model(_indices, _values, _values2, _shape, _y, _ind,
                                                                 feature_cnt, field_cnt, params)

    # auc = tf.metrics.auc(_y, preds)


    saver = tf.train.Saver()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    log_writer = tf.summary.FileWriter(params['log_path'], graph=sess.graph)

    glo_ite = 0

    #saver.restore(sess, 'models/[500, 100]0.001-36')

    for eopch in range(n_epoch):
        iteration = -1
        start = clock()

        time_load_data, time_sess = 0, 0
        time_cp02 = clock()
        
        train_loss_per_epoch = 0
       
        for training_input_in_sp in load_data_cache(params['train_file']):            
            time_cp01 = clock()
            time_load_data += time_cp01 - time_cp02
            iteration += 1
            glo_ite += 1
            _,  cur_loss, summary, _tmp = sess.run([train_step,  loss, merged_summary, tmp], feed_dict={
                _indices: training_input_in_sp['indices'], _values: training_input_in_sp['values'],
                _shape: training_input_in_sp['shape'], _y: training_input_in_sp['labels'],
                _values2: training_input_in_sp['values2'], _ind: training_input_in_sp['feature_indices']
            })

            time_cp02 = clock()

            time_sess += time_cp02 - time_cp01

            train_loss_per_epoch += cur_loss
            

            log_writer.add_summary(summary, glo_ite)
        end = clock()
        #print('time for eopch ', eopch, ' ', "{0:.4f}min".format((end - start) / 60.0), ' time_load_data:', "{0:.4f}".format(time_load_data), ' time_sess:',
        #      "{0:.4f}".format(time_sess), ' train_loss: ', train_loss_per_epoch, ' train_error: ', train_error_per_epoch)
        if eopch % 5 == 0:
            model_path = params['model_path'] + "/" + str(params['layer_sizes']).replace(':', '_') + str(
                params['reg_w_linear']).replace(':', '_')
            os.makedirs(model_path, exist_ok=True)
            saver.save(sess, model_path, global_step=eopch)            
            auc=predict_test_file(preds, sess, params['test_file'], feature_cnt, _indices, _values, _shape, _y,
                              _values2, _ind, eopch, batch_size, 'test', model_path, params['output_predictions'])
            print('auc is ', auc, ', at epoch  ', eopch, ', time is {0:.4f} min'.format((end - start) / 60.0)
                  , ', train_loss is {0:.2f}'.format(train_loss_per_epoch))
            

    log_writer.close()


def predict_test_file(preds, sess, test_file, feature_cnt, _indices, _values, _shape, _y, _values2, _ind, epoch,
                      batch_size, tag, path, output_prediction = True):
    if output_prediction:
        wt = open(path + '/deepFM_pred_' + tag + str(epoch) + '.txt', 'w')

    gt_scores = []
    pred_scores = []

    for test_input_in_sp in load_data_cache(test_file):
        predictios = sess.run(preds, feed_dict={
            _indices: test_input_in_sp['indices'], _values: test_input_in_sp['values'],
            _shape: test_input_in_sp['shape'], _y: test_input_in_sp['labels'], _values2: test_input_in_sp['values2'],
            _ind: test_input_in_sp['feature_indices']
        }).reshape(-1).tolist()
        
        if output_prediction:
            for (gt, preded) in zip(test_input_in_sp['labels'].reshape(-1).tolist(), predictios):
                wt.write('{0:d},{1:f}\n'.format(int(gt), preded))
                gt_scores.append(gt)
                #pred_scores.append(1.0 if preded >= 0.5 else 0.0)
                pred_scores.append(preded)
        else:
            gt_scores.extend(test_input_in_sp['labels'].reshape(-1).tolist())
            pred_scores.extend(predictios)

    auc = roc_auc_score(np.asarray(gt_scores), np.asarray(pred_scores))
    #print('auc is ', auc, ', at epoch  ', epoch)
    if output_prediction:
        wt.close()
    return auc


def build_model(_indices, _values, _values2, _shape, _y, _ind, feature_cnt, field_cnt, params):
    eta = tf.constant(params['eta'])
    _x = tf.SparseTensor(_indices, _values, _shape)  # m * feature_cnt sparse tensor
    _xx = tf.SparseTensor(_indices, _values2, _shape)

    model_params = []
    tmp = []

    init_value = params['init_value']
    dim = params['dim']
    layer_sizes = params['layer_sizes']

    # w_linear = tf.Variable(tf.truncated_normal([feature_cnt, 1], stddev=init_value, mean=0), name='w_linear',
    #                        dtype=tf.float32)
    w_linear = tf.Variable(tf.truncated_normal([feature_cnt, 1], stddev=init_value, mean=0),  #tf.random_uniform([feature_cnt, 1], minval=-0.05, maxval=0.05), 
                        name='w_linear', dtype=tf.float32)

    bias = tf.Variable(tf.truncated_normal([1], stddev=init_value, mean=0), name='bias')
    model_params.append(bias)
    model_params.append(w_linear)
    preds = bias
    # linear part
    preds += tf.sparse_tensor_dense_matmul(_x, w_linear, name='contr_from_linear')

    w_fm = tf.Variable(tf.truncated_normal([feature_cnt, dim], stddev=init_value / math.sqrt(float(dim)), mean=0),
                           name='w_fm', dtype=tf.float32)
    model_params.append(w_fm)
    # fm order 2 interactions
    if params['is_use_fm_part']:  
        preds = preds + 0.5 * tf.reduce_sum(
            tf.pow(tf.sparse_tensor_dense_matmul(_x, w_fm), 2) - tf.sparse_tensor_dense_matmul(_xx, tf.pow(w_fm, 2)), 1,
            keep_dims=True)

    ## deep neural network
    if params['is_use_dnn_part']:
        w_fm_nn_input = tf.reshape(tf.gather(w_fm, _ind) * tf.expand_dims(_values, 1), [-1, field_cnt * dim])
        print(w_fm_nn_input.shape)

        # tmp.append(tf.shape(tf.expand_dims(_values, 1)))
        # tmp.append(tf.shape(w_fm_nn_input))
        # tmp.append(tf.shape(tf.gather(w_fm, _ind) * tf.expand_dims(_values, 1)))
        # tmp.append(tf.shape(tf.gather(w_fm, _ind)))


        #w_nn_layers = []
        hidden_nn_layers = []
        hidden_nn_layers.append(w_fm_nn_input)
        last_layer_size = field_cnt * dim
        layer_idx = 0

        w_nn_params = []
        b_nn_params = []

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

    # l2_norm = tf.multiply(lambda_w_linear, tf.pow(bias, 2)) + tf.reduce_sum(
    #     tf.add(tf.multiply(lambda_w_linear, tf.pow(w_linear, 2)),
    #            tf.multiply(lambda_w_fm, tf.pow(w_fm, 2)))) + tf.reduce_sum(
    #     tf.multiply(lambda_w_nn, tf.pow(w_nn_output, 2)))

    # l2_norm = tf.multiply(lambda_w_linear, tf.pow(bias, 2)) \
    #           + tf.multiply(lambda_w_linear, tf.reduce_sum(tf.pow(w_linear, 2)))

    l2_norm = tf.multiply(lambda_w_linear, tf.reduce_sum(tf.pow(w_linear, 2))) 
    l2_norm += tf.multiply(lambda_w_l1, tf.reduce_sum(tf.abs(w_linear)))

    if params['is_use_fm_part'] or params['is_use_dnn_part']:
        l2_norm += tf.multiply(lambda_w_fm, tf.reduce_sum(tf.pow(w_fm, 2)))

    if params['is_use_dnn_part']:
        for i in range(len(w_nn_params)):
            l2_norm += tf.multiply(lambda_w_nn, tf.reduce_sum(tf.pow(w_nn_params[i], 2)))

        for i in range(len(b_nn_params)):
            l2_norm += tf.multiply(lambda_w_nn, tf.reduce_sum(tf.pow(b_nn_params[i], 2)))


    # tmp.append(tf.shape(tf.pow(w_linear, 2)))
    # tmp.append(tf.shape(tf.pow(w_fm, 2)))



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

    tf.summary.scalar('square_error', error)
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('linear_weights_hist', w_linear)

    if params['is_use_fm_part']:
        tf.summary.histogram('fm_weights_hist', w_fm)
    if params['is_use_dnn_part']:
        for idx in range(len(w_nn_params))  :
            tf.summary.histogram('nn_layer'+str(idx)+'_weights', w_nn_params[idx])
        
    merged_summary = tf.summary.merge_all()


    return train_step, loss, error, preds, merged_summary, tmp

def pre_build_data_cache_if_need(infile, feature_cnt, batch_size):
    outfile = infile.replace('.csv','.pkl').replace('.txt','.pkl')
    if not os.path.isfile(outfile):
        print('pre_build_data_cache for ', infile)
        pre_build_data_cache(infile, outfile, feature_cnt, batch_size)
        print('pre_build_data_cache finished.' )

def run():
    # train_file = r'\\mlsdata\e$\Users\v-lianji\DeepRecsys\DeepFM\part01.svmlight_balanced.csv'
    # test_file = r'\\mlsdata\e$\Users\v-lianji\DeepRecsys\DeepFM\part02.svmlight.csv'

    print ('begin running')

    field_cnt = 46 #83
    feature_cnt = 46 #5000


    params = {
        'reg_w_linear': 0.00010, 'reg_w_fm':0.0001, 'reg_w_nn': 0.0001,  #0.001
        'reg_w_l1': 0.0001,
        'init_value': 0.1,
        'layer_sizes': [10, 5],
        'activations':['tanh','tanh'],
        'eta': 0.1,
        'n_epoch': 5000,  # 500
        'batch_size': 50,
        'dim': 8,
        'model_path': 'models',
        'log_path': 'logs/' + datetime.utcnow().strftime('%Y-%m-%d_%H_%M_%S'),
        'train_file':  'data/S1_4.txt',  #'data/part01.svmlight_balanced.csv',
        'test_file':    'data/S5.txt',#'data/part02.svmlight.csv',
        'output_predictions':False,
        'is_use_fm_part':False,
        'is_use_dnn_part':True,
        'learning_rate':0.01, # [0.001, 0.01]
        'loss': 'log_loss', # [cross_entropy_loss, square_loss, log_loss]
        'optimizer':'sgd' # [adam, ftrl, sgd]
    }
  

    single_run(feature_cnt, field_cnt, params)

        
if __name__ == '__main__':
    run()

