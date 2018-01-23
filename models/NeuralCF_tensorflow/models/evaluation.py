'''
Created on Jan 23, 2018

@author: v-lianji
'''

import numpy as np
import math 

def evaluate_one_case(u,i, key2candidates, sess, model, topk):
    key = (u,i) 
    assert(key in key2candidates)
    items = key2candidates[key] 
    users = np.full(len(items), key[0], dtype=np.int32)
    predictions = sess.run(model.output, { model.user_indices : users, model.item_indices : items})
    #print(predictions)
    k = min(topk, len(items))
    sorted_idx = np.argsort(predictions)[::-1]
    selected_items = items[sorted_idx[0:k]]
    #print(sorted_idx)
    #print(i,items[sorted_idx[0]])
    ndcg = getNDCG(selected_items,i)
    hit = getHitRatio(selected_items,i)
    return hit,ndcg
    
def getHitRatio(items,iid):
    if iid in items:
        return 1.0 
    else:
        return 0.0
        

def getNDCG(items,iid):
    for i in range(len(items)):
        if items[i]==iid:
            return math.log(2)/math.log(i+2) 
    return 0.

def evaluate_model(sess, model, dataset, topk):
    hits, ndcgs = [],[]
    for (u,i) in dataset.testPosSet:
        hit,ndcg = evaluate_one_case(u,i,dataset.testPair2NegList,  sess, model, topk)
        hits.append(hit)
        ndcgs.append(ndcg)
        #break 
    return np.asarray(hits).mean(), np.asarray(ndcgs).mean()
        