'''
Created on Jan 23, 2018

@author: v-lianji
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', nargs='?', default='../Data/ml-1m/',
                        help='Input data path.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=32,
                        help='Embedding size.')
    parser.add_argument('--reg_id_embedding', nargs='?', default=0.0, type=int,
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--reg_others', nargs='?', default=0.0,  type=float,
                        help="Regularization for general variables.")
    parser.add_argument('--init_stddev', nargs='?', default=0.1,  type=float,
                        help="Init stddev value for variables.")
    parser.add_argument('--num_neg_inst', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--loss',   default='log_loss',
                        help='type of loss function.')
    parser.add_argument('--eta', type=float, default=0.1,
                        help='eta of adadelta')
    parser.add_argument('--topk', type=int, default=10,
                        help='Evaluate the top k items.')
    
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
   
    return parser.parse_args()
