'''
Created on Mar 1, 2017

@author: v-lianji
'''

import pandas
import codecs
import pickle
from scipy.sparse import find
import numpy as np

class movie_lens_data_repos:
	def __init__(self, file):
		with codecs.open(file,'rb') as f:
			train,validate,test,user_content,item_content = pickle.load(f)
		
		train = train.reindex(np.random.permutation(train.index))
		
		self.training_ratings_user = train.loc[:,'user']
		self.training_ratings_item = train.loc[:,'item']
		self.training_ratings_score = train.loc[:,'rate'] 
		self.test_ratings_user = validate.loc[:,'user']
		self.test_ratings_item = validate.loc[:,'item']
		self.test_ratings_score = validate.loc[:,'rate'] 
		self.eval_ratings_user = test.loc[:,'user']
		self.eval_ratings_item = test.loc[:,'item']
		self.eval_ratings_score = test.loc[:,'rate'] 
		
		self.n_user = max([self.training_ratings_user.max(),self.test_ratings_user.max(),self.eval_ratings_user.max()])+1
		self.n_item = max([self.training_ratings_item.max(),self.test_ratings_item.max(),self.eval_ratings_item.max()])+1

		self.n_user_attr, self.n_item_attr = user_content.shape[1], item_content.shape[1]
		print('n_user=%d n_item=%d n_user_attr=%d n_item_attr=%d' %(self.n_user,self.n_item,self.n_user_attr, self.n_item_attr))
		
		self.user_attr = self.BuildAttributeFromSPMatrix(user_content,self.n_user,self.n_user_attr)
		self.item_attr = self.BuildAttributeFromSPMatrix(item_content,self.n_item,self.n_item_attr)
		
	def BuildAttributeFromSPMatrix(self, sp_matrix, n, m):
		res = [] 
		for _ in range(n):
			res.append([])
		(row,col,value) = find(sp_matrix)
		for r,c,v in zip(row,col,value):
			res[r].append([c,float(v)])
		return res
			
		

class sparse_data_repos:
	def __init__(self, n_user, n_item, n_user_attr = 0, n_item_attr = 0):
		self.n_user = n_user 
		self.n_item = n_item 
		self.n_user_attr = n_user_attr 
		self.n_item_attr = n_item_attr 	
		self.user_attr = []
		self.item_attr = [] 	
		self.training_ratings_user = []
		self.training_ratings_item = []
		self.training_ratings_item02 = []
		self.training_ratings_score = []
		self.test_ratings_user = []
		self.test_ratings_item = []
		self.test_ratings_item02 = []
		self.test_ratings_score = []
		self.eval_ratings_user=[]
		self.eval_ratings_item=[]
		self.eval_ratings_score=[]
	
	def load_user_attributes(self, infile,spliter='\t'):
		self.load_attributes(self.user_attr, self.n_user, self.n_user_attr, infile,spliter)
	
	def load_item_attributes(self, infile,spliter='\t'):
		self.load_attributes(self.item_attr, self.n_item, self.n_item_attr, infile,spliter)
		
			
	def load_attributes(self, res, n, m, infile,spliter):		
		del res[:]
		for i in range(n):
			res.append([]) 
		
		with open(infile, 'r') as rd:
			while True:
				line = rd.readline()
				if not line:
					break 
				words = line.replace('\r\n','').replace('\n','').split(spliter)
				uid = int(words[0])
				for i in range(len(words)-1):
					tokens = words[i+1].split(':')
					res[uid].append([int(tokens[0]),float(tokens[1])]) 
					
					
	def load_trainging_ratings(self, infile, spliter = '\t'):
		self.load_rating_file(infile,self.training_ratings_user, self.training_ratings_item, self.training_ratings_score, spliter)
	
	def load_test_ratings(self, infile, spliter = '\t'):
		self.load_rating_file(infile,self.test_ratings_user, self.test_ratings_item, self.test_ratings_score, spliter)
		
	def load_eval_ratings(self, infile, spliter = '\t'):
		self.load_rating_file(infile,self.eval_ratings_user, self.eval_ratings_item, self.eval_ratings_score, spliter)
	
	def load_rating_file(self,infile,rating_user, rating_item, rating_score,spliter):
		del rating_user[:]
		del rating_item[:]
		del rating_score[:]
		
		with open(infile,'r') as rd:
			while True:
				line = rd.readline()
				if not line:
					break 
				words = line.replace('\r\n','').replace('\n','').split(spliter)
				rating_user.append(int(words[0]))
				rating_item.append(int(words[1]))
				rating_score.append(float(words[2]))
		#print(rating_list)		
	
					
	def load_trainging_pairwise_ratings(self, infile, spliter = '\t'):
		self.load_pairwise_rating_file(infile,self.training_ratings_user, self.training_ratings_item, self.training_ratings_item02, self.training_ratings_score, spliter)
	
	def load_test_pairwise_ratings(self, infile, spliter = '\t'):
		self.load_pairwise_rating_file(infile,self.test_ratings_user, self.test_ratings_item, self.test_ratings_item02, self.test_ratings_score, spliter)

	def load_pairwise_rating_file(self,infile,rating_user,rating_item01,rating_item02,rating_score, spliter):
		del rating_user[:]
		del rating_item01[:]
		del rating_item02[:]
		del rating_score[:]
		
		
		with open(infile,'r') as rd:
			while True:
				line = rd.readline()
				if not line:
					break 
				words = line.replace('\r\n','').replace('\n','').split(spliter)
				rating_user.append(int(words[0]))
				rating_item01.append(int(words[1]))
				rating_item02.append(int(words[2]))
				rating_score.append(float(words[3]))
			
							

class dense_data_repos:
	def __init__(self, n_user, n_item, n_user_attr = 0, n_item_attr = 0):
		self.n_user = n_user 
		self.n_item = n_item 
		self.n_user_attr = n_user_attr 
		self.n_item_attr = n_item_attr 	
		self.user_attr = []
		self.item_attr = [] 	
		self.training_ratings = []
		self.test_ratings = [] 
	
	def load_user_attributes(self, infile,spliter='\t'):
		self.load_attributes(self.user_attr, self.n_user, self.n_user_attr, infile,spliter)
	
	def load_item_attributes(self, infile,spliter='\t'):
		self.load_attributes(self.item_attr, self.n_item, self.n_item_attr, infile,spliter)
	
	def load_attributes(self, res, n, m, infile,spliter):
		#res = [[0.0]*m for i in range(n)]
		del res[:]
		for i in range(n):
			res.append([0.0]*m) 
		
		with open(infile, 'r') as rd:
			while True:
				line = rd.readline()
				if not line:
					break 
				words = line.replace('\r\n','').replace('\n','').split(spliter)
				uid = int(words[0])
				for i in range(len(words)-1):
					tokens = words[i+1].split(':')
					res[uid][int(tokens[0])] = float(tokens[1])
					
	def load_trainging_ratings(self, infile, spliter = '\t'):
		self.load_rating_file(infile,self.training_ratings,spliter)
	
	def load_test_ratings(self, infile, spliter = '\t'):
		self.load_rating_file(infile, self.test_ratings, spliter)

	def load_rating_file(self,infile,rating_list,spliter):
		del rating_list[:]
		with open(infile,'r') as rd:
			while True:
				line = rd.readline()
				if not line:
					break 
				words = line.replace('\r\n','').replace('\n','').split(spliter)
				rating_list.append([int(words[0]),int(words[1]),float(words[2])])
		#print(rating_list)		


def load_rating_tsv(filename):
	'''
	res: [ [uid,iid,score], ... ]
	'''
	res = []
	with open(filename,'r') as rd:
		while True:
			line = rd.readline()
			if not line:
				break 
			words = line.replace('\r\n','').replace('\n','').split('\t')
			res.append([words[0],words[1],float(words[2])])
	return res

def load_content_tsv(filename):
	'''
	res: dict --> uid : [ [tag,value], ...]
	'''
	res = {}
	with open(filename,'r') as rd:
		while True:
			line = rd.readline()
			if not line:
				break 
			words = line.replace('\r\n','').replace('\n','').split('\t')
			res[words[0]]=[]
			for i in range(len(words)-1):
				tokens = words[i+1].split(':')
				res[words[0]].append([tokens[0],float(tokens[1])])
	return res


if __name__ == '__main__':
	pass