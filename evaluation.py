# Author: Maiyaporn Phanich
# p.maiyaporn@gmail.com
# Fall 2014 Information Retrieval
# Final Project - Restaurant Categories Prediction

from sklearn.metrics import precision_recall_fscore_support
from gensim import corpora, models, similarities, utils, matutils
import numpy as np
from pandas import *
import matplotlib.pyplot as plt
import pickle
import os
import numpy
import logging

class Evaluation(object):
	"""docstring for Evaluation"""
	def __init__(self, dictDir, trainMap, trainCorpus):
		# Load training corpus map of docno, cid
		self.dictionary = corpora.Dictionary.load(dictDir)
		self.train_map = self.loadFromPickle(trainMap)
		# Load training corpus for evaluation
		self.train_corpus = corpora.MmCorpus(trainCorpus)

	def loadLdaModel(self, n):
		model = 'tmp/lda_n' + str(n) + '_training_corpus.lda'
		return models.LdaModel.load(model)

	def loadFromPickle(self, pickleFile):
		with open(pickleFile, 'rb') as f:
			return pickle.load(f)
		
	def eval_cs(self, n, model, corpus_map, corpus):
		print "Current n...", n
		pathToIndex = 'tmp/index_' + str(n)
		if os.path.isfile(pathToIndex):
			index = similarities.Similarity.load(pathToIndex)
		else:
			# Transform corpus to lda space and Build an index it - the order of document should be preserved
			index = similarities.Similarity(pathToIndex, model[self.train_corpus], num_features=n)
			index.save(pathToIndex)

		index.num_best = 1
		y_true = []
		y_pred =[]
		for i, doc in enumerate(model[corpus]):	
			cid_v = corpus_map.get(i)
			sims = index[doc]
			if len(sims) > 0:
				cid_s = self.train_map.get(sims[0][0])
				y_true.append(cid_v)
				y_pred.append(cid_s)

		# compute precision, recall, f-measure
		scores = precision_recall_fscore_support(y_true, y_pred, average='weighted')
		print '######### weighted scores #########'
		print scores
		return scores

	def hellinger(self, vec1, vec2):
		return numpy.sqrt(0.5 * ((numpy.sqrt(vec1) - vec2)**2).sum())

	def eval(self, ns, corpusMapDir, corpusDir, sim_method='cs'):
		# Load a map of docno and category id
		corpus_map = self.loadFromPickle(corpusMapDir)
		# Load corpus for evaluation
		corpus = corpora.MmCorpus(corpusDir)
		print len(corpus), len(corpus_map)

		grid = dict()
		for n in ns:
			model = self.loadLdaModel(n)

			grid[n] = list()
			if sim_method == 'cs':
				scores = self.eval_cs(n, model, corpus_map, corpus)
			elif sim_method == 'hl':
				scores = self.eval_hl(n, model, corpus_map, corpus)

			grid[n].append(scores[0])
			grid[n].append(scores[1])
			grid[n].append(scores[2])

		if len(grid) > 1:
			self.plotGraph(grid, sim_method)

	def eval_hl(self, n, model, corpus_map, corpus):
		print "Current n...", n
		train_lda = model[self.train_corpus]

		y_true = []
		y_pred = []
		precomputed_vec2s = numpy.sqrt(matutils.corpus2dense(train_lda, num_terms=n, num_docs=len(train_lda)).transpose())
		for vid, vdoc in enumerate(model[corpus]):
			cid_v = corpus_map.get(vid)
			sims = [(tid, self.hellinger(matutils.sparse2full(vdoc, n), tdoc)) for tid, tdoc in enumerate(precomputed_vec2s)]
			sims = sorted(sims, key=lambda item: item[1])
			cid_s = self.train_map.get(sims[0][0])
			y_true.append(cid_v)
			y_pred.append(cid_s)

		# compute precision, recall, f-measure
		scores = precision_recall_fscore_support(y_true, y_pred, average='weighted')
		print '######### weighted scores #########'
		print scores
		return scores

	def printClassScore(self, scores):
		print '######### class scores #########'
		print scores
		ps = scores[0]
		rs = scores[1]
		fs = scores[2]
		for i in np.arange(0,25,1):
			print str(i) + ': ', ps[i], rs[i], fs[i]

	def plotGraph(self, grid, sim_method):
		df = DataFrame(grid)
		df.to_csv('tmp/measures_' + sim_method + '.csv')
		print df
		plt.figure(figsize=(14,12), dpi=120)
		plt.subplot(311)
		plt.plot(df.columns.values, df.iloc[2].values, '#007A99')
		plt.xticks(df.columns.values)
		plt.ylabel('F-measure', fontsize='large')
		plt.grid(True)

		plt.subplot(312)
		plt.plot(df.columns.values, df.iloc[0].values, 'b')
		plt.xticks(df.columns.values)
		plt.ylabel('Precision', fontsize='large')
		plt.xlabel("Number of topics", fontsize='large')
		plt.grid(True)

		plt.subplot(313)
		plt.plot(df.columns.values, df.iloc[1].values, 'b')
		plt.xticks(df.columns.values)
		plt.ylabel('Recall', fontsize='large')
		plt.xlabel("Number of topics", fontsize='large')
		plt.grid(True)

		plt.savefig('tmp/val_lda_topic_measures_' + sim_method + '.png', bbox_inches='tight', pad_inches=0.1)
		
	def getTopics(self,n):
		model = self.loadLdaModel(n)
		topics_tuple = model.show_topics(n, num_words=100, formatted=False)
		with open( 'tmp/' + str(n) + '_topics', 'wb') as f:
			pickle.dump(topics_tuple, f)

	def getDocumentTopics(self, n):
		train_map = self.loadFromPickle('tmp/train-corpus-map')
		train_corpus = corpora.MmCorpus('tmp/train-corpus.mm')

		model = self.loadLdaModel(n)
		train_lda = model[train_corpus]

		f1 = open('tmp/' + str(n) + '_categories-topics.txt', 'wb')
		for i, doc in enumerate(train_lda):
			f1.write( str(train_map.get(i)) + ',' + str(sorted(doc, key=lambda item: -(item[1]))) + '\n')
			#print train_map.get(i), sorted(doc, key=lambda item: -(item[1]))

	def tfidf_eval(self, corpusMapDir, corpusDir, sim_method):
		# Load a map of docno and category id
		corpus_map = self.loadFromPickle(corpusMapDir)
		# Load corpus for evaluation
		corpus = corpora.MmCorpus(corpusDir)
		print len(corpus), len(corpus_map)

		tfidf = models.TfidfModel(self.train_corpus)
		y_true = []
		y_pred = []
		if sim_method == 'cs':
			pathToIndex = 'tmp/index_tfidf'
			if os.path.isfile(pathToIndex):
				index = similarities.Similarity.load(pathToIndex)
			else:
				# Transform corpus to tf-idf space and Build an index it - the order of document should be preserved
				index = similarities.Similarity(pathToIndex, tfidf[self.train_corpus], num_features=len(self.dictionary))
				index.save(pathToIndex)

			index.num_best = 1
			for i, doc in enumerate(tfidf[corpus]):
				cid_v = corpus_map.get(i)
				sims = index[doc]
				if len(sims) > 0:
					cid_s = self.train_map.get(sims[0][0])
					y_true.append(cid_v)
					y_pred.append(cid_s)
		elif sim_method == 'hl':
			precomputed_vec2s = numpy.sqrt(matutils.corpus2dense(tfidf[self.train_corpus], num_terms=len(self.dictionary), num_docs=len(self.train_corpus)).transpose())	
			for vid, vdoc in enumerate(tfidf[corpus]):
				#print vid
				cid_v = corpus_map.get(vid)
				sims = [(tid, self.hellinger(matutils.sparse2full(vdoc, len(self.dictionary)), tdoc)) for tid, tdoc in enumerate(precomputed_vec2s)]
				sims = sorted(sims, key=lambda item: -(item[1]))
				cid_s = self.train_map.get(sims[0][0])
				y_true.append(cid_v)
				y_pred.append(cid_s)

		# compute precision, recall, f-measure
		scores = precision_recall_fscore_support(y_true, y_pred, average='weighted')
		print scores

def main():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	e = Evaluation('tmp/train-corpus.dict', 'tmp/train-corpus-map', 'tmp/train-corpus.mm')

	# Evaluate models to find the best k value from k = 50 to 700
	num_of_topics = np.arange(50,701,50)
	e.eval(num_of_topics, 'tmp/val-corpus-map', 'tmp/val-corpus.mm', 'cs')
	e.eval(num_of_topics, 'tmp/val-corpus-map', 'tmp/val-corpus.mm', 'hl')
	
	# Get scores for k=350
	e.getTopics(350)
	e.getDocumentTopics(350)
	e.eval([350], 'tmp/test-corpus-map', 'tmp/test-corpus.mm', 'cs')
	e.eval([350], 'tmp/test-corpus-map', 'tmp/test-corpus.mm', 'hl')
	

	# Evaluate tf-idf model
	e.tfidf_eval('tmp/test-corpus-map', 'tmp/test-corpus.mm', 'cs')
	e.tfidf_eval('tmp/test-corpus-map', 'tmp/test-corpus.mm', 'hl')

if __name__ == '__main__':
	main()