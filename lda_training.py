# Author: Maiyaporn Phanich
# p.maiyaporn@gmail.com
# Fall 2014 Information Retrieval
# Final Project - Restaurant Categories Prediction

import logging
from gensim import corpora, models, similarities
import numpy as np
from pandas import *
import matplotlib.pyplot as plt

class LDATraining(object):
	"""docstring for LDATraining"""
	def __init__(self, dictDir, trainDir, valDir):
		self.dictionary = corpora.Dictionary.load(dictDir)
		self.train_corpus = corpora.MmCorpus(trainDir)
		self.val_corpus = corpora.MmCorpus(valDir)
		self.num_of_words = sum(cnt for doc in self.val_corpus for _, cnt in doc)
	
	def train(self, n, i):
		model = models.ldamodel.LdaModel(self.train_corpus, id2word=self.dictionary, num_topics=n, update_every=0, passes=i)
		model.save('tmp/' + 'lda_n' + str(n) + '_training_corpus.lda')

		# model perplexity
		perplex = model.log_perplexity(self.val_corpus)
		print "Perplexity: %s" % perplex

		perplex_bound = model.bound(self.val_corpus)
		per_word_perplex = np.exp2(-perplex_bound/self.num_of_words)
		print "Per-word Perplexity: %s" % per_word_perplex

		return (perplex, per_word_perplex)

	def train_ns(self, num_topics, i):
		grid = dict()
		for n in num_topics:
			perplex, per_word_perplex = self.train(n, i)
			grid[n] = list()
			grid[n].append(perplex)
			grid[n].append(per_word_perplex)
		
		self.plotGraph(grid)

	def plotGraph(self, grid):
		df = DataFrame(grid)
		print df

		plt.figure(figsize=(14,8), dpi=120)
		plt.subplot(211)
		plt.plot(df.columns.values, df.iloc[0].values, '#007A99')
		plt.xticks(df.columns.values)
		plt.ylabel('Perplexity', fontsize='large')
		plt.grid(True)

		plt.subplot(212)
		plt.plot(df.columns.values, df.iloc[1].values, 'b')
		plt.xticks(df.columns.values)
		plt.ylabel('Perplexity', fontsize='large')
		plt.xlabel("Number of topics", fontsize='large')
		plt.grid(True)

		df.to_csv('tmp/perplexity.csv')
		plt.savefig('tmp/lda_topic_perplexity.png', bbox_inches='tight', pad_inches=0.1)

def main():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	training = LDATraining('tmp/train-corpus.dict', 'tmp/train-corpus.mm', 'tmp/val-corpus.mm')
	num_topics = np.arange(50,701,50)
	training.train_ns(num_topics, 20)
	
if __name__ == '__main__':
	main()
