# Author: Maiyaporn Phanich
# p.maiyaporn@gmail.com
# Fall 2014 Information Retrieval
# Final Project - Restaurant Categories Prediction

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
import numpy as np
import glob
import os
import codecs
import string
import logging
import pickle

class BuildCorpus(object):
	"""docstring for BuildCorpus"""
	def __init__(self):
		self.customwords = [i.encode('utf-8') for i in ["n't", "'ve", "'m", "'ll", "'re"]]
		self.stoplists = stopwords.words('english') + self.customwords
		self.lmtzr = WordNetLemmatizer()

	def isPunctuation(self, text):
		if type(text) is not str:
			text = text.encode('utf-8')
		text_ = text.translate(string.maketrans("",""), string.punctuation)
		return bool(len(text_)==0)

	def tokenizeDoc(self, doc):
		tokens = []
		for text in codecs.open(doc, "r", "utf-8"):
			tokens += self.preprocess(text)
		return tokens

	def preprocess(self, text):
		return [self.lmtzr.lemmatize(word) for word in word_tokenize(text.lower()) if len(word) > 3 and word not in self.stoplists and not self.isPunctuation(word)]

	def buildDictionary(self, directory, dictName):
		dictionary = corpora.Dictionary()
		for doc in glob.glob(directory + "/*"):
			dictionary.add_documents([self.tokenizeDoc(doc)])
		dictionary.filter_extremes(no_above=0.7)
		dictionary.compactify()
		dictionary.save(dictName)
		print (dictionary)
		return dictionary

	def dumpWithPickle(self, filename, data):
		with open(filename, 'wb') as f:
			pickle.dump(data, f)

	def buildCorpus(self, directory, dictName, corpusName, mapName):
		if os.path.isfile(dictName):
			dictionary = corpora.Dictionary.load(dictName)
		else:
			dictionary = self.buildDictionary(directory, dictName)

		corpus_class_map = dict()
		docno = 0
		corpus = []
		for doc in glob.glob(directory + "/*"):
			cid = doc.split("/")[-1]
			vector = dictionary.doc2bow(self.tokenizeDoc(doc))
			corpus.append(vector)
			corpus_class_map[docno] = cid
			docno += 1
		corpora.MmCorpus.serialize(corpusName, corpus)
		self.dumpWithPickle(mapName, corpus_class_map)
		print len(corpus)

	def buildCorpusByLineInDoc(self, directory, dictName, corpusName, mapName):
		dictionary = corpora.Dictionary.load(dictName)
		corpus_class_map = dict()
		docno = 0
		corpus = []
		for doc in glob.glob(directory + "/*"):
			cid = doc.split("/")[-1]
			with open(doc, 'rb') as f:
				texts = pickle.load(f)
				for text in texts:
					vector = dictionary.doc2bow(self.preprocess(text.decode('utf-8')))
					corpus.append(vector)
					corpus_class_map[docno] = cid
					docno += 1
		corpora.MmCorpus.serialize(corpusName, corpus)
		self.dumpWithPickle(mapName, corpus_class_map)
		print len(corpus)


def main():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	corpus = BuildCorpus()
	corpus.buildCorpus('c25/train', 'tmp/train-corpus.dict', 'tmp/train-corpus.mm', 'tmp/train-corpus-map')
	corpus.buildCorpusByLineInDoc('c25/validation', 'tmp/train-corpus.dict', 'tmp/val-corpus.mm', 'tmp/val-corpus-map')
	corpus.buildCorpusByLineInDoc('c25/test', 'tmp/train-corpus.dict', 'tmp/test-corpus.mm', 'tmp/test-corpus-map')

if __name__ == '__main__':
	main()