from sklearn.metrics import precision_recall_fscore_support
from gensim import corpora, models, similarities
import numpy as np
from pandas import *
import matplotlib.pyplot as plt
import pickle

class Evaluation(object):
	"""docstring for Evaluation"""

	def loadLdaModel(self, n):
		model = 'clustering/tmp/lda_n' + str(n) + '_training_corpus.lda'
		return models.LdaModel.load(model)

	def loadFromPickle(self, pickleFile):
		with open(pickleFile, 'rb') as f:
			return pickle.load(f)
		
	def eval(self):
		# Load training corpus ictionary of cid, vector
		train_map = self.loadFromPickle('clustering/tmp/train-corpus-map')
		
		# Build a training corpus sorted by id from 1 to 152
		corpus = []
		for c in range(1, 153):
			cid = 'c' + str(c)
			corpus.append(train_map.get(cid))
		print len(corpus)

		val_map = self.loadFromPickle('clustering/tmp/val-corpus-map')
		print len(val_map)
		grid = dict()
		
		numOfTopics = [100,200]
		for n in numOfTopics:
			model = self.loadLdaModel(n)

			# Save topics distribution in a file
			topics = model.show_topics(n, num_words=25, formatted=True)
			topics_tuple = model.show_topics(n, num_words=25, formatted=False)
			with open( 'clustering/tmp/' + str(n) + '_topics.txt', 'wb') as f1:
				for t in topics:
					f1.write("%s\n" % t.encode('utf-8'))
			with open( 'clustering/tmp/' + str(n) + '_topics', 'wb') as f2:
				pickle.dump(topics_tuple, f2)

			# Transform corpus to lda space and Build an index it - the order of document should be preserved
			# doc_n refer to c_n
			index = similarities.MatrixSimilarity(model[corpus])
			
			y_true = []
			y_pred =[]
			for cid, vector, _ in val_map:
				#print cid
				c = int(cid.strip('c'))
				vector_lda = model[vector]
				#print vector_lda
				sims = index[vector_lda]
				sims = sorted(enumerate(sims), key=lambda item: -item[1])
				y_true.append(c-1)
				y_pred.append(sims[0][0])

			# compute precision, recall, f-measure
			grid[n] = list()
			scores = precision_recall_fscore_support(y_true, y_pred, average='weighted')
			grid[n].append(scores[0])
			grid[n].append(scores[1])
			grid[n].append(scores[2])

		df = DataFrame(grid)
		df.to_csv('clustering/tmp/measures.csv')
		print df
		plt.figure(figsize=(14,8), dpi=120)
		plt.subplot(211)
		plt.plot(df.columns.values, df.iloc[2].values, '#007A99')
		plt.xticks(df.columns.values)
		plt.ylabel('F-measure', fontsize='large')
		plt.grid(True)

		plt.subplot(212)
		plt.plot(df.columns.values, df.iloc[1].values, 'b')
		plt.xticks(df.columns.values)
		plt.ylabel('Precision', fontsize='large')
		plt.xlabel("Number of topics", fontsize='large')
		plt.grid(True)

		plt.savefig('clustering/tmp/lda_topic_measures.png', bbox_inches='tight', pad_inches=0.1)

def main():
	e = Evaluation()
	e.eval()

if __name__ == '__main__':
	main()