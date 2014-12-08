import logging
from gensim import corpora, models, similarities
import numpy as np
from pandas import *
import matplotlib.pyplot as plt

def main():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	# load id->word mapping (the dictionary), one of the results of step 2 above
	dictionary = corpora.Dictionary.load('clustering/tmp/train-corpus.dict')
	# load corpus iterator
	traincorpus = corpora.MmCorpus('clustering/tmp/train-corpus.mm')
	valcorpus = corpora.MmCorpus('clustering/tmp/val-corpus.mm')
	#print valcorpus

	grid = dict()
	num_topics = np.arange(100,501,100)
	num_of_words = sum(cnt for doc in valcorpus for _, cnt in doc)
	for n in num_topics:
		grid[n] = list()
		model = models.ldamodel.LdaModel(traincorpus, id2word=dictionary, num_topics=n, update_every=0, passes=20)

		# model perplexity
		perplex = model.log_perplexity(valcorpus)
		print "Perplexity: %s" % perplex
		grid[n].append(perplex)

		perplex_bound = model.bound(valcorpus)
		per_word_perplex = np.exp2(-perplex_bound/num_of_words)
		print "Per-word Perplexity: %s" % per_word_perplex
		grid[n].append(per_word_perplex)
		model.save('clustering/tmp/' + 'lda_n' + str(n) + '_training_corpus.lda')

	df = DataFrame(grid)
	df.to_csv('clustering/tmp/perplexity.csv')
	print df
	plt.figure(figsize=(14,8), dpi=120)
	plt.subplot(211)
	plt.plot(df.columns.values, df.iloc[0].values, '#007A99')
	plt.xticks(df.columns.values)
	plt.ylabel('Perplexity')
	plt.grid(True)

	plt.subplot(212)
	plt.plot(df.columns.values, df.iloc[1].values, 'b')
	plt.xticks(df.columns.values)
	plt.ylabel('Perplexity')
	plt.xlabel("Number of topics", fontsize='large')
	plt.grid(True)

	plt.savefig('clustering/tmp/lda_topic_perplexity.png', bbox_inches='tight', pad_inches=0.1)

	
if __name__ == '__main__':
	main()
