import re
import pandas as pd

REGEX = re.compile('[^a-zA-Z]')

class TFIDF:

	def __init__(self, retval = 'default'):
		self.retval = retval

	def fit(self, corpus, labels):
		corpus = [doc[0].lower() for doc in corpus]

		clean_corpus = self._clean(corpus)
		corpus_str = " ".join(clean_corpus).split(' ')
		self._unique_words = list(set(corpus_str))

		# document term
		dt_dict = {}
		for i, doc in enumerate(corpus):
			if labels[i][0] not in dt_dict:
				dt_dict[labels[i][0]] = {}
			for word in self._unique_words:
				try:
					dt_dict[labels[i][0]][word] += doc.split(' ').count(word)
				except:
					dt_dict[labels[i][0]][word] = doc.split(' ').count(word)

		# term frequency
		word_count = self._total_count(corpus)
		tfidf = {}
		for k in dt_dict:
			tfidf[k] = {}
			for word in dt_dict[k]:
				tfidf[k][word] = dt_dict[k][word] / word_count[word]
		if self.retval == 'df':
			return pd.DataFrame(tfidf).T
		else:
			return tfidf


	def _total_count(self, corpus):
		word_count = {}
		corpus = " ".join(corpus).split(' ')
		for word in self._unique_words:
				word_count[word] = corpus.count(word)
		return word_count

	def _clean(self, corpus):
		clean_corpus = []
		for doc in corpus:
			clean_corpus.append(REGEX.sub(' ', doc))
		return clean_corpus







if __name__ == '__main__':

	corpus =[['This is a dog and this is a dog'],
			['This is a cat'],
			['This is a frog']]

	labels = [[0], 
		 [1],
		 [1]]


	tfidf = TFIDF(retval = 'df')
	df = tfidf.fit(corpus, labels)
	print(df)

