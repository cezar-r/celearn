# count vectorizer with no numpy
import pandas as pd
import re 

REGEX = re.compile('[^a-zA-Z]')

class CountVectorizer:

	def fit_transform(self, corpus):
		corpus = [doc[0].lower() for doc in corpus]

		clean_corpus = self._clean(corpus)
		corpus_str = " ".join(clean_corpus).split(' ')
		self._unique_words = list(set(corpus_str))
		phrases = []
		for doc in corpus:
			token_phrase = []
			for word in self._unique_words:
				doc_split = doc.split()
				token_phrase.append(doc_split.count(word))
			phrases.append(token_phrase)
		return phrases
	
	def get_feature_names(self):
		return self._unique_words

	def _clean(self, corpus):
		clean_corpus = []
		for doc in corpus:
			clean_corpus.append(REGEX.sub(' ', doc))
		return clean_corpus

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
	
	def to_numpy(self):
		return pd.DataFrame(self.tfidf).T.to_numpy()

	def to_pandas(self):
		return pd.DataFrame(self.tfidf).T
