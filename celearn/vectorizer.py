# count vectorizer with no numpy
import pandas as pd
import re 

REGEX = re.compile('[^a-zA-Z]')

class CountVectorizer:
	"""Converts a corpus into a document term matrix"""

	def fit_transform(self, corpus):
		"""Main function in which the corpus -> document term matrix conversion occurs
		
		Parameters
		----------
		corpus: 2D list
		
		Returns
		-------
		phrases: 2D list
		"""
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
		"""Returns the unique words in the entire corpus that were used in creating the document term matrix
		
		Returns
		-------
		self._unique_words: list
		"""
		return self._unique_words

	def _clean(self, corpus):
		"""Cleans the corpus of any unwanted characters
		
		Parameters
		----------
		corpus: list
		
		Returns
		-------
		clean_corpus: list
		"""
		clean_corpus = []
		for doc in corpus:
			clean_corpus.append(REGEX.sub(' ', doc))
		return clean_corpus

class TFIDF:
	"""Converts a corpus into a term frequency X inverse document frequency matrix
	Mainly used for fitting text data into models
	"""

	def __init__(self, retval = 'default'):
		"""Initalizes how the TFIDF is to be returned
		
		Parameters
		----------
		retval: str
			if retval is "default" -> return dict
			if retval is "df" -> return pd.DataFrame
			if retval is "np" -> return np.array
		"""
		self.retval = retval

	def fit(self, corpus, labels):
		"""Main function in which the corpus -> TFIDF conversion occurs
		
		Parameters
		----------
		corpus: 2D list
		labels: list
		
		Returns
		-------
		tfidf: type is subject to retval variable
		"""
		corpus = [doc[0].lower() for doc in corpus]

		clean_corpus = self._clean(corpus)
		corpus_str = " ".join(clean_corpus).split(' ')
		self._unique_words = list(set(corpus_str))

		# document term
		dt_dict = {}
		for i, doc in enumerate(corpus):
			doc_split = doc.split()
			if labels[i][0] not in dt_dict:
				dt_dict[labels[i][0]] = {}
			for word in self._unique_words:
				try:
					dt_dict[labels[i][0]][word] += doc_split.count(word)
				except:
					dt_dict[labels[i][0]][word] = doc_split.count(word)

		# term frequency
		word_count = self._total_count(corpus)
		self._tfidf = {}
		for k in dt_dict:
			self._tfidf[k] = {}
			for word in dt_dict[k]:
				self._tfidf[k][word] = dt_dict[k][word] / word_count[word]
		if self.retval == 'df':
			return self.to_pandas()
		elif self.retval == 'np':
			return self.to_numpy()
		else:
			return self._tfidf

	def _total_count(self, corpus):
		"""Gets a count for each word in the corpus
		
		Parameters
		----------
		corpus: list
		
		Returns
		-------
		word_count: dict
		"""
		word_count = {}
		corpus = " ".join(corpus).split(' ')
		for word in self._unique_words:
				word_count[word] = corpus.count(word)
		return word_count

	def _clean(self, corpus):
		"""Cleans the corpus of any unwanted characters
		
		Parameters
		----------
		corpus: list
		
		Returns
		-------
		clean_corpus: list
		"""
		clean_corpus = []
		for doc in corpus:
			clean_corpus.append(REGEX.sub(' ', doc))
		return clean_corpus
	
	def to_numpy(self):
		"""Converts tfidf dict to numpy array
		
		Returns
		-------
		pd.DataFrame(self._tfidf).T.to_numpy(): np.array
		"""
		return pd.DataFrame(self._tfidf).T.to_numpy()

	def to_pandas(self):
		"""Converts tfidf dict to pandas DataFrame
		
		Returns
		-------
		pd.DataFrame(self._tfidf).T: pd.DataFrame
		"""
		return pd.DataFrame(self._tfidf).T
