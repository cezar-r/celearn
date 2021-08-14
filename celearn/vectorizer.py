# count vectorizer with no numpy
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
