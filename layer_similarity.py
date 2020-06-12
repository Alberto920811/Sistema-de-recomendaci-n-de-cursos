from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import CountVectorizer
from nltk.util import ngrams
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
from preprocess import stop_words
np.random.seed(2018)
#import nltk
#nltk.download('wordnet')

class similarity:
    File_class = []
    Courses = []
    Names = []

    def __init__(self, names, file_class, courses):
        self.File_class = file_class
        self.Courses = courses
        self.Names = names
    
    def infer(self):
        courses = [list(set(stop_words(item).remove())) 
                   for item in [w.split() for w in self.Courses]]
        classes = list(set(stop_words(self.File_class).remove()))

        dictionary = corpora.Dictionary(courses)
        feature_cnt = len(dictionary.token2id)
        corpus = [dictionary.doc2bow(text) for text in courses]
        tfidf = models.TfidfModel(corpus)        
        kw_vector = dictionary.doc2bow(classes)
        index = similarities.SparseMatrixSimilarity(tfidf[corpus], 
                                            num_features = feature_cnt)
        sim = index[tfidf[kw_vector]]

        course_rec = dict(zip(sim, self.Names))
        course_sort = sorted(course_rec.items(), reverse=True)
        
        lda_model = models.LdaMulticore(tfidf[corpus], 
                    num_topics=10, 
                    id2word=dictionary, 
                    passes=2, 
                    workers=2)
        
        for idx, topic in lda_model.print_topics(-1):
            print('Topic: {} \nWords: {}'.format(idx, topic))
        
        for index, score in sorted(lda_model[tfidf[kw_vector]], key=lambda tup: -1*tup[1]):
            print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
        
        return course_sort
    