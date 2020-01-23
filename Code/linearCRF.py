
# coding: utf-8


from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
import sys
import traceback
import pystruct
from pystruct.models import ChainCRF
from pystruct.models import GraphCRF
from pystruct.learners import FrankWolfeSSVM
import re
import datetime
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
nltk.download('punkt')
from sklearn import preprocessing
from numpy import dot
from numpy.linalg import norm
import tensorflow as tf
import tensorflow_hub as hub
import os
from tensorflow import Graph
import scipy.stats
import string
import itertools
from itertools import chain
import spicy
from scipy import spatial
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score, jaccard_similarity_score, roc_curve, roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.metrics import average_precision_score,precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature


import nltk, string
# plt.style.use('ggplot')
import sklearn_crfsuite

from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics as crf_metrics
from sklearn.manifold import TSNE
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector





# ------------------------------------------------------------------
# cmv_threads column index (starting from 0)
TH_ID_INDEX = 0
TH_TITLE_INDEX = 1
TH_TEXT_INDEX = 4
TH_AUTHOR_INDEX = 5
TH_LENGTH_INDEX = 6

# cmv_comments column index (starting from 0)
THREAD_ID_INDEX = 0
COMMENT_ID_INDEX = 2
REPLY_TO_INDEX = 3
AUTHOR_ID_INDEX = 5
CONTENT_INDEX = 8  # text content
DELTA_EARN_INDEX = 9
FLAIR_INDEX = 6
SENTIMENT_INDEX = 16
COMMENT_ORDER_INDEX = 1

HEDGE_INDEX = 23
ASSERTIVES_INDEX = 17
FACTIVES_INDEX =  18
IMPLICATIVES_INDEX = 19 
REPORT_VERBS_INDEX = 20
OPINION_NEG_INDEX = 21
OPINION_POS_INDEX = 22
STRONG_SUBJ_INDEX = 24
WEAK_SUBJ_INDEX = 25

# parameters
THRESHOLD_COS = 0.5
TFIDF_MAX_FEATURES = 100

HAS_DELTA_ONLY = False

FEATURE_NUM = 29 #57

pickle_overwrite = False

FEED_PAIRED = False

test_name = "linear_thread"

prefix_pkl = "data_cmv/pkl/"

file_raw_data = prefix_pkl + test_name + "_raw"
file_dataset = prefix_pkl + test_name + "_dataset"
file_model = prefix_pkl + test_name + "_crf"
file_lp_prob = prefix_pkl + test_name + "_result_prob"
file_encoded_data = prefix_pkl + test_name + "_encoded_data"





# variables
data_set = []
data_label = []

train_set = []
train_label = []

test_set = []
test_label = []

delta_record_cnt = 0
nondelta_record_cnt = 0

cnt_ignored_seq = 0
cnt_thread = 0
cnt_train_subseq = 0
cnt_test_subseq = 0





# using tensorflow USE for word embeddings

USE_URL = 'https://tfhub.dev/google/universal-sentence-encoder/2'
cachedir = 'c6f5954ffa065cdb2f2e604e740e8838bf21a2d3'

class USE(object):
    def __init__(self, model_url=USE_URL):
        if "TFHUB_CACHE_DIR" in os.environ:
            tfdata = os.environ['TFHUB_CACHE_DIR']
            model_url= tfdata + "/" + cachedir
#             model_url = tfdata
            print(tfdata)
        else:
            print("TFHUB_CACHE_DIR=None")

        graph = Graph()
        with graph.as_default():
            embed = hub.Module(model_url)
            self.sentences = tf.placeholder(dtype=tf.string, shape=[None])
            self.encoded_text = tf.cast(embed(self.sentences), tf.float32)
            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        graph.finalize()

        self.session = tf.Session(graph=graph)
        self.session.run(init_op)

    def encode(self, sentences):
        return self.session.run(self.encoded_text, feed_dict={self.sentences: sentences})






def form_lexicon(filename):
    word_list = []
    with open(filename, 'r', encoding='mac_roman', newline='') as f:
        content = f.readlines()
        for row in content:
            if len(row.strip())==0 or row[0][0] == "#":
                continue
            word_list.append(row.strip())
    return word_list
            
hedges = form_lexicon('data_cmv/input/hedge_list.txt')

assertives = form_lexicon('data_cmv/input/assertives_hooper1975.txt')

factives = form_lexicon('data_cmv/input/factives_hooper1975.txt')

implicatives = form_lexicon('data_cmv/input/implicatives_karttunen1971.txt')

opinion_neg = form_lexicon('data_cmv/input/opinion_negative-words.txt')

opinion_pos = form_lexicon('data_cmv/input/opinion_positive-words.txt')

report_verbs = form_lexicon('data_cmv/input/report_verbs.txt')


def print_time(*args):
    text = ""
    for arg in args:
        text += arg
    print("[",datetime.datetime.now(),"]", text)


def curtime():    
    currentDT = datetime.datetime.now()
    return currentDT.strftime("%Y-%m-%d_%H%M%S")


def pickle_save(filename, data2pkl):
    global pickle_overwrite
    if pickle_overwrite == True:
        fileObject = open(filename,'wb') 
        pickle.dump(data2pkl,fileObject)
        fileObject.close()

        
def pickle_load(filename):
    fileObject = open(filename,'rb') 
    data_unpkl = pickle.load(fileObject)
    fileObject.close()
    return data_unpkl


print_time("Program begins.")





# ------------------------------------------------------------------
# main entrance of the project
# ------------------------------------------------------------------
threads_reader = csv.reader(open("data_cmv/cmv_threads.csv", newline='', encoding='mac_roman'), delimiter=',',
                        quotechar='"')


threads_dict = {}
    
try:
    for row in threads_reader:
        if row[0] != "thread_id":
            threads_dict[row[0]] = row
        
except Exception as ex:
    sys.stderr.write('Exception\n')
    extype, exvalue, extrace = sys.exc_info()
    traceback.print_exception(extype, exvalue, extrace)





# ------------------------------------------------------------------
# # Extract features
# import spacy
# nlp = spacy.load('en')

class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class ColumnExtractor(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, seq):
        column = np.recarray(shape=(len(seq),),
                             dtype=[('seq', object), ('content', object)])

        column['seq'] = seq
        seq_array = np.array(seq)

        column['content'] = seq_array[:, CONTENT_INDEX]
        return column

def RemovePuncMap():
    return dict((ord(char), None) for char in string.punctuation)


def SimilarityBetween(text1, text2):   
    # nltk.download('punkt') # if necessary...
#     stemmer = nltk.stem.porter.PorterStemmer()
    vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')
    return cosine_sim(text1, text2, vectorizer)

def SimilarityWithRecentPosts(entry, recent_posts):
    simi = []
    for post in recent_posts:
        simi.append(SimilarityBetween(entry, post))
    
    return sum(l)/len(l) if len(l) != 0 else 0
    

def stem_tokens(tokens):
    stemmer = nltk.stem.porter.PorterStemmer()
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(RemovePuncMap())))


def cosine_sim(text1, text2, vectorizer):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


def HedgeUsage(text):
    return sum(map(text.lower().count, hedges))


def FirstPersonPronUsage(text_token):
    pron_1st = ["i", "me", "we", "us", "my", "mine"]
#     return sum(map(text.lower().count, pron_1st))
    return sum(el in pron_1st for el in text_token )


def SecondPersonPronUsage(text_token):
    pron_2nd = ["you", "your", "yours"]
#     return sum(map(text.lower().count, pron_2nd))
    return sum(el in pron_2nd for el in text_token )


def SecondPersonPronUsageInSent(sent_token):
    pron_2nd = ["you", "your", "yours"]
    cnt_sent = 0
    for sent in sent_token:
        if sum(map(sent.count, pron_2nd)) > 0:
            cnt_sent += 1
    return cnt_sent


def RecentSimiCommentSenti(entry, oh_reply_dict):
    if len(oh_reply_dict) == 0:
        return 0,0
    
    d = {}
   
    for key, value in oh_reply_dict.items():
        if value[2] < entry[COMMENT_ORDER_INDEX]:
            d[key] = value
        
    if len(d) == 0:
        return 0,0
    
    arr = sorted(d.items(), key=lambda x: x[1][1])

    return pair_simi(entry[CONTENT_INDEX], arr[0][0]), pair_simi(entry[CONTENT_INDEX],arr[len(arr)-1][0])


def pair_simi(com_1, com_2):
    if com_1 == "" or com_2 == "":
        return 0
    
    token_1 = encoder.encode(nltk.sent_tokenize(com_1))
    token_2 = encoder.encode(nltk.sent_tokenize(com_2))
    
    max_simi = 0
    for x,y in itertools.product(token_1,token_2):
        simi = calculate_similarity(x,y)
        if simi > max_simi and simi != 1:
            max_simi = simi     
            
    return max_simi
    

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def form_labels(seq):
    labels = []
    for entry in seq:
        if entry[DELTA_EARN_INDEX] == "1":
            labels.append('1')
        else:
            labels.append('0')

    return labels





def form_edges(seq, cos):
    edges_linear = np.vstack([np.arange(len(seq)-1), np.arange(1, len(seq))])
    edges_skip = []
    #edges_skip = np.array([[0, 2], [1, 2]])

    docs = []
    # ---------------------------------------------------
    # option 1: only if similarity between two comments > threshold
    for i in range(1, len(seq)):
        for j in range(i+2, len(seq)):
            # if cosine similarity > THRESHOLD_COS, form an edge
            # print(cos[i,j], i, j)            
            # form skip edges using TF-IDF cos similarity           
            if cos[i,j] > THRESHOLD_COS:
                edges_skip.append([i-1, j-1])
#             # form skip edges using direct response
#             elif seq[j][REPLY_TO_INDEX] == seq[i][COMMENT_ID_INDEX]:
# #                 print("direct response")
#                 edges_skip.append([i-1, j-1]) 
#             elif seq[j][AUTHOR_ID_INDEX] == seq[i][AUTHOR_ID_INDEX]:
# #                 print("same author")
#                 edges_skip.append([i-1, j-1]) 

    # ---------------------------------------------------
        
    edges_skip = np.array(edges_skip)
    
    if len(edges_skip) > 0:
        edges = np.vstack([edges_linear.T, edges_skip]).T
    else:
        edges = edges_linear
    # print(edges)
    return edges.T





# ------------------------------------------------------------------
# for linear CRF thread structure
# ------------------------------------------------------------------

# for each thread, form a feature vector
def form_feature_dicts(seq, sent_seq, cos, oh_reply_dict):
    recent_posts = []
    cur_thread = threads_dict[seq[0][THREAD_ID_INDEX]]

    # get recent posts from the opinion holder
    holder_post_cnt = 0
    comment_id_list = []
    for entry in seq:
        comment_id_list.append(entry[COMMENT_ID_INDEX])
        if entry[AUTHOR_ID_INDEX] == cur_thread[TH_AUTHOR_INDEX]:
            recent_posts.append(entry[CONTENT_INDEX])
            holder_post_cnt += 1

    features_list = []
    for i, entry in enumerate(seq):
        word_cnt_entry = len(re.split(r'[^0-9A-Za-z]+',entry[CONTENT_INDEX]))
        word_token = re.split(r'[^0-9A-Za-z]+',entry[CONTENT_INDEX].lower())
        sent_token = sent_seq[i+1]     
        
        most_neg_simi, most_pos_simi = RecentSimiCommentSenti(entry, oh_reply_dict)
        
        if FEED_PAIRED:
            simi_init = cos[0][i+1]
            simi_recent = 0
        else:
            simi_init = cos[0][i+1]
            simi_recent = cos[i+1].T[0:holder_post_cnt].mean() if holder_post_cnt > 0 and i >= holder_post_cnt else 0    
        
        feature_dict = {
         # ---------------- linguistic features ----------------
        'num_words': word_cnt_entry,
        'length_comment': len(entry[CONTENT_INDEX]),
        'num_sentences': entry[CONTENT_INDEX].count('.'),  
        'hedge': int(entry[HEDGE_INDEX])*1.0/word_cnt_entry,
          # from the 2013 ACL - Bias paper
        'assertives': int(entry[ASSERTIVES_INDEX])*1.0/word_cnt_entry,
        'factives': int(entry[FACTIVES_INDEX])*1.0/word_cnt_entry,
        'implicatives': int(entry[IMPLICATIVES_INDEX])*1.0/word_cnt_entry,
        'report_verbs': int(entry[REPORT_VERBS_INDEX])*1.0/word_cnt_entry,
        'opinion_neg': int(entry[OPINION_NEG_INDEX])*1.0/word_cnt_entry,
        'opinion_pos': int(entry[OPINION_POS_INDEX])*1.0/word_cnt_entry, 
        'strong_subj': int(entry[STRONG_SUBJ_INDEX])*1.0/word_cnt_entry,
        'weak_subj': int(entry[WEAK_SUBJ_INDEX])*1.0/word_cnt_entry,
         # word category features from [Tan2016WWW]
        'definite_articles': entry[CONTENT_INDEX].lower().count('the')*1.0/word_cnt_entry,
        'indefinite_articles': sum(entry[CONTENT_INDEX].lower().count(x) for x in ("a", "an"))*1.0/word_cnt_entry,        
        "1st_pron": FirstPersonPronUsage(word_token)*1.0/word_cnt_entry,
        "2nd_pron": SecondPersonPronUsage(word_token)*1.0/word_cnt_entry,
        "2nd_pron_sent":  SecondPersonPronUsageInSent(sent_token)*1.0/len(sent_token) if len(sent_token) > 0 else 0,
        'sentiment': float(re.sub('[^0-9]', '', entry[SENTIMENT_INDEX])) if (
            re.sub('[^0-9]', '', entry[SENTIMENT_INDEX]).strip()) else 0,
        'num_questions': entry[CONTENT_INDEX].count('?'),   
        'examples':  sum(entry[CONTENT_INDEX].lower().count(x) for x in ("for example", "for instance", "e.g", "eg")),
        'links': 1 if "http" in entry[CONTENT_INDEX] else 0,

        # ---------------- context-based features ----------------
        'order_in_discussion': (int(entry[COMMENT_ORDER_INDEX])+1.0)/int(cur_thread[TH_LENGTH_INDEX]),
        'sub_order': i + 1,
        'relative_order': (i + 1) * 1.0 / len(seq),
        'user_flair': int(re.sub('[^0-9]', '', entry[FLAIR_INDEX])) if (
            re.sub('[^0-9]', '', entry[FLAIR_INDEX]).strip()) else 0,
        'authentication': 1 if entry[AUTHOR_ID_INDEX] == cur_thread[TH_AUTHOR_INDEX] else 0,

        # ---------------- interaction-based features ----------------
        # similarities with the initial post
        'simi_initial':simi_init,
        # similarity with the recent opinion holder's post  
        'simi_recent': cos[i+1].T[0:holder_post_cnt].mean() if holder_post_cnt > 0 \
        and i >= holder_post_cnt 
        else 0,            
        'direct_response': (comment_id_list.index(entry[REPLY_TO_INDEX])+1.0)/len(comment_id_list) if entry[REPLY_TO_INDEX] in comment_id_list
        else 0,
        'quotation': 1 if "&gt;" in entry[CONTENT_INDEX] else 0 ,
        'most_neg_simi': most_neg_simi,
        'most_pos_simi': most_pos_simi    
        }
        features_list.append(feature_dict)
  
    return features_list





# ------------------------------------------------------------------
# for skip-chain CRF and edge feature CRF
# ------------------------------------------------------------------
# Extract features

# for each thread, form a feature vector
def form_features_vec(seq,cos):
    features = []
    feature_unioned = Pipeline([
        ('extractColumns', ColumnExtractor()),
        # Use FeatureUnion to combine the features from subject and body
        ('union', FeatureUnion(
            transformer_list=[
                ('seq_stats', Pipeline([
                    ('select', ItemSelector(key='seq')),
                    ('stats', ProduceStatFeatureDict(cos=cos)),  # returns a list of dicts
                    ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                ])),
            ],
            transformer_weights={
                'seq_stats': 1,
            },
        )
         )
    ])

    output = feature_unioned.fit_transform(seq)

    return output.toarray()


class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class ColumnExtractor(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, seq):
        column = np.recarray(shape=(len(seq),),
                               dtype=[('seq', object), ('content', object)])

        column['seq'] = seq
        seq_array = np.array(seq)

        column['content'] = seq_array[:, CONTENT_INDEX]

        return column


class ProduceStatFeatureDict(BaseEstimator, TransformerMixin):
    def __init__(self, cos):
        self.cos = cos
    
    def fit(self, x, y=None):
        return self

    def transform(self, seq):
        # get recent posts from the opinion holder
        recent_posts = []
        cur_thread = threads_dict[seq[0][THREAD_ID_INDEX]]
        
        # get recent posts from the opinion holder
        features_list = []
        for i, entry in enumerate(seq):
            word_cnt_entry = len(re.split(r'[^0-9A-Za-z]+',entry[CONTENT_INDEX]))
            word_token = re.split(r'[^0-9A-Za-z]+',entry[CONTENT_INDEX].lower())

            if FEED_PAIRED:
                simi_init = self.cos[0][i+1]
#                 simi_recent = 0
            else:
                simi_init = self.cos[0][i+1]

            feature_dict = {
            # ---------------- linguistic features ----------------
            'num_words': word_cnt_entry,
            'length_comment': len(entry[CONTENT_INDEX]),
            'num_sentences': entry[CONTENT_INDEX].count('.'),  
            'hedge': int(entry[HEDGE_INDEX])*1.0/word_cnt_entry,
              # from the 2013 ACL - Bias paper
            'assertives': int(entry[ASSERTIVES_INDEX])*1.0/word_cnt_entry,
            'factives': int(entry[FACTIVES_INDEX])*1.0/word_cnt_entry,
            'implicatives': int(entry[IMPLICATIVES_INDEX])*1.0/word_cnt_entry,
            'report_verbs': int(entry[REPORT_VERBS_INDEX])*1.0/word_cnt_entry,
            'opinion_neg': int(entry[OPINION_NEG_INDEX])*1.0/word_cnt_entry,
            'opinion_pos': int(entry[OPINION_POS_INDEX])*1.0/word_cnt_entry, 
            'strong_subj': int(entry[STRONG_SUBJ_INDEX])*1.0/word_cnt_entry,
            'weak_subj': int(entry[WEAK_SUBJ_INDEX])*1.0/word_cnt_entry,
             # word category features from [Tan2016WWW]
            'definite_articles': entry[CONTENT_INDEX].lower().count('the')*1.0/word_cnt_entry,
            'indefinite_articles': sum(entry[CONTENT_INDEX].lower().count(x) for x in ("a", "an"))*1.0/word_cnt_entry,        
            "1st_pron": FirstPersonPronUsage(word_token)*1.0/word_cnt_entry,
            "2nd_pron": SecondPersonPronUsage(word_token)*1.0/word_cnt_entry,
            'sentiment': float(re.sub('[^0-9]', '', entry[SENTIMENT_INDEX])) if (
                re.sub('[^0-9]', '', entry[SENTIMENT_INDEX]).strip()) else 0,
            'num_questions': entry[CONTENT_INDEX].count('?'),     
            'examples':  sum(entry[CONTENT_INDEX].lower().count(x) for x in ("for example", "for instance", "e.g", "eg")),
            'links': 1 if "http" in entry[CONTENT_INDEX] else 0,

            # ---------------- context-based features ----------------
            'order': i + 1,
            'length_discussion': len(seq),
            'relative_order': (i + 1) * 1.0 / len(seq),
            'user_flair': int(re.sub('[^0-9]', '', entry[FLAIR_INDEX])) if (
                re.sub('[^0-9]', '', entry[FLAIR_INDEX]).strip()) else 0,
            'authentication': 1 if entry[AUTHOR_ID_INDEX] == cur_thread[TH_AUTHOR_INDEX] else 0,
            # ---------------- interaction-based features ----------------
             # similarities with the initial post
            'simi_initial':simi_init,
            'quotation': 1 if "&gt;" in entry[CONTENT_INDEX] else 0  
            }
            features_list.append(feature_dict)
  
        return features_list





# ------------------------------------------------------------------
# for edge feature CRF thread structure
# ------------------------------------------------------------------

def form_edge_features(features, edges):
#     print("-------- form_edge_features, node feature --------")
#     print(features[0])
#     print("-------- form_edge_features, edges --------")
#     print(edges)

    edge_features = []
    # each edge feature is a numpy array
    for pair in edges:
        node1 = pair[0] # index of the nodes in features
        node2 = pair[1]
        edge_feature = []
        edge_feature.append(features[node1][0])
        edge_feature.append(features[node1][22])
        edge_feature.append(features[node1][24])
        
        edge_feature_vec = np.array(edge_feature)
        edge_features.append(edge_feature_vec)
    
#     print(edge_features)
#     print(np.array(edge_features))
    return np.array(edge_features)






def calculate_similarity(vec1,vec2):
#     cos[i,j] = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    simi = 1 - spatial.distance.cosine(vec1, vec2) 
    return simi
    

def sent_encode(text):
    return nltk.sent_tokenize(text)
 

# def USE_simi(docs,cur_thread,thread_op_encoded):
def USE_simi(seq,docs,cur_thread):
    sentences = []
    encoded_sentences = []
    thread_author = cur_thread[TH_AUTHOR_INDEX]    

    # encode each comments, including texts from opinion holder. docs could be seq or subseq
    for i,para in enumerate(docs):
        sent_token = nltk.sent_tokenize(para)
        sentences.append(sent_token)
        if para == "":
            encoded_sentences.append(np.zeros(len(encoded_sentences[0])))
        else:
            encoded_sentences.append(encoder.encode(sent_token))
        
    if len(encoded_sentences) == 0:
        print("Ah!! empty docs")
        

    cos_simi = np.zeros((len(encoded_sentences),len(encoded_sentences)))
    
    if FEED_PAIRED:
        # only need to calculate cos with op
        comm_sent_i = encoded_sentences[0]
        for j in range(1,len(encoded_sentences)):
            comm_sent_j = encoded_sentences[j]
            max_simi = 0
            for x,y in itertools.product(comm_sent_i,comm_sent_j):
                simi = calculate_similarity(x,y)
                if simi>max_simi and simi != 1:
                    max_simi = simi                
            cos_simi[0][j] = max_simi

    else:
        # calculate cos_simi
        for i,comm_sent_i in enumerate(encoded_sentences):
            for j in range(i,len(encoded_sentences)):
                comm_sent_j = encoded_sentences[j]
                # in pairwise, compare sentences in comment i and comment j,
                # calculate max(similarity)            
                if i==j:
                    cos_simi[i][j] = 1
                elif seq[i-1][AUTHOR_ID_INDEX] != thread_author and seq[j-1][AUTHOR_ID_INDEX] != thread_author:
                    cos_simi[i][j] = 0
                    cos_simi[j][i] = 0
                else:
                    max_simi = 0
                    for x,y in itertools.product(comm_sent_i,comm_sent_j):
                        simi = calculate_similarity(x,y)
                        if simi>max_simi and simi != 1:
                            max_simi = simi                
                    cos_simi[i][j] = max_simi
                    cos_simi[j][i] = max_simi

    return cos_simi,sentences 

print_time()





# initiate an encoder
encoder = USE()





def ProcessThreadSeq(thread_seq, oh_reply_dict):
    global cnt_thread, encoder
    cnt_thread += 1
    cur_thread = threads_dict[thread_seq[0][THREAD_ID_INDEX]]
    thread_author = cur_thread[TH_AUTHOR_INDEX]
    prev_author = ""
    sub_seq = []
    sent_token_seq = []
       
    # ----------------------------------------------------------------------------

    for i, row in enumerate(thread_seq):
        sent_token = nltk.sent_tokenize(row[CONTENT_INDEX].lower())
        # form sub sequences for each thread: 
        if prev_author != "" and (prev_author != thread_author and                                   row[AUTHOR_ID_INDEX] == thread_author and len(sub_seq) >= 3):
            ProcessSubSeq(sub_seq,cur_thread, oh_reply_dict)
#             sent_token_seq = []
            sub_seq = []
#             sent_token_seq.append(sent_token)
            sub_seq.append(row)
        else:
            sub_seq.append(row)
#             sent_token_seq.append(sent_token)
        prev_author = row[AUTHOR_ID_INDEX]
        
    ProcessSubSeq(sub_seq,cur_thread, oh_reply_dict)
    

# test 2: mix sub-sequences with/without delta
def ProcessSubSeq(seq, cur_thread,oh_reply_dict):
    global cnt_ignored_seq, delta_record_cnt, nondelta_record_cnt, delta_threads
    global cnt_train_subseq,cnt_test_subseq,data_raw_set

    # add the initial post to the beginning of each seq or subseq
    thread_op = cur_thread[TH_TITLE_INDEX]+". "+cur_thread[TH_TEXT_INDEX] if len(cur_thread[TH_TEXT_INDEX]) > 0  else cur_thread[TH_TITLE_INDEX]
    seq_array = np.array(seq)
    docs = [thread_op] + seq_array[:, CONTENT_INDEX].tolist()
    
    simi = np.zeros((len(docs),len(docs)))
    simi,sent_seq = USE_simi(seq,docs,cur_thread)
    # --------------------------------

    features = form_feature_dicts(seq, sent_seq, simi, oh_reply_dict)
    labels = form_labels(seq)

    delta_sum = sum(list(map(int, labels)))
        
        
    if len(features[0]) < FEATURE_NUM:
        print(seq[0][0] + " - less features")
        cnt_ignored_seq += 1
    elif HAS_DELTA_ONLY and delta_sum == 0:
        cnt_ignored_seq += 1
    else:
        data_set.append(features)
        data_label.append(labels)
        
        data_raw_set.append(seq)



print("complete:")
print_time()





def GetTokenAndCos(thread_seq):
    cur_thread = threads_dict[thread_seq[0][THREAD_ID_INDEX]]
    thread_author = cur_thread[TH_AUTHOR_INDEX]
    
    encoded_thread = {}
    
    # calculate similarity between comments based on the whole thread
    
    # --------------------------------
    # option 4 - calculate cos with USE at sentence level and calculate the max(similarity)
    
    # add the initial post to the beginning of each seq or subseq
    thread_op = cur_thread[TH_TITLE_INDEX]+". "+cur_thread[TH_TEXT_INDEX] if len(cur_thread[TH_TEXT_INDEX]) > 0  else cur_thread[TH_TITLE_INDEX]
    comments =  np.array(thread_seq)
    docs = comments[:, CONTENT_INDEX]
    
    thread_id = thread_seq[0][THREAD_ID_INDEX]    
    
    encoded_thread[thread_id] = USE_simi_preprocess(thread_op,docs,cur_thread)
    
#     print(encoded_thread)
    
    return encoded_thread


# def USE_simi(docs,cur_thread,thread_op_encoded):
def USE_simi_preprocess(thread_op,docs,cur_thread):
    sentences = []
    encoded_sentences = []
    thread_author = cur_thread[TH_AUTHOR_INDEX]    
    
    # encode each comments, including texts from opinion holder. docs could be seq or subseq
    for i,para in enumerate(docs):
        sent_token = nltk.sent_tokenize(para)
        sentences.append(sent_token)
        if para == "":
            encoded_sentences.append(np.zeros(len(encoded_sentences[0])))
        else:
            encoded_sentences.append(encoder.encode(sent_token))
        
    if len(encoded_sentences) == 0:
        print("Ah!! empty docs")
        

    # simi with op    
    simi_op = np.zeros(len(docs))
    if thread_op != "":
        token_op = nltk.sent_tokenize(thread_op)
        encoded_op = encoder.encode(token_op)

    for j in range(len(encoded_sentences)):
        comm_sent_j = encoded_sentences[j]
        max_simi = 0
        for x,y in itertools.product(encoded_op,comm_sent_j):
            simi = calculate_similarity(x,y)
            if simi>max_simi and simi != 1:
                max_simi = simi                
        simi_op[j] = max_simi
    
    # simi for paired comments
    simi_comments = np.zeros((len(encoded_sentences),len(encoded_sentences)))
    
    for i,comm_sent_i in enumerate(encoded_sentences):
        for j in range(i,len(encoded_sentences)):
            comm_sent_j = encoded_sentences[j]
            # in pairwise, compare sentences in comment i and comment j,
            # calculate max(similarity)            
            if i==j:
                simi_comments[i][j] = 1
            else:
                max_simi = 0
                for x,y in itertools.product(comm_sent_i,comm_sent_j):
                    simi = calculate_similarity(x,y)
                    if simi>max_simi and simi != 1:
                        max_simi = simi                
                simi_comments[i][j] = max_simi
                simi_comments[j][i] = max_simi

    return (simi_op,simi_comments,sentences, encoded_sentences)



    
    





# Linear CRF

comments_reader = csv.reader(open("data_cmv/cmv_comments.csv", newline='', encoding='mac_roman'), delimiter=',',
                        quotechar='"')

prev_thread = ""
# prev_author = ""
thread_seq = []
# thread_content_seq = []

oh_reply_dict = {} # key is the comment id which OP replied to; value is tuple (text, sentiment, order_index)

# variables
data_set = []
data_label = []

train_set = []
train_label = []

test_set = []
test_label = []

data_raw_set = []
encoded_all_comments = []

try: 
    print_time()
    row_cnt = 0
    
    for row in comments_reader:
        row_cnt += 1
        if len(row) == 0 or row_cnt == 1:
            continue
            
        # form a sequence for each thread        
        cur_thread = threads_dict[row[THREAD_ID_INDEX]]
        thread_author = cur_thread[TH_AUTHOR_INDEX]
        
        if row[AUTHOR_ID_INDEX] == thread_author:
            oh_reply_dict[row[REPLY_TO_INDEX]] = (row[CONTENT_INDEX], row[SENTIMENT_INDEX], row[COMMENT_ORDER_INDEX])
        
        if row[THREAD_ID_INDEX] == prev_thread or prev_thread == "":
            thread_seq.append(row)

        elif prev_thread != "" and prev_thread != "thread_id":
            ProcessThreadSeq(thread_seq, oh_reply_dict)
            thread_seq = []
            oh_reply_dict = {}
            thread_seq.append(row)
 
    
 
        prev_thread = row[THREAD_ID_INDEX]

        if row_cnt % 10000 == 0:
            print(row_cnt, datetime.datetime.now())
            
    # for the last thread
    print(row_cnt, oh_reply_dict)
    ProcessThreadSeq(thread_seq, oh_reply_dict)
    print("Done processing.")

except Exception as ex:
    sys.stderr.write('Exception\n')
    extype, exvalue, extrace = sys.exc_info()
    traceback.print_exception(extype, exvalue, extrace)
    #sys.exc_clear()

print_time()
# print(row)
print("total encoded dataset: ", len(encoded_all_comments)) # show number of encoded threads
print("total raw data: ", len(data_raw_set))

# print(data_raw_set[0])





# process sequences from pickled data for edge CRF
# transform it to linear-crf structure

TRAIN_THREAD_NUM = 5000 #10000 comments: 227 threads; 1000 comments: 13 threads

data_set_loaded = pickle_load("data_cmv/pkl/edge_thread_dataset.pkl")

# split the dataset into train and test
train_set = data_set_loaded["train_set"]
train_label = data_set_loaded["train_label"]

test_set = data_set_loaded["test_set"]
test_label = data_set_loaded["test_label"]

# combine the incorrectly-splitted train-test into dataset
data_set = train_set + test_set
data_label = train_label + test_label


print(len(data_set), len(data_label))

print_time()





crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs', 
    c1=0.1, 
    c2=0.1, 
    max_iterations=100, 
    all_possible_transitions=True
)

print_time("Start fitting...")

crf.fit(train_set, train_label)

print_time("Done fitting.")

# pickle trained crf
if pickle_overwrite:
    fileObject = open(file_model+"_"+curtime()+".pkl",'wb') 
    pickle.dump(crf,fileObject)
    fileObject.close()





pickle_overwrite = True    

# Return double of n 
def pos_value(lst): 
    return [ele['1'] for ele in lst] 

labels = list(crf.classes_)

# ------------------------------
# marginal probability of label

predict_result_bi = crf.predict(test_set)
predict_result = crf.predict_marginals(test_set) # y_score for both classes '0' and '1'

# pickle predicted results
if pickle_overwrite:
    fileObject = open("test_linear_seq_new_result_bi_"+curtime()+".pkl",'wb') 
    pickle.dump(predict_result_bi,fileObject)
    fileObject.close()

    fileObject = open("test_linear_seq_new_result_"+"_"+curtime()+".pkl",'wb') 
    pickle.dump(predict_result,fileObject)
    fileObject.close()

y_score = list(map(pos_value, (lst for lst in predict_result))) 

test_label_trans = list(map(int,np.hstack(test_label)))
predict_label_trans = list(map(float,np.hstack(y_score)))

print_time()





# AUC-ROC plot

print('ROC-AUC: ', roc_auc_score(test_label_trans, predict_label_trans))
fpr, tpr, thresholds = roc_curve(test_label_trans, predict_label_trans, pos_label=1)

print(classification_report(np.concatenate(test_label), np.concatenate(predict_result_bi)))

roc_auc = metrics.auc(fpr, tpr)

import matplotlib
# matplotlib.use("TkAgg")
matplotlib.rcParams.update({'font.size': 20})
f=plt.figure()
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
if pickle_overwrite:
    f.savefig("data_cmv/figures/exp-"+test_name+"-auc-"+curtime()+".pdf", bbox_inches='tight')

print_time()





# precision-recall-curve (PRC)

test_label_trans = list(map(int,np.hstack(test_label)))
predict_label_trans = list(map(float,np.hstack(y_score)))

print(test_label_trans[:10])
print(predict_label_trans[:10])

average_precision = average_precision_score(test_label_trans, predict_label_trans)

print('PRC for skip-chain CRF with sub-sequences \n {0:0.2f}'.format(
      average_precision))

# plot PRC

precision, recall, thresholds = precision_recall_curve(test_label_trans, predict_label_trans)

pos_rate = sum(test_label_trans)/len(predict_label_trans)

f=plt.figure()
matplotlib.rcParams.update({'font.size': 20})
# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post', label='AP = %0.2f' % average_precision)
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
plt.plot([0, 1], [pos_rate, pos_rate], 'r--')
plt.legend(loc='upper right')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
if pickle_overwrite:
    f.savefig("data_cmv/figures/exp-"+test_name+"-prc-"+curtime()+".pdf", bbox_inches='tight')


print_time()


# measure MRR (mean reciprocal rank)
# wikipedia: https://en.wikipedia.org/wiki/Mean_reciprocal_rank

print("sub seq total #", len(y_score))
ranked_results = []

for i in range(len(y_score)):
    list1 = test_label[i]
    list2 = y_score[i]
    list2, list1 = zip(*sorted(zip(list2,list1), reverse=True))
    list1_ranked = list(map(int, list1))
    # ignore non-delta sub sequences
    if sum(list1_ranked) != 0:
        ranked_results.append(list1_ranked)

print("sub seq for mrr #", len(ranked_results))

rs = (np.asarray(r).nonzero()[0] for r in ranked_results)

mrr = np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

print(mrr)

