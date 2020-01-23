
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

import numpy as np
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

import scipy.stats

import spicy 
from scipy import spatial





# ------------------------------------------------------------------
# cmv_threads column index (starting from 0)
TH_ID_INDEX = 0
TH_TITLE_INDEX = 1
TH_TEXT_INDEX = 4
TH_AUTHOR_INDEX = 5
TH_LENGTH_INDEX = 6

# cmv_comments column index (starting from 0)
THREAD_ID_INDEX = 0
CONTENT_INDEX = 8 # text content
AUTHOR_ID_INDEX = 5
DELTA_EARN_INDEX = 9
FLAIR_INDEX = 6
SENTIMENT_INDEX = 16
COMMENT_ID_INDEX = 2
REPLY_TO_INDEX = 3


HEDGE_INDEX = 23
ASSERTIVES_INDEX = 17
FACTIVES_INDEX =  18
IMPLICATIVES_INDEX = 19 
REPORT_VERBS_INDEX = 20
OPINION_NEG_INDEX = 21
OPINION_POS_INDEX = 22
STRONG_SUBJ_INDEX = 24
WEAK_SUBJ_INDEX = 25

# running example
EXAMPLE = '3j895i'
EXAMPLE_LEN = 5
EXAMPLE_INDEX = 4

# parameters
# TODO - grid search
THRESHOLD_COS = 0.5
TFIDF_MAX_FEATURES = 100

HAS_DELTA_ONLY = False
TRAIN_THREAD_NUM = 900

FEED_PAIRED = False

# variables
data_set = []
data_label = []

train_set = []
train_label = []

test_set = []
test_label = []

pickle_overwrite = False

prefix_pkl = "data_cmv/pkl/new_skip_thread_"

file_dataset = prefix_pkl + "dataset.pkl"
file_model = prefix_pkl + "crf.pkl"
file_lp_prob = prefix_pkl + "result_prob.pkl"

hedges = []
with open('data_cmv/input/hedge_list.txt', 'r') as f:
    content = f.readlines()
    hedges = [x.strip() for x in content]

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



def print_time():
    print(datetime.datetime.now())

print_time()


def pickle_save(filename, data2pkl):
    global pickle_overwrite
    if pickle_overwrite == True:
        fileObject = open(filename,'wb') 
        pickle.dump(data2pkl,fileObject)
        fileObject.close()
        

        
def pickle_load(filename, data2pkl):
    fileObject = open(filename,'wb') 
    pickle.dump(data2pkl,fileObject)
    fileObject.close()
        
# ------------------------------------------------------------------
# Extract features

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

        features_list = []
        for i, entry in enumerate(seq):
            word_cnt_entry = len(re.split(r'[^0-9A-Za-z]+',entry[CONTENT_INDEX]))
            word_token = re.split(r'[^0-9A-Za-z]+',entry[CONTENT_INDEX].lower())

            if FEED_PAIRED:
                simi_init = self.cos[0][i+1]
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
            'simi_initial':simi_init,
            # similarity with the recent opinion holder's post
            'simi_recent': cos[int(entry[COMMENT_ORDER_INDEX])].T[0:holder_post_cnt].mean() if holder_post_cnt > 0 \
                and i >= holder_post_cnt 
                else cos[len(seq)][int(entry[COMMENT_ORDER_INDEX])],      
                    'simi_recent': simi_recent,
            'direct_response': (comment_id_list.index(entry[REPLY_TO_INDEX])+1.0)/len(seq) if entry[REPLY_TO_INDEX] in comment_id_list
                else 0,
            'quotation': 1 if "&gt;" in entry[CONTENT_INDEX] else 0  
            }
            features_list.append(feature_dict)
  
        return features_list


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

# ------------------------------------------------------------------

# for each thread, form a feature vector
def form_features(seq,cos):
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
#                 'content_similarity': 0.8,
                'seq_stats': 1,
            },
        )
         )
    ])

    output = feature_unioned.fit_transform(seq)

    return output.toarray()


def form_edges(seq, cos):
    edges_linear = np.vstack([np.arange(len(seq)-1), np.arange(1, len(seq))])
    edges_skip = []
    #edges_skip = np.array([[0, 2], [1, 2]])

    docs = []

    # ---------------------------------------------------
    # option 1: if similarity between two comments > threshold
    for i in range(1, len(seq)):
        for j in range(i+2, len(seq)):
            # if cosine similarity > THRESHOLD_COS, form an edge
            # print(cos[i,j], i, j)            
            # form skip edges using TF-IDF cos similarity           
            if cos[i,j] > THRESHOLD_COS:
                edges_skip.append([i-1, j-1])
            # form skip edges using direct response
            elif seq[j][REPLY_TO_INDEX] == seq[i][COMMENT_ID_INDEX]:
#                 print("direct response")
                edges_skip.append([i-1, j-1]) 
            elif seq[j][AUTHOR_ID_INDEX] == seq[i][AUTHOR_ID_INDEX]:
#                 print("direct response")
                edges_skip.append([i-1, j-1]) 
        
    edges_skip = np.array(edges_skip)
    
    if len(edges_skip) > 0:
        edges = np.vstack([edges_linear.T, edges_skip]).T
    else:
        edges = edges_linear
    return edges

def form_labels(seq):
    labels = []
    for entry in seq:
        if entry[DELTA_EARN_INDEX] == "1":
            labels.append(1)
        else:
            labels.append(0)
    return labels


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


def calculate_similarity(vec1,vec2):
    simi = 1 - spatial.distance.cosine(vec1, vec2) 
    return simi
    

def sent_encode(text):
    return nltk.sent_tokenize(text)


# def USE_simi(docs,cur_thread,thread_op_encoded):
def USE_simi(seq,docs,cur_thread):
    sentences = []
    encoded_sentences = []
    thread_author = cur_thread[TH_AUTHOR_INDEX]    

    # encode each comments, including texts from opinion holder. docs could be seq or subseq or paired
    for i,para in enumerate(docs):
        if para == "":
            encoded_sentences.append(np.zeros(len(encoded_sentences[0])))
        else:
            sent_token = nltk.sent_tokenize(para)
            sentences.append(sent_token)

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
        # calculate cos_simi for every pair of seq[], and fetch later
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
            
    return cos_simi  
    

cnt_thread = 0 

def ProcessThread(thread_seq):
    global cnt_thread, encoder
    cur_thread = threads_dict[thread_seq[0][THREAD_ID_INDEX]]
    thread_author = cur_thread[TH_AUTHOR_INDEX]
    
    # calculate similarity between comments based on the whole thread
    seq_array = np.array(thread_seq)
#     print(seq_array)
    docs = seq_array[:, CONTENT_INDEX]
    docs = docs.tolist()    
   
    # add the initial post to the end of seq
    docs.append(cur_thread[TH_TEXT_INDEX])
    
    # create a 2D numpy array as cos
    cos_len = len(thread_seq)+1 # appended initial statement to thread
    cos = np.zeros(shape=(cos_len,cos_len))    
    
     # ----------------------------------------------------------------    
    # option 1 - calculate cos with tfidf, cos[0][0] ... cos[n][n]
    # --------------------------------
#     vect = TfidfVectorizer()
#     tfidf = vect.fit_transform(docs)
#     trunc = TruncatedSVD(tfidf)
#     cos = (tfidf * tfidf.T).A

    # --------------------------------
    # option 2 - calculate cos with ELMO embeddings

#     cos =Elmo4Docs(docs)
    # --------------------------------
    # option 3 - calculate cos with USE  
#     emb = encoder.encode(docs)
    
#     for i,vec1 in enumerate(emb):
#         for j,vec2 in enumerate(emb):
#             cos[i,j] = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
         
    # --------------------------------
    # option 4 - calculate cos with USE at sentence level and calculate the max(similarity)
    
    # add the initial post to the beginning of each seq or subseq
    thread_op = cur_thread[TH_TITLE_INDEX]+". "+cur_thread[TH_TEXT_INDEX] if len(cur_thread[TH_TEXT_INDEX]) > 0  else cur_thread[TH_TITLE_INDEX]
    seq_array = np.array(thread_seq)
    docs = [thread_op] + seq_array[:, CONTENT_INDEX].tolist()
    
    simi = np.zeros((len(docs),len(docs)))
    simi = USE_simi(thread_seq,docs,cur_thread)
    # --------------------------------
    
    # print("--------------------- thread ---------------------")
    if len(thread_seq) < 3:
        print(thread_seq[0][0] + " - less than 2 comments")
    else:
        edges = form_edges(thread_seq, simi)
        features = form_features(thread_seq, simi)
        labels = form_labels(thread_seq)

        if len(features[0])< 26:
            print(thread_seq[0][0] + " - less features")
        else:
#             features_norm = preprocessing.normalize(features, norm='l2')
            data_set.append((features, edges))
            data_label.append(np.array(labels))
#             print(thread_seq[0][0])





# using tensorflow USE for word embeddings

USE_URL = 'https://tfhub.dev/google/universal-sentence-encoder/2'
cachedir = 'c6f5954ffa065cdb2f2e604e740e8838bf21a2d3'
# cachedir = 'test1'

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





# ------------------------------------------------------------------
# initiate an encoder
encoder = USE()





# ------------------------------------------------------------------
# main entrance of the project
# ------------------------------------------------------------------

threads_reader = csv.reader(open("data_cmv/cmv_threads.csv", newline='', encoding='mac_roman'), delimiter=',',
                       quotechar='"')

                            
threads_dict = {}
examined_threads = {}

    
try:
    for row in threads_reader:
        if row[0] != "thread_id":
            threads_dict[row[0]] = row
            examined_threads[row[0]] = 0
        
except Exception as ex:
    sys.stderr.write('Exception\n')
    extype, exvalue, extrace = sys.exc_info()
    traceback.print_exception(extype, exvalue, extrace)
    
print_time()    





# ------------------------------------------------------------------
spamreader = csv.reader(open("data_cmv/cmv_comments.csv", newline='', encoding='mac_roman'), delimiter=',', quotechar='"')

prev_thread = ""
thread_seq = []


    
try:
    row_cnt = 0
    print("processing thread...", datetime.datetime.now())
    for row in spamreader:

        if row[0] == prev_thread:
            thread_seq.append(row)
        elif prev_thread != "" and prev_thread != "thread_id":
            ProcessThread(thread_seq)

            thread_seq = []
            thread_seq.append(row)
        row_cnt += 1    
        if row_cnt % 10000 == 0:
            print(row_cnt, datetime.datetime.now())
          
        prev_thread = row[0]
    # for the list thread
    ProcessThread(thread_seq)
    print("Done processing.")

except Exception as ex:
    sys.stderr.write('Exception\n')
    extype, exvalue, extrace = sys.exc_info()
    traceback.print_exception(extype, exvalue, extrace)
    #sys.exc_clear()
    
print(datetime.datetime.now())

print(len(data_set))





TRAIN_THREAD_NUM = 5000
# split the dataset into train and test
train_set = data_set[:TRAIN_THREAD_NUM]
train_label = data_label[:TRAIN_THREAD_NUM]

test_set = data_set[TRAIN_THREAD_NUM:]
test_label = data_label[TRAIN_THREAD_NUM:]

print("train: ",len(train_set), "test: ", len(test_set))
print("total: ", len(data_set))


# pickle dataset

train_test_dict = {
    "train_set":train_set,
    "train_label":train_label,
    "test_set":test_set,
    "test_label":test_label
}

if pickle_overwrite:
    fileObject = open(file_dataset,'wb') 
    pickle.dump(train_test_dict,fileObject)
    fileObject.close()





# fit the CRF model into a SVM classifier
# model = GraphCRF(directed=True, inference_method="max-product")
# model = GraphCRF(directed=True, inference_method="ad3")

print(datetime.datetime.now())

model = GraphCRF(directed=True, inference_method = ('lp', {'relaxed' : True}))

# Use a n-slack SSVM learner
# ssvm = FrankWolfeSSVM(model=model, C=.1, max_iter=50)
# predict_result = ssvm.predict(test_set)
# ssvm.fit(train_set, train_label)
# score = ssvm.score(test_set, test_label)

from pystruct.learners import OneSlackSSVM
learner = OneSlackSSVM(model=model, C=.02, max_iter=10)
 
learner.fit(train_set, train_label)

lp_probs = learner.predict(test_set)

# print(lp_probs[0][0][:,1])
print(datetime.datetime.now())

# pickle trained crf
if pickle_overwrite:
    fileObject = open(file_model,'wb') 
    pickle.dump(learner,fileObject)
    fileObject.close()





xx = []
# new_xx = []
for x in lp_probs:
#     new_x = []
    xx.append(x[0][:,1])
    
lp_probs_trans = np.concatenate(xx)
test_label_trans = np.concatenate(test_label)

print(len(lp_probs_trans),len(test_label_trans))


# pickle predicted results
fileObject = open(file_lp_prob,'wb') 
pickle.dump(lp_probs_trans,fileObject)
fileObject.close()

import matplotlib

import matplotlib as plt
import matplotlib.pyplot as plt

import sklearn.metrics as metrics
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score, jaccard_similarity_score, roc_curve, roc_auc_score

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.utils.fixes import signature

# ----------------  plot AUC-ROC ----------------

print('ROC-AUC: ', roc_auc_score(test_label_trans, lp_probs_trans))

fpr, tpr, thresholds = roc_curve(test_label_trans, lp_probs_trans, pos_label=1)

roc_auc = metrics.auc(fpr, tpr)



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
f.savefig("data_cmv/figures/exp-skip-thread-auc-large-6.pdf", bbox_inches='tight')

# ----------------  plot PRC ----------------


average_precision = average_precision_score(test_label_trans, lp_probs_trans)

precision, recall, thresholds = precision_recall_curve(test_label_trans, lp_probs_trans)

pos_rate = sum(test_label_trans)/len(test_label_trans)

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
# plt.title('PRC for skip-chain CRF with threads \n AP={0:0.2f}'.format(average_precision))
f.savefig("data_cmv/figures/exp-skip-thread-prc-large-6.pdf", bbox_inches='tight')

print(precision)
print(recall)





# measure MRR (mean reciprocal rank)

print("discussion total #", len(test_label))
ranked_results = []

multi_delta_cnt = 0
for i in range(len(xx)):
    list1 = test_label[i]
    list2 = xx[i]

    list2_ranked, list1_ranked = zip(*sorted(zip(list2,list1), reverse=True))
    list1_ranked_int = list(map(int, list1_ranked))
    # optional: ignore non-delta sub sequences
    if sum(list1_ranked_int) > 1:
        multi_delta_cnt += 1
    ranked_results.append(list1_ranked_int)

print("discussion for mrr #", len(ranked_results))

rs = (np.asarray(r).nonzero()[0] for r in ranked_results)

mrr = np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

print(mrr)

print(multi_delta_cnt)

