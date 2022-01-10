import csv #read csv files
import nltk # word tokenization
import spacy #lemmatization and sentence boundary detection
import json # read and write json files
import os
import numpy as np
import string
import random # initialization of x vector in PageRank algorithm
from pysbd.utils import PySBDFactory #sentence boundary detection
import xml.etree.ElementTree as ET # to parse topic xml file


def read_data(): #read abstract data and group each document w.r.t its topic
    topics = ['1', '4', '13']
    with open('qrels-rnd1.txt', 'r', ) as f:
        relativeness_list = f.read().split('\n')
    doc_ids = {}
    doc_ids[topics[0]] = []
    doc_ids[topics[1]] = []
    doc_ids[topics[2]] = []
    for i in relativeness_list:
        doc_rel = i.split()
        if len(doc_rel) == 4:
            if doc_rel[3] == '2' and doc_rel[0] in topics:
                doc_ids[doc_rel[0]].append(doc_rel[2])
    id_abs = {}
    with open('04-10-mag-mapping.txt', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            id_abs[row[0]] = row[8]

    return doc_ids, id_abs, topics


def lemmatize_corpus(id_abs): # lemmatize whole corpus for tf-idf calculations and save
    if os.path.exists('lemmatized_dictionary.json'):
        with open('lemmatized_dictionary.json') as json_file:
            lemma_id_abs = json.load(json_file)
    else:
        nlp = spacy.load('en_core_web_sm')
        punctuation = ',\':;'
        lemma_id_abs = {}
        for key, value in id_abs.items():
            words = nlp(value)
            if words.text != '':
                lemmatized_words = []
                for word in words:
                    if word.text not in punctuation and word.text != 'BACKGROUND':
                        lemmatized_words.append(word.lemma_)
                lemma_id_abs[key] = ' '.join(lemmatized_words)
        with open("lemmatized_dictionary.json", "w") as f:
            json.dump(lemma_id_abs, f)

    return lemma_id_abs


def create_vocabulary(id_abs):  # create vocabulary for tf-idf vectors and save
    if os.path.exists('vocabulary.txt'):
        with open('vocabulary.txt', 'r', encoding='utf-8') as f:
            vocab = f.read().split('\n')
    else:
        punc = string.punctuation
        vocab = []
        for value in id_abs.values():
            words = nltk.word_tokenize(value)
            for i in words:
                if i not in vocab and i not in punc:
                    vocab.append(i)
        with open('vocabulary.txt', 'w', encoding='utf-8') as f:
            for word in vocab:
                f.write("%s\n" % word)
    return vocab


def preprocess_data():  # complete all preprocessing steps
    doc_ids, id_abs, topics = read_data()
    lem_id_abs = lemmatize_corpus(id_abs)
    vocab = create_vocabulary(lem_id_abs)
    idf = create_idf_vector(lem_id_abs, vocab)
    sel_id_abs = select_relevant_docs(lem_id_abs, doc_ids)
    new_doc_ids = update_doc_ids(sel_id_abs, doc_ids)
    return sel_id_abs, new_doc_ids, vocab, idf, topics, id_abs


def select_relevant_docs(id_abs, doc_ids): # select only the documents fully related to the topic
    selected_id_abs = {}
    for i in doc_ids.values():
        for j in i:
            if j in id_abs.keys():
                selected_id_abs[j] = id_abs[j]
    return selected_id_abs


def create_idf_vector(id_abs, vocab): # create idf vector for the words in the vocab, over whole corpus
    if os.path.exists('idf_vector.npy'):
        idf = np.load('idf_vector.npy')
    else:
        N = len(id_abs)
        idf = np.zeros([1, len(vocab)])
        for i in range(len(vocab)):
            n = 0
            for j in id_abs.values():
                if vocab[i] in j:
                    n += 1
            if n != 0:
                idf[0, i] = np.log(N/n)
            else:
                idf[0, i] = 0
        np.save('idf_vector.npy', idf)
    return idf


def create_doc_tf_table(id_abs, vocab, docs_id, topic_id): # for a given topic, creates tf table for the documents
    topic_docs = list(docs_id[topic_id])
    tf_table = np.zeros([len(topic_docs), len(vocab)])
    tf_id_list = []
    for doc in topic_docs:
        if doc in id_abs.keys():
            words = nltk.word_tokenize(id_abs[doc])
            for j in range(len(vocab)):
                tf_table[len(tf_id_list), j] = words.count(vocab[j])
            tf_id_list.append(doc)

    return tf_table, tf_id_list


def cosine_similarity(vec1, vec2):  # calculates cosine similarity for two tf-idf vectors
    inner = np.dot(vec1, vec2.T)
    length = (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    if length != 0:
        sim = inner/length
    else:
        sim = 0
    return sim


def normalize_similarities(matrix, tel_rate=0.15): # normalizes rows of transition matrix so that sum of a row is 1
    total = 1 - tel_rate
    doc_size = np.size(matrix, 0)
    for i in range(doc_size):
        matrix[i, :] = matrix[i, :]*total/np.sum(matrix[i, :]) + tel_rate/doc_size
    return matrix


def power_iteration(transition, n, tol=0.00001): # apply power method to find SS probabilities, return top n docs/sentences
    length = np.size(transition, 0)
    x = np.zeros([1, length])
    for i in range(length): # change to numpy
        x[0, i] = random.uniform(0, 1)
    x = x/np.sum(x)
    x_p = np.zeros([1, length])
    iter = 0
    while np.linalg.norm(x-x_p) >= tol:
        x_p = x
        x = x.dot(transition)
        iter += 1
    top_ten, scores = get_top_n_item(x, n)
    return top_ten, scores


def update_doc_ids(id_abs, doc_ids): # remove documents with no abstracts from the document dictionary
    new_doc_ids = {}
    id_list = []
    for key, value in doc_ids.items():
        for i in value:
            if i in id_abs.keys():
                id_list.append(i)
        new_doc_ids[key] = id_list
        id_list = []
    return new_doc_ids


def get_top_n_item(x, n): # select the greatest n elements and their corresponding values from a vector
    top = []
    scores = []
    for i in range(n):
        ind = np.argmax(x)
        top.append(ind)
        scores.append(str(x[0, ind]))
        x[0, ind] = 0
    return top, scores


def get_salient_docs(id_abs, idf, docs_id, vocab): # get top 10 most important docs for all topics
    salient_docs = {}
    scores = {}
    for key, value in docs_id.items():
        id_list = list(value)
        similarity_matrix = np.zeros([len(id_list), len(id_list)])
        tf_table, _ = create_doc_tf_table(id_abs, vocab, docs_id, key)
        for i in range(len(id_list)):
            vec1 = tf_table[i]*idf
            for j in range(len(id_list)):
                vec2 = tf_table[j]*idf
                similarity_matrix[i, j] = cosine_similarity(vec1, vec2)
        similarity_matrix[similarity_matrix < 0.1] = 0
        similarity_matrix = normalize_similarities(similarity_matrix)
        top_docs, scores[key] = power_iteration(similarity_matrix, 10)
        top_docs_ids = []
        for i in top_docs:
            top_docs_ids.append(id_list[i])
        salient_docs[key] = top_docs_ids

    return salient_docs, scores


def create_dictionary_of_sentences(salient_docs, id_abs, topics): # using the top documents, parse abstracts into sentences
    topic_sen = {}
    doc_id_len = {}
    for i in topics: # in each topic, we have 10 docs, and each doc has list of sentences
        topic_sen[i] = {}
        topic_sen[i], doc_id_len[i] = split_into_sentences(id_abs, salient_docs[i])
    return topic_sen, doc_id_len


def split_into_sentences(id_abs, doc_list):
    nlp = spacy.blank('en')
    nlp.add_pipe(PySBDFactory(nlp))
    sentence_list = {}
    doc_id_len = {}
    for i in doc_list:
        text = id_abs[i]
        doc = nlp(text)
        sentence_list[i] = list(doc.sents)
        doc_id_len[i] = len(list(doc.sents))
    return sentence_list, doc_id_len


def create_sentence_tf_table(id_sentences, vocab): # create tf table for sentences given a topic
    size = 0 #total number of sentences in selected documents for one topic
    for sen_list in id_sentences.values():
        size = size + len(sen_list)
    tf_table = np.zeros([size, len(vocab)])
    i = 0
    for sen_list in id_sentences.values():
        for sen in sen_list:
            words = nltk.word_tokenize(sen.text)
            for j in range(len(vocab)):
                tf_table[i, j] = words.count(vocab[j])
            i += 1
    return tf_table


def extractive_summary(salient_docs, sel_id_abs, topics, vocab, idf, id_abs): # extract 20 sentences for each topic from 10 docs for each topic
    topic_sen, doc_id_len = create_dictionary_of_sentences(salient_docs, sel_id_abs, topics)
    topic_summary = {}
    scores = {}
    for i in topics:
        tf_table = create_sentence_tf_table(topic_sen[i], vocab)
        sen_no = np.size(tf_table, 0)
        similarity = np.zeros([sen_no, sen_no])
        for j in range(sen_no):
            vec1 = tf_table[j]*idf
            for k in range(sen_no):
                vec2 = tf_table[k]*idf
                similarity[j, k] = cosine_similarity(vec1, vec2)
        similarity[similarity < 0.1] = 0
        similarity = normalize_similarities(similarity)
        top_sentences, scores[i] = power_iteration(similarity, 20)
        topic_summary[i] = get_topic_summary_from_original_text(id_abs, doc_id_len[i], top_sentences)
    return topic_summary, scores


def get_topic_summary_from_original_text(id_abs, doc_id_len, index_list): # take sentences from original abstract rather than lemmatized version
    nlp = spacy.blank('en')
    nlp.add_pipe(PySBDFactory(nlp))
    summary = []
    for index in index_list:
        doc_id, sentence_index = convert_index_to_sentence_index(doc_id_len, index)
        doc = nlp(id_abs[doc_id])
        doc_sentences = list(doc.sents)
        try:
            summary.append((doc_sentences[sentence_index]).text)
        except IndexError:
            print('Error in one sentence, not included in the summary')
    summary_text = '\n'.join(summary)
    return summary_text

def convert_index_to_sentence_index(doc_id_len, index):  # locating sentence position
    doc_index = 0
    for key, value in doc_id_len.items():
        if (doc_index + value) > index:
            doc_id = key
            sentence_index = index - doc_index
            break
        doc_index = doc_index + value
    return doc_id, sentence_index


def print_results(salient_docs, topic_summary, doc_scores, sen_scores): # print results
    root = ET.parse('topics-rnd1.xml').getroot()
    query = []
    for elem in root:
        value = elem.get('number')
        if value in topic_summary.keys():
            txt = elem[0].text
            query.append(txt)
    i = 0
    for key, value in salient_docs.items():
        print('Most Important 10 Documents for the topic: ' + query[i])
        doc_txt = []
        for doc in value:
            doc_txt.append(doc)
        print(', '.join(doc_txt))
        print(',\t'.join(doc_scores[key]))
        print('----------------')
        print('Extractive summary of the topic ' + query[i] + ' by 20 sentences')
        print(topic_summary[key])
        print(','.join(sen_scores[key]))
        print('----------------')
        i += 1


if __name__ == '__main__':
    sel_id_abs, doc_ids, vocab, idf, topics, id_abs = preprocess_data()
    salient_docs, doc_scores = get_salient_docs(sel_id_abs, idf, doc_ids, vocab)
    topic_summary, sen_scores = extractive_summary(salient_docs, sel_id_abs, topics, vocab, idf, id_abs)
    print_results(salient_docs, topic_summary, doc_scores, sen_scores)
