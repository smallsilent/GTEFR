# coding=utf-8
import codecs
import nltk
import re
import cPickle
import numpy as np
import math

name = 'd389h'
filepath_sen = 'updateLDA/' + name + '/svector'  #句子向量文件的路径
filepath_word = 'updateLDA/' + name + '/wvector'  #句子向量文件的路径
filepath_text = 'data/' + name + '.txt'  #要计算摘要的文件
filepath_topic = 'updateLDA/' + name + '/sdistribution'


#a_tf = 0.45
#a_wordsim = 0.45
#a_sensim = 0.05
#a_topicsim = 0.05
senscore = {}
score = []

def countword():
    wordcount = {}
    fileopen = codecs.open(filepath_text, 'r', 'utf-8')
    file_r = fileopen.readlines()
    fileopen.close
    senstring = []
    count = 0
    for line in file_r:
        line = line.strip()
        line = line.split('::')
        if line[0].find('d') >-1:
            if line[0].find('q') >-1:
                query = line[1]
            elif line[0].find('s') >-1:
                supplment = line[1]
        else:
            words = line[1].split(' ')
            for word in words:
                word = re.sub("[^A-Za-z0-9]", "", word).lower()  # 去除非英文字符 变小写
                s = nltk.stem.SnowballStemmer('english')
                word = s.stem(word)
                word = stop(word)
                word = word.lower()
                if word == ' ':
                    continue
                elif wordcount.has_key(word):
                    count = count + 1
                    wordcount[word] = wordcount[word] + 1
                else:
                    count = count + 1
                    wordcount[word] = 1
            senstring.append(line[1])
    file_w = open('summarization/' + name + '/wordcount', 'wb')
    cPickle.dump(wordcount, file_w)
    file_w.close()
    file_w = open('summarization/' + name + '/wordcount.txt', 'wb')
    wkeys = wordcount.keys()
    for key in wkeys:
        file_w.write(key + ' ' + str(wordcount[key]).replace('\n', '') + '\n')
    file_w.close()
    return query,supplment,senstring,count


def stop(string):
    stopwords = []
    fileopen = codecs.open('stopwords.txt', 'r','utf-8')
    file_r = fileopen.readlines()
    fileopen.close
    for line in file_r:
        stopwords.append(line.strip())
    for i in range(len(stopwords)):
        if stopwords[i] == string:
            string = string.replace(stopwords[i],' ')
    return string


def computecos(x,y):
    if (len(x) != len(y)):
        print('error input,x and y is not in the same space')
        return
    x = np.matrix(x)
    y = np.matrix(y)
    result1 = x * y.T
    result2 = x * x.T
    result3 = y * y.T
    result = result1 / (math.pow(result2 * result3 , 0.5))
    return result


def computewordsim_max(words,query):
    file_r = open(filepath_word, 'rb')
    wvector = cPickle.load(file_r)
    file_r.close()
    #wvector = {}
    #fileopen = codecs.open('wei_new_combine_100.txt', 'r', 'utf-8')
    #file_r = fileopen.readlines()
    #fileopen.close()
    #for line in file_r:
        #line = line.strip()
        #line = line.split(' ')
        #vector = [float(line[i]) for i in range(1, len(line))]
        #wvector[line[0]] = np.array(vector)
    max = -1
    words = words.split(' ')
    query = query.split(' ')
    for word in words:
        if word not in wvector:
            continue
        else:
            for query_word in query:
                cos = computecos(wvector[word],wvector[query_word])
                if cos > max:
                    max = cos
    return max

def computetf(words,count):
    file_r = open('summarization/' + name + '/wordcount', 'rb')
    wordcount = cPickle.load(file_r)
    file_r.close
    tf = 0.0
    words = words.split(' ')
    for word in words:
        tf = tf + wordcount[word]
    tf = tf/count
    return tf

def computewordsim_mean(words,query):
    wvector = {}
    fileopen = codecs.open('wei_new_combine_100.txt', 'r', 'utf-8')
    file_r = fileopen.readlines()
    fileopen.close()
    for line in file_r:
        line = line.strip()
        line = line.split(' ')
        vector = [float(line[i]) for i in range(1, len(line))]
        wvector[line[0]] = np.array(vector)
    cos_sum = 0.0
    for word in words:
        for query_word in query:
            cos = computecos(wvector[word],wvector[query_word])
            cos_sum = cos_sum + cos
    cos_sum = cos_sum/len(words)
    return cos_sum

def compute_sen_cos(sen,supplment):
    result = computecos(sen,supplment)
    return result

def compute_topic_cos(sen,supplment):
    result = computecos(sen,supplment)
    return result

def get_alltext_senvector():
    file_r = open(filepath_sen, 'rb')
    svector = cPickle.load(file_r)
    file_r.close()
    return svector

def get_alltext_topicvector():
    file_r = open(filepath_topic, 'rb')
    topicvector = cPickle.load(file_r)
    file_r.close()

    return topicvector



def pro_sen_string(senstring):
    words = ''
    senstring = senstring.split(' ')
    for word in senstring:
        word = re.sub("[^A-Za-z0-9]", "", word).lower()  # 去除非英文字符 变小写
        s = nltk.stem.SnowballStemmer('english')
        word = s.stem(word)
        word = stop(word)
        word = word.lower()
        if word == ' ':
            continue
        else:
            words = words + word
            words = words + ' '
    words = words.strip()
    return words

def summarization():
    query, supplment, sen_string, count = countword()
    print query
    print supplment
    print count
    svector = get_alltext_senvector()
    topicvector = get_alltext_topicvector()
    for i in range(len(sen_string)):
        print i
        if sen_string[i] not in svector:
            continue
        elif sen_string[i] == query:
            continue
        elif sen_string[i] == supplment:
            continue
        pro_senstring = pro_sen_string(sen_string[i])#对句子去停用词，词干化
        if pro_senstring == '':
            score = -1
        else:
            pro_query = pro_sen_string(query)#对句子去停用词，词干化
            tf = computetf(pro_senstring,count)#计算句子的tf值
            wordsim = computewordsim_max(sen_string[i],pro_query)
            sensim = compute_sen_cos(svector[sen_string[i]],svector[supplment])
            topicsim = compute_topic_cos(topicvector[sen_string[i]],topicvector[supplment])
            score = [tf,wordsim,sensim,topicsim]
            #score = a_tf*tf + a_wordsim*wordsim + a_sensim*sensim + a_topicsim*topicsim
            senscore[sen_string[i]] = score
        print score
    file_w = open('mertdata/' + name, 'wb')
    cPickle.dump(senscore, file_w)
    file_w.close()
    file_w = open('mertdata/' + name + '.txt', 'w')
    keys = senscore.keys()
    for key in keys:
        file_w.write(key + '/t' + str(senscore[key]).replace('\n', '') + '\n')
    file_w.close()


summarization()


