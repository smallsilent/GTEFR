# coding=utf-8
import codecs
import sys
import numpy as np
import random
from numpy import random
import cPickle
from sklearn import mixture
import math
import time

vecdim = 100
m = 6
K = 100
a = 0.025
reduce = 0.0001
Iteration = 50
frequence = 1
updatedoc = []
weight = []
mean = []
covars = []

# name = sys.argv[1]
name = 'd301i'


def countword():
    wordcount = {}
    fileopen = codecs.open('data/' + name + '.txt', 'r', 'utf-8')
    file_r = fileopen.readlines()
    for line in file_r:
        line = line.strip()
        line = line.split('::')
        if line[0].find('d') > -1:
            line[1] = stop(line[1])
            words = line[1].split(' ')
            for word in words:
                if wordcount.has_key(word):
                    wordcount[word] = wordcount[word] + 1
                else:
                    wordcount[word] = 1
    fileopen.close
    file_w = open('init/wordcount', 'wb')
    cPickle.dump(wordcount, file_w)
    file_w.close()


def stop(string):
    stopwords = []
    fileopen = codecs.open('stopwords.txt', 'r', 'utf-8')
    file_r = fileopen.readlines()
    fileopen.close
    for line in file_r:
        stopwords.append(' ' + line.strip() + ' ')
    for i in range(len(stopwords)):
        if stopwords[i] in string:
            string = string.replace(stopwords[i], ' ')
    return string


def init():
    # docs=[]
    fileopen = codecs.open('data/' + name + '.txt', 'r', 'utf-8')
    file_r = fileopen.readlines()
    doc = []
    # worddic = {}
    dswvector = {}
    for line in file_r:
        line = line.strip()
        line = line.split('::')
        if line[0].find('d') > -1 or line[0].find('q') > -1:
            print line[0]
            updatedoc.append(line[0])
            if len(doc) > 0:
                # docs.append(doc)
                file_w = open(docname, 'wb')
                cPickle.dump(doc, file_w)
                file_w.close()
                doc = []
            docname = 'init/' + line[0]
            rd = random.random(size=(vecdim))
            # rd = [0.5 for i in range(vecdim)]
            doc = [line[1], rd, []]
            dswvector[line[1]] = rd
            continue
        else:
            sen = []
            rd = random.random(size=(vecdim))
            # rd = [0.5 for i in range(vecdim)]
            sen = [line[1], rd, []]
            dswvector[line[1]] = rd
            words = line[1].split(' ')
            for index, word in enumerate(words):
                rd = random.random(size=(vecdim))
                # rd = [0.5 for i in range(vecdim)]
                # worddic[word] = rd
                dswvector[word] = rd
                item = []
                dtow = np.zeros((vecdim))
                stow = np.zeros((vecdim))
                item = [word, dtow, stow, [], []]
                if index < m:
                    for i in range(0, index):
                        wtow = np.zeros((vecdim))
                        item[3].append(words[i])
                        item[4].append(wtow)
                else:
                    for i in range(index - m, index):
                        wtow = np.zeros((vecdim))
                        item[3].append(words[i])
                        item[4].append(wtow)
                sen[2].append(item)
            doc[2].append(sen)
    file_w = open(docname, 'wb')
    cPickle.dump(doc, file_w)
    file_w.close()
    # file_w = open('LDAGMM/worddic', 'wb')
    # cPickle.dump(worddic, file_w)
    # file_w.close()
    file_w = open('init/dswvector', 'wb')
    cPickle.dump(dswvector, file_w)
    file_w.close()
    fileopen.close()


def init_with_stop_and_frequence():
    wvector = {}
    fileopen = codecs.open('wei_new_new_vectors.txt', 'r', 'utf-8')
    file_r = fileopen.readlines()
    fileopen.close()
    for line in file_r:
        line = line.strip()
        line = line.split(' ')
        vector = [float(line[i]) for i in range(1,len(line))]
        wvector[line[0]] = np.array(vector)
    countword()
    # 读取单词词频文件
    file_r = open('init/wordcount', 'rb')
    wordcount = cPickle.load(file_r)
    file_r.close
    fileopen = codecs.open('data/' + name + '.txt', 'r', 'utf-8')
    file_r = fileopen.readlines()
    doc = []
    dswvector = {}
    for line in file_r:
        line = line.strip()
        line = line.split('::')
        if line[0].find('d') > -1 or line[0].find('q') > -1:
            print line[0]
            updatedoc.append(line[0])
            if len(doc) > 0:
                # docs.append(doc)
                file_w = open(docname, 'wb')
                cPickle.dump(doc, file_w)
                file_w.close()
                doc = []
            docname = 'init/' + line[0]
            rd = random.random(size=(vecdim))*2-1
            # rd = [0.5 for i in range(vecdim)]
            doc = [line[1], rd, []]
            dswvector[line[1]] = rd
            continue
        else:
            sen = []
            rd = random.random(size=(vecdim))*2-1
            # rd = [0.5 for i in range(vecdim)]
            sen = [line[1], rd, []]
            dswvector[line[1]] = rd
            # 句子去掉停用词
            line[1] = stop(line[1])
            words = line[1].split(' ')
            for word in words:
                if wordcount.has_key(word):
                    if wordcount[word] < (frequence + 1):
                        words.remove(word)

            for index, word in enumerate(words):
                if word == '':
                    rd = random.random(size=(vecdim))*2-1
                else:
                    rd = wvector[word]
                dswvector[word] = rd
                #rd = random.random(size=(vecdim))
                # rd = [0.5 for i in range(vecdim)]
                # worddic[word] = rd

                item = []
                dtow = np.zeros((vecdim))
                stow = np.zeros((vecdim))
                item = [word, dtow, stow, [], []]
                if index < m:
                    for i in range(0, index):
                        wtow = np.zeros((vecdim))
                        item[3].append(words[i])
                        item[4].append(wtow)
                else:
                    for i in range(index - m, index):
                        wtow = np.zeros((vecdim))
                        item[3].append(words[i])
                        item[4].append(wtow)
                sen[2].append(item)
            doc[2].append(sen)
    file_w = open(docname, 'wb')
    cPickle.dump(doc, file_w)
    file_w.close()
    # file_w = open('LDAGMM/worddic', 'wb')
    # cPickle.dump(worddic, file_w)
    # file_w.close()
    file_w = open('init/dswvector', 'wb')
    cPickle.dump(dswvector, file_w)
    file_w.close()
    fileopen.close()


# 进行GMM算法
def ldagmm():
    file_r = open('init/dswvector', 'rb')
    alldata = cPickle.load(file_r)
    alldataarray = np.array(alldata.values())
    print 'start-------------'
    g = mixture.GMM(n_components=K, n_iter=1)
    g.fit(alldataarray)
    print np.round(g.weights_, 2)
    print type(g.weights_)
    file_r.close()


def diffgmm(score, data, weight, mean, cov):  # 高斯模型求导
    vector = [0 for i in range(vecdim)]
    vector = np.array(vector)
    # score = math.exp(score)
    # print score
    for j in range(K):
        x = data - mean[j]  # x-u
        x = np.matrix(x)
        E = np.linalg.inv(np.matrix(cov[j]))  # 协方差矩阵的逆
        # multiplier= score * weight[j]
        vector = vector + np.array((-1) * weight[j] * (x * E))[0, 0:vecdim]
    result = vector
    return result


def diffsigmoid(doc, sindex, windex, dswvector, vector):  # log sigmoid(a + a + a) 求导
    docvector = doc[1]
    sen = doc[2][sindex]
    senvector = sen[1]
    word = sen[2][windex]
    dtow = word[1]
    stow = word[2]
    sum = 0.0
    if windex < m:
        for i in range(0, windex):
            prewordstring = word[3]
            prewordstring = prewordstring[i]
            prewtow = word[4]
            prewtow = prewtow[i]
            prewvector = dswvector[prewordstring]
            sum = sum + np.dot(prewtow, prewvector)
    else:
        for i in range(m):
            prewordstring = word[3]
            prewordstring = prewordstring[i]
            prewtow = word[4]
            prewtow = prewtow[i]
            prewvector = dswvector[prewordstring]
            sum = sum + np.dot(prewtow, prewvector)
    result = math.exp((-1) * (np.dot(dtow, docvector) + np.dot(stow, senvector) + sum))
    result = (1 - 1 / (1 + result)) * vector
    return result


def diffsigmoid_with_LDA(doc, sindex, windex, dswvector, topic, vector):  # log sigmoid(a + a + a) 求导
    docvector = doc[1]
    sen = doc[2][sindex]
    senvector = sen[1]
    word = sen[2][windex]
    dtow = topic
    stow = word[2]
    sum = 0.0
    if windex < m:
        for i in range(0, windex):
            prewordstring = word[3]
            prewordstring = prewordstring[i]
            prewtow = word[4]
            prewtow = prewtow[i]
            prewvector = dswvector[prewordstring]
            sum = sum + np.dot(prewtow, prewvector)
    else:
        for i in range(m):
            prewordstring = word[3]
            prewordstring = prewordstring[i]
            prewtow = word[4]
            prewtow = prewtow[i]
            prewvector = dswvector[prewordstring]
            sum = sum + np.dot(prewtow, prewvector)
    result = math.exp((-1) * (np.dot(dtow, docvector) + np.dot(stow, senvector) + sum))
    result = (1 - 1 / (1 + result)) * vector
    return result


def gettopic():  # 得到LDA输出的文档在主题上的分布
    topic = []
    file_open = codecs.open('data/' + name + '/model-final.theta', 'r', 'utf-8')
    file_r = file_open.readlines()
    for line in file_r:
        line = line.strip()
        line = line.split(' ')
        line = [float(x) for x in line]
        line = np.array(line)
        topic.append(line)
    return topic


def iter():  # 进行迭代
    currenta = a
    print 'start-------------'
    for t in range(Iteration):
        print time.strftime(ISOTIMEFORMAT, time.localtime())
        print 'The Iteration of   ' + str(t) + ':'
        # 进行一次高斯迭代
        file_r = open('init/dswvector', 'rb')
        alldata = cPickle.load(file_r)
        file_r.close
        alldataarray = np.array(alldata.values())

        if t == 0:
            g = mixture.GMM(n_components=K, covariance_type='full', n_iter=1)
        else:
            g = mixture.GMM(n_components=K, covariance_type='full', n_iter=1, init_params='')
            g.weights_ = weight
            g.means_ = mean
            g.covars_ = covars
        g.fit(alldataarray)
        weight = g.weights_
        mean = g.means_
        covars = g.covars_
        predict = g.predict(alldataarray)
        file_w = open('update/' + name + '/weight' + str(t) + '.txt', 'w')
        file_w.write(str(g.weights_).replace('\n', '') + '\n')
        file_w.close()
        file_w = open('update/' + name + '/predict' + str(t) + '.txt', 'w')
        file_w.write(str(predict).replace('\n', '') + '\n')
        file_w.close()
        file_w = open('update/' + name + '/mean_' + str(t) + '.txt', 'w')
        for i in range(K):
            file_w.write(str(g.means_[i]).replace('\n', '') + '\n')
        file_w.close()

        # 对文档，句子，单词向量进行更新
        # 加载单词向量文档
        # file_r = open('LDAGMM/worddic', 'rb')
        # wvector = cPickle.load(file_r)
        # file_r.close
        print 'update the vector of d,s,w...'

        dvector = {}  # 存放句子的字典
        svector = {}  # 存放句子的字典
        sdistribution = {}  # 存放句子在每一个高斯成分上的分布
        wvector = {}  # 存放句子的字典
        wdistribution = {}  # 存放单词在每一个高斯成分上的分布

        for nameofupdatedoc in updatedoc:
            # 加载文档数据结构
            print 'update ' + nameofupdatedoc
            file_r = open('init/' + nameofupdatedoc, 'rb')
            doc = cPickle.load(file_r)
            file_r.close
            for s in range(len(doc[2])):
                sen = doc[2][s]
                sdistribution[sen[0]] = g.predict_proba(sen[1])
                for w in range(len(sen[2])):
                    item = sen[2][w]
                    wdistribution[item[0]] = g.predict_proba(alldata[item[0]])
                    # 学习率设置
                    if currenta > reduce:
                        currenta = currenta - (a - reduce) / (len(alldata) * 2)
                    else:
                        currenta = reduce
                    # 对文档进行更新
                    score = g.score([doc[1]])
                    d_topic = g.predict_proba([doc[1]])[0]
                    vious = diffgmm(score, doc[1], g.d_topic, g.means_, g.covars_) + diffsigmoid(doc, s, w, alldata,
                                                                                                 item[1])
                    doc[1] = doc[1] + currenta * vious
                    alldata[doc[0]] = doc[1]
                    dvector[doc[0]] = doc[1]
                    # 对文档对单词的影响更新
                    vious = diffsigmoid(doc, s, w, alldata, doc[1])
                    item[1] = item[1] + currenta * vious
                    sen[2][w] = item
                    doc[2][s] = sen
                    # 对句子向量进行更新
                    score = g.score([sen[1]])
                    s_topic = g.predict_proba([sen[1]])[0]
                    vious = diffgmm(score, sen[1], s_topic, g.means_, g.covars_) + diffsigmoid(doc, s, w, alldata,
                                                                                               item[2])
                    sen[1] = sen[1] + currenta * vious
                    doc[2][s] = sen
                    alldata[sen[0]] = sen[1]
                    svector[sen[0]] = sen[1]
                    # 对句子对单词的影响进行更新
                    vious = diffsigmoid(doc, s, w, alldata, sen[1])
                    item[2] = item[2] + currenta * vious
                    sen[2][w] = item
                    doc[2][s] = sen
                    # 对单词向量进行更新
                    score = g.score([alldata[item[0]]])
                    w_topic = g.predict_proba([alldata[item[0]]])[0]
                    vious = diffgmm(score, alldata[item[0]], w_topic, g.means_, g.covars_)
                    alldata[item[0]] = alldata[item[0]] + currenta * vious
                    wvector[item[0]] = alldata[item[0]]
                    # 对上下文对单词的影响进行更新
                    for windex in range(len(item[3])):
                        vious = diffsigmoid(doc, s, w, alldata, alldata[item[3][windex]])
                        item[4][windex] = item[4][windex] + currenta * vious
                        sen[2][w] = item
                        doc[2][s] = sen

            # 更新完毕，对结果进行保存
            file_w = open('init/' + nameofupdatedoc, 'wb')
            cPickle.dump(doc, file_w)
            file_w.close()
            # file_w = open('LDAGMM/worddic', 'wb')
            # cPickle.dump(wvector, file_w)
            # file_w.close()
        file_w = open('init/dswvector', 'wb')
        cPickle.dump(alldata, file_w)
        file_w.close()
        file_w = open('update/' + name + '/svector', 'wb')
        cPickle.dump(svector, file_w)
        file_w.close()
        # 写入更新的文档，句子，单词的更新向量
        file_w = open('update/' + name + '/dvector' + str(t) + '.txt', 'w')
        dkeys = dvector.keys()
        for key in dkeys:
            file_w.write(key + '/t' + str(dvector[key]).replace('\n', '') + '\n')
        file_w.close()

        file_w = open('update/' + name + '/svector' + str(t) + '.txt', 'w')
        skeys = svector.keys()
        for key in skeys:
            file_w.write(key + '/t' + str(svector[key]).replace('\n', '') + '\n')
        file_w.close()

        file_w = open('update/' + name + '/wvector' + str(t) + '.txt', 'w')
        wkeys = wvector.keys()
        for key in wkeys:
            file_w.write(key + '/t' + str(wvector[key]).replace('\n', '') + '\n')
        file_w.close()
        # 对句子，单词在每个高斯成分上的分布进行更新

        file_w = open('update/' + name + '/sdistribution' + str(t) + '.txt', 'w')
        skeys = sdistribution.keys()
        for key in skeys:
            file_w.write(key + '/t' + str(sdistribution[key]).replace('\n', '') + '\n')
        file_w.close()

        file_w = open('update/' + name + '/wdistribution' + str(t) + '.txt', 'w')
        wkeys = wdistribution.keys()
        for key in wkeys:
            file_w.write(key + '/t' + str(wdistribution[key]).replace('\n', '') + '\n')
        file_w.close()


def iter_with_LDA():  # 进行迭代
    currenta = a
    print 'start-------------'
    # 读取LDA的文档主题分布
    topic = gettopic()
    for t in range(Iteration):
        print time.strftime(ISOTIMEFORMAT, time.localtime())
        print 'The Iteration of   ' + str(t) + ':'
        # 进行一次高斯迭代
        file_r = open('init/dswvector', 'rb')
        alldata = cPickle.load(file_r)
        file_r.close
        alldataarray = np.array(alldata.values())
        if t == 0:
            g = mixture.GMM(n_components=K, covariance_type='full', n_iter=1)
        else:
            g = mixture.GMM(n_components=K, covariance_type='full', n_iter=1, init_params='')
            g.weights_ = weight
            g.means_ = mean
            g.covars_ = covars
        g.fit(alldataarray)
        weight = g.weights_
        mean = g.means_
        covars = g.covars_
        predict = g.predict(alldataarray)
        file_w = open('updateword2vec/' + name + '/weight' + str(t) + '.txt', 'w')
        file_w.write(str(g.weights_).replace('\n', '') + '\n')
        file_w.close()
        file_w = open('updateword2vec/' + name + '/predict' + str(t) + '.txt', 'w')
        file_w.write(str(predict).replace('\n', '') + '\n')
        file_w.close()
        file_w = open('updateword2vec/' + name + '/mean_' + str(t) + '.txt', 'w')
        for i in range(K):
            file_w.write(str(g.means_[i]).replace('\n', '') + '\n')
        file_w.close()

        # 对文档，句子，单词向量进行更新

        print 'update the vector of d,s,w...'
        dvector = {}  # 存放文档的字典
        svector = {}  # 存放句子的字典
        sdistribution = {}  # 存放句子在每一个高斯成分上的分布
        wvector = {}  # 存放单词的字典
        wdistribution = {}  # 存放单词在每一个高斯成分上的分布

        for topicindex, nameofupdatedoc in enumerate(updatedoc):
            # 加载文档数据结构
            print 'update ' + nameofupdatedoc
            file_r = open('init/' + nameofupdatedoc, 'rb')
            doc = cPickle.load(file_r)
            file_r.close
            for s in range(len(doc[2])):
                sen = doc[2][s]
                sdistribution[sen[0]] = g.predict_proba([sen[1]])
                for w in range(len(sen[2])):
                    item = sen[2][w]
                    wdistribution[item[0]] = g.predict_proba([alldata[item[0]]])
                    # 学习率设置
                    if currenta > reduce:
                        currenta = currenta - (a - reduce) / (len(alldata) * 2)
                    else:
                        currenta = reduce
                    # 对文档进行更新
                    score = g.score([doc[1]])
                    d_topic = g.predict_proba([doc[1]])
                    d_topic = d_topic[0]
                    # vious = diffgmm(score, doc[1], g.weights_, g.means_, g.covars_) + diffsigmoid_with_LDA(doc, s, w,alldata, topic[topicindex],topic[topicindex])
                    vious = diffgmm(score, doc[1], d_topic, g.means_, g.covars_) + diffsigmoid_with_LDA(doc, s, w,
                                                                                                        alldata,
                                                                                                        d_topic,
                                                                                                        d_topic)

                    doc[1] = doc[1] + currenta * vious
                    alldata[doc[0]] = doc[1]
                    dvector[doc[0]] = doc[1]

                    # 对句子向量进行更新
                    score = g.score([sen[1]])
                    s_topic = g.predict_proba([sen[1]])
                    s_topic = s_topic[0]
                    # vious = diffgmm(score, sen[1], g.weights_, g.means_, g.covars_) + diffsigmoid_with_LDA(doc, s, w,alldata,topic[topicindex], item[2])
                    vious = diffgmm(score, sen[1], s_topic, g.means_, g.covars_) + diffsigmoid_with_LDA(doc, s, w,
                                                                                                        alldata,
                                                                                                        d_topic,
                                                                                                        item[2])
                    sen[1] = sen[1] + currenta * vious
                    doc[2][s] = sen
                    alldata[sen[0]] = sen[1]
                    svector[sen[0]] = sen[1]
                    # 对句子对单词的影响进行更新
                    # vious = diffsigmoid_with_LDA(doc, s, w, alldata,topic[topicindex], sen[1])
                    vious = diffsigmoid_with_LDA(doc, s, w, alldata, d_topic, sen[1])
                    item[2] = item[2] + currenta * vious
                    sen[2][w] = item
                    doc[2][s] = sen
                    # 对单词向量进行更新
                    score = g.score([alldata[item[0]]])
                    w_topic = g.predict_proba([alldata[item[0]]])
                    w_topic = w_topic[0]
                    vious = diffgmm(score, alldata[item[0]], w_topic, g.means_, g.covars_)
                    alldata[item[0]] = alldata[item[0]] + currenta * vious
                    wvector[item[0]] = alldata[item[0]]
                    # 对上下文对单词的影响进行更新
                    for windex in range(len(item[3])):
                        # vious = diffsigmoid_with_LDA(doc, s, w, alldata,topic[topicindex], alldata[item[3][windex]])
                        vious = diffsigmoid_with_LDA(doc, s, w, alldata, d_topic, alldata[item[3][windex]])
                        item[4][windex] = item[4][windex] + currenta * vious
                        sen[2][w] = item
                        doc[2][s] = sen

            # 更新完毕，对结果进行保存
            file_w = open('init/' + nameofupdatedoc, 'wb')
            cPickle.dump(doc, file_w)
            file_w.close()
            # file_w = open('LDAGMM/worddic', 'wb')
            # cPickle.dump(wvector, file_w)
            # file_w.close()
        file_w = open('init/dswvector', 'wb')
        cPickle.dump(alldata, file_w)
        file_w.close()
        file_w = open('updateword2vec/' + name + '/svector', 'wb')
        cPickle.dump(svector, file_w)
        file_w.close()
        # 写入更新的文档，句子，单词的更新向量
        file_w = open('updateword2vec/' + name + '/dvector' + str(t) + '.txt', 'w')
        dkeys = dvector.keys()
        for key in dkeys:
            file_w.write(key + '/t' + str(dvector[key]).replace('\n', '') + '\n')
        file_w.close()

        file_w = open('updateword2vec/' + name + '/svector' + str(t) + '.txt', 'w')
        skeys = svector.keys()
        for key in skeys:
            file_w.write(key + '/t' + str(svector[key]).replace('\n', '') + '\n')
        file_w.close()

        file_w = open('updateword2vec/' + name + '/wvector' + str(t) + '.txt', 'w')
        wkeys = wvector.keys()
        for key in wkeys:
            file_w.write(key + '/t' + str(wvector[key]).replace('\n', '') + '\n')
        file_w.close()
        # 对句子，单词在每个高斯成分上的分布进行更新

        file_w = open('updateword2vec/' + name + '/sdistribution' + str(t) + '.txt', 'w')
        skeys = sdistribution.keys()
        for key in skeys:
            file_w.write(key + '/t' + str(sdistribution[key]).replace('\n', '') + '\n')
        file_w.close()

        file_w = open('updateword2vec/' + name + '/wdistribution' + str(t) + '.txt', 'w')
        wkeys = wdistribution.keys()
        for key in wkeys:
            file_w.write(key + '/t' + str(wdistribution[key]).replace('\n', '') + '\n')
        file_w.close()


# 初始化随机向量
print 'start'
ISOTIMEFORMAT = '%Y-%m-%d %X'
print time.strftime(ISOTIMEFORMAT, time.localtime())
# init()
# 去停用词和频率低的词的初始化
init_with_stop_and_frequence()
# 进行更新迭代
# iter()
# 进行LDA迭代
iter_with_LDA()
print time.strftime(ISOTIMEFORMAT, time.localtime())
# ldagmm()



