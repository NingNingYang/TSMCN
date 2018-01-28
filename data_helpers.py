#! usr/bin/python
#coding=utf-8
import csv
import pickle
import json
import os
import numpy as np
import thulac

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

class DataProcessor(object):
    def __init__(self,datafile, rnd_seed):
        self.datafile = datafile
        self.docs, self.labels, self.column_names = self.readdatafile(self.datafile)
        self.vocab = self.genVocab(self.docs)
        #self.char_count = self.countcharnumber(self.docs, self.vocab)
        self.index_docs = self.word2index(self.docs, self.vocab)
        random_order = np.random.permutation(len(self.labels))
        self.labels = self.labels[random_order, :]
        self.index_docs = self.index_docs[random_order, :]


    @staticmethod
    def readdatafile(datafile):
        thul = thulac.thulac(seg_only=True)
        if datafile.endswith('csv'):
            f = file(datafile, "rb")
            reader = csv.reader(f)
            lines = [line for line in reader]
        else:
            raise NotImplementedError
        lines = [line for line in lines if len(line[3])<=450] #lines[0] is the name of categories
        docs = []
        for line in lines:
            line.pop(70) # Location
            line.pop(29) # InjDate
            line.pop(4) # Answer
            docs.append(line.pop(3)) # Problem
            line.pop(2) # CaseID
            line.pop(1) # Username
            line.pop(0) # Id
        names = lines[0]
        labels = np.array(lines[1:], dtype='int')
        docs = [thul.cut(doc.decode('gb18030').encode('utf-8')) for doc in docs[1:]]
        docs_re = []
        for item in docs:
            docs_re.append([''.join(word) for word in item])
        return docs_re, labels, names


    @staticmethod
    def genVocab(lines, maskid=0 ):
        """generate vocabulary from contents"""
        #lines = [' '.join(line) for line in lines]
        wordset = set(item for line in lines for item in line)
        word2index = {word: index + 1 for index, word in enumerate(wordset)}
        word2index['mask'] = maskid
        word2index['unknown'] = len(word2index)
        return word2index


    @staticmethod
    def countcharnumber(docs, vocab):
        count = [0]*2210
        docs = [' '.join(line) for line in docs]
        for doc in docs:
            for char in doc.strip().split():
                if vocab.has_key(char):
                    count[vocab[char]] = count[vocab[char]] + 1
                else:
                    count[2089] = count[2089] + 1
        file = open('count.txt','w')
        pickle.dump(count, file, 1)
        file.close()
        return count


    @staticmethod
    def word2index(docs, vocab):
        #docs = [' '.join(line) for line in docs]
        index_docs = []
        for doc in docs:
            index_doc = []
            for word in doc:
                if vocab.has_key(word):
                    index_doc.append(vocab[word])
                else:
                    index_doc.append(vocab['unknown'])
            index_docs.append(index_doc)
        max_len = 212
        index_docs = [doc+[vocab['mask']]*(max_len - len(doc)) for doc in index_docs]
        index_docs = np.array(index_docs)
        return index_docs

    def getInputs(self):
        return self.index_docs

    def getLabels(self):
        return self.labels

    def getVocab(self):
        return self.vocab

    def getEmbeddings(self):
        return self.embeddings

    def getdocs(self):
        return self.docs
        #return self.reorder_docs
    def getCharCount(self):
        self.char_count


class DataSelector(DataProcessor):
    """
    This is for select data cases of certain category.
    """
    def __init__(self, datafile, rnd_seed, select_columns):
        super(DataSelector, self).__init__(datafile, rnd_seed)
        self.select(select_columns)

    def select(self, select_columns):
        column_index = []
        for column_name in select_columns:
            column_index.append(self.column_names.index(column_name))
        self.labels = self.labels[:, column_index]
        assert len(self.labels) == len(self.index_docs)


if __name__ == "__main__":
    datafile = "./data/feature20170609_original.csv"
    select_columns = [
        'GetPay', 'AssoPay', 'WorkTime',  'WorkPlace', 'JobRel', 'DiseRel', 'OutForPub',
        'OnOff', 'InjIden', 'EndLabor', 'LaborContr', 'ConfrmLevel', 'Level', 'Insurance', 'HaveMedicalFee'
    ]
    data = DataSelector(datafile, 1234, select_columns)
    labels = data.getLabels()
    doc_indexes = data.getInputs()
    vocab = data.getVocab()
    docs =  data.getdocs()

