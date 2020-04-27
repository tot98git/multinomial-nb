from collections import Counter, defaultdict
import numpy as np 
import pandas as pd
import re
from time import time  
import pickle

class MultinominalNB():
    def __init__(self):
        self.docs = []
        self.classes = []
        self.vocab = []
        self.logprior = dict()
        self.class_vocab = dict()
        self.loglikelihood = dict()
  
    def countCls(self, cls):
        cnt = 0
        for idx, _docs in enumerate(self.docs):
            if (self.classes[idx] == cls):
                cnt += 1

        return cnt

    def buildGlobalVocab(self):
        vocab = []
        for doc in self.docs:
            vocab.extend(self.cleanDoc(doc)) 

        return np.unique(vocab)

    def buildClassVocab(self, _cls):
        curr_word_list = []
        for idx, doc in enumerate(self.docs):
            if self.classes[idx] == _cls:
                curr_word_list.extend(self.cleanDoc(doc))

        if _cls not in self.class_vocab:
            self.class_vocab[_cls]=curr_word_list
        else:
            self.class_vocab[_cls].append(curr_word_list)

    @staticmethod
    def cleanDoc(doc):
        return re.sub(r'[^a-z\d ]', '', doc.lower()).split(' ')

    def fit(self, x, y, save = False):
        self.docs = x
        self.classes = y
        num_doc = len(self.docs)
        uniq_cls = np.unique(self.classes)
        self.vocab = self.buildGlobalVocab()
        vocab_cnt = len(self.vocab)

        t = time()

        for _cls in uniq_cls:
            cls_docs_num = self.countCls(_cls)
            self.logprior[_cls] = np.log(cls_docs_num/num_doc)
            self.buildClassVocab(_cls)
            class_vocab_counter = Counter(self.class_vocab[_cls])
            class_vocab_cnt = len(self.class_vocab[_cls])

            for word in self.vocab:
                w_cnt = class_vocab_counter[word]
                self.loglikelihood[word, _cls] = np.log((w_cnt + 1)/(class_vocab_cnt + vocab_cnt))
        
        if save:
            self.saveModel()

        print('Training finished at {} mins.'.format(round((time() - t) / 60, 2)))


    def saveModel(self):
        try:
            f = open("models/classifier", "wb")
            pickle.dump([self.logprior, self.vocab, self.loglikelihood, self.classes], f)
            f.close()
        except:
            print('Error saving the model')
    
    @staticmethod
    def readModel():
        try: 
            f = open("models/classifier", "rb")
            model = pickle.load(f)
            f.close()
            return model
        except:
            print('Error reading the model')
        
        
    def predict(self,test_docs, cached = False):
        output = []

        if not cached:
            logprior = self.logprior
            vocab = self.vocab
            loglikelihood = self.loglikelihood
            classes = self.classes
        else:
            logprior, vocab, loglikelihood, classes = self.readModel()

        for doc in test_docs:
            uniq_cls = np.unique(classes)
            sum = dict()

            for  _cls in uniq_cls:
                sum[_cls] = logprior[_cls]

                for word in self.cleanDoc(doc):
                    if word in vocab:
                        try:
                            sum[_cls] += loglikelihood[word, _cls]
                        except:
                            print(sum, _cls)

            result = np.argmax(list(sum.values()))
            output.append(uniq_cls[result])

        return output
    


class Implementation():
    def __init__(self):
        self.labels = dict()

    @staticmethod
    def accuracy(prediction, test):
        acc = 0
        test_list = list(test)
        for idx, result in enumerate(prediction):
            if result == test_list[idx]:
                acc += 1

        return acc/len(test)

    def main(self):
        x_train, y_train, x_test, y_test = self.readFile(size = 1000, testSize=0.3)
        nb = MultinominalNB()

        """ 
        Run the code below the first time you run the script
        nb.fit(x_train, y_train) 
        """        
        nb.fit(x_train, y_train, save = True) 
        predictions = nb.predict([x_test], cached = False)

        print('Accuracy: ', predictions, self.accuracy(predictions, y_test))
      
    def readFile(self, size = 70000, testSize = 0.3):
        lines = pd.read_csv("data/news-aggregator.csv", nrows = size);
        x = lines.TITLE
        y = lines.CATEGORY
        skip = round(size * (1 - testSize))
        x_train, y_train, x_test, y_test = x[:skip], y[:skip], x[skip:size], y[skip:size]

        print('Train data: ', len(x_train), 'Testing data: ', len(x_test), 'Total: ', len(x))

        return x_train, y_train, x_test, y_test

if __name__=="__main__":
    Implementation().main()