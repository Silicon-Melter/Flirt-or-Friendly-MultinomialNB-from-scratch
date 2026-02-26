import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

class MultinomialNB():
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.labels = [] 
        self.logprior = {}
        self.loglikelihood = {}
        self.results = {}

    def input_df(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        self.X_train = np.array(X_train) 
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
    
    def train(self):
        ind = {}
        texts = {}
        logprior = {}
        loglikelihood = {}
        word_counts = {}
        total_words = {}
        self.vocab = []
        
        #get labels/classes
        self.labels = np.unique(self.y_train)
        
        #seperate text by class
        for label in self.labels:
            ind[label] = np.where(self.y_train == label)[0]
            texts[label] = self.X_train[ind[label]]


        for label in self.labels:
            word_counts[label] = {}
            #cal log prior
            logprior[label] = math.log(len(ind[label]) / len(self.y_train))

            #sum word count in each class
            #collect vocab
            for text in texts[label]:
                words = text.split() 
                for word in words:
                    self.vocab.append(word)
                    word_counts[label][word] = word_counts[label].get(word, 0) + 1

            total_words[label] = sum(word_counts[label].values())
            
        self.vocab = np.unique(self.vocab)
        vocab_size = len(self.vocab)

        #cal likelihood
        for label in self.labels:
            loglikelihood[label] = {}
            for word in self.vocab:
                count = word_counts[label].get(word, 0)
                loglikelihood[label][word] = math.log((count + 1) / (total_words[label] + vocab_size))
        
        self.logprior = logprior
        self.loglikelihood = loglikelihood

    def predict(self, text):
        test_words = text.split()
            
        sum_prob = {}
        post_prob = {}

        #cal post prob each class
        for label in self.labels:
            sum_prob[label] = self.logprior[label]
            for word in test_words:
                if word in self.vocab:
                    sum_prob[label] += self.loglikelihood[label][word]
                        
            post_prob[label] = sum_prob[label]

        #prediction = max prob
        value = max(list(post_prob.values()))
        for key, val in post_prob.items():
            if val == value:
                keys = key
        predicted_label = keys
        return predicted_label, value

    def test(self):
        correct_predictions = 0
        tp = tn = fp = fn = 0
        for i in range(len(self.X_test)):
            predicted_label, _ = self.predict(self.X_test[i])
            
            if predicted_label == self.y_test[i] and predicted_label == 0:
                correct_predictions += 1
                tn += 1 
            elif predicted_label == self.y_test[i] and predicted_label == 1:
                correct_predictions += 1
                tp += 1 
            elif predicted_label == 0:
                fn +=1
            elif predicted_label == 1:
                fp +=1
                
        self.results = {"TP":tp,"TN":tn,"FP":fp,"FN":fn}
        accuracy = correct_predictions / len(self.y_test)
        print(f"Accuracy: {accuracy * 100:.2f}%")

    def get_features(self):
        likelihood = {}
        max_val = {}
        #get top 3 most prob word of each class
        ####todo get most diff prob instead
        for label in self.labels:
            likelihood[label] = list(self.loglikelihood[label].values())
            max_val[label] = {}
            for i in range(0,3):
                value = max(likelihood[label])
                keys = [key for key, val in self.loglikelihood[label].items() if val == value]
                max_val[label][keys[0]] = value
                likelihood[label].remove(max(likelihood[label]))
            print(max_val[label])
    
    def plot_confusion_matrix(self):
        tp = self.results["TP"]
        tn = self.results["TN"]
        fp = self.results["FP"]
        fn = self.results["FN"]
        
        cm = np.array([[tn, fp],
                       [fn, tp]])
        
        print(f"Accuracy: {(tp+tn)/(tp+tn+fp+fn):.4f}")
        print(f"Precision: {(tp)/(tp+fp):.4f}")
        print(f"Recall: {(tp)/(tp+fn):.4f}")
        print(f"Specificity: {(tn)/(tn+fp):.4f}")
        print(f"F1 Score: {(2*tp)/(2*tp+fp+fn):.4f}")
        
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])
        cm_display.plot()
        plt.show()