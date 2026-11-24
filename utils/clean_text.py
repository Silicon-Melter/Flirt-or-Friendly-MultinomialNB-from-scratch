import re
import json
import contractions
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

spell = SpellChecker()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

with open('slang.json', 'r') as f:
    slang_words = json.load(f)

class CleanText():
    def __init__(self):
        self.text = ""

    def input_text(self, text):
        self.text = text
        
    def lower_case(self):
        self.text = self.text.lower()

    def expand_contractions(self):
        self.text = contractions.fix(self.text)

    def remove_punctuations(self):
        punc = []
        for ch in self.text:
            if ch not in string.punctuation:
                punc.append(ch)
        self.text = "".join(punc)

    def flag_emoticon(self):
        #emoticons carry tone, keep them as tokens before punctuation is removed
        emoticons = {";)": "_wink", ";-)": "_wink", "😉": "_wink",
                     ":)": "_smile", ":-)": "_smile", "😊": "_smile",
                     ":p": "_tongue", ":d": "_laugh", "😂": "_laugh",
                     "<3": "_heart", "❤": "_heart", "😍": "_heart", "🥰": "_heart",
                     ":*": "_kiss", "😘": "_kiss", "😏": "_smirk"}
        for emo, token in emoticons.items():
            self.text = self.text.replace(emo, f" {token} ")

    def flag_elongate(self):
        if not self.text: return self.text
        result = []
        for word in self.text.split():
            if re.search(r'(.)\1{2,}', word):
                word = re.sub(r'(.)\1{2,}', r'\1', word)
                word = f"{word} _elongated"
                result.append(word)
            else:
                result.append(word)
        self.text = " ".join(result)

    def flag_slang(self):
        if not self.text: return self.text
        result = []
        for word in self.text.split():
            if word in slang_words:
                result.append(slang_words[word] + " _slang") 
            else:
                result.append(word)
        self.text = " ".join(result)
        
    def lemme(self):
        if not self.text: return self.text
        result = []
        for word in self.text.split():
            if word.startswith('_'):
                result.append(word)
            else:
                lemma = lemmatizer.lemmatize(word, pos='v')
                result.append(lemma)
        self.text = " ".join(result)
        
    def remove_stopwords(self):
        if not self.text: return self.text
        result = []
        for word in self.text.split():
            if word not in stop_words:
                result.append(word)
        self.text = " ".join(result)

    def flag_negative(self):
        negation_triggers = {"not", "no", "never"} 
        result = []
        words = self.text.split()
        i = 0
        while i < len(words):
            word = words[i]
            if word in negation_triggers and i + 1 < len(words):
                next_word = words[i + 1]
                result.append(f"NOT_{next_word}")
                i += 2 
            else:
                result.append(word)
                i += 1
                
        self.text = " ".join(result)

    def get_result(self):
        return self.text