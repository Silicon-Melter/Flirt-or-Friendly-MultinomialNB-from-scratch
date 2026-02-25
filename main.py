from utils.handle_csv import HandleCSV
from utils.clean_text import CleanText
from utils.classifier import MultinomialNB

handlecsv = HandleCSV()
df = handlecsv.get_df("data.csv")

cleaner = CleanText()
def apply_clean(text):
        cleaner.input_text(text)
        cleaner.lower_case()
        cleaner.expand_contractions()
        cleaner.remove_punctuations()
        cleaner.flag_negative()
        cleaner.remove_stopwords()
        cleaner.flag_elongate()
        cleaner.flag_slang()
        cleaner.lemme()
        return cleaner.get_result()

df['cleaned_text'] = df['text'].apply(apply_clean)

classify = MultinomialNB()
X = df['cleaned_text']
y = df['label']
    
classify.input_df(X, y)
classify.train()
    
classify.test()
    
classify.plot_confusion_matrix()