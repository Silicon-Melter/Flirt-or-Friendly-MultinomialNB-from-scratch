import pandas as pd

class HandleCSV():
    def __init__(self):
        self.filename = ""
        self.df = None
        
    def clean_csv(self):
        df = pd.read_csv(self.filename)
        df.dropna(inplace=True, ignore_index=True)
        df.drop_duplicates(subset=["text"], inplace=True, ignore_index=True)
        
        count_neg = df['label'].value_counts().get(0, 0)
        count_pos = df['label'].value_counts().get(1, 0)
        delta = count_neg - count_pos

        indices_to_drop = df[df['label'] == 0].sample(n=delta, random_state=42).index
        df_balance = df.drop(indices_to_drop)
        df_balance = df_balance.reset_index(drop=True)
        
        self.df = df_balance

    def get_df(self, filename):
        self.filename = filename
        self.clean_csv()
        return self.df