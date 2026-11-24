# Flirt or Friendly: Multinomial Naive Bayes from scratch

A text classifier that tells flirty chat messages (label 1) from friendly ones (label 0). The Naive Bayes model (log priors, Laplace-smoothed log likelihoods, prediction by the larger log posterior) is written from scratch with NumPy in `utils/classifier.py`. Course project for 01416305 Artificial Intelligence Technology, KMITL.

## Dataset
`data.csv` is the "Flirty or Not" dataset from IE University, about 5,000 short messages labelled flirty or friendly:
- Hugging Face: [ieuniversity/flirty_or_not](https://huggingface.co/datasets/ieuniversity/flirty_or_not)
- Reference project: [alyssafrndz/Flirtation-analysis](https://github.com/alyssafrndz/Flirtation-analysis)

Credit for collecting and labelling the data goes to its authors. This repo only adds the classifier and the cleaning experiments.

## Pipeline
1. `utils/handle_csv.py`: drop empty rows and duplicate texts, then balance the classes by dropping friendly messages at random (1,055 of each).
2. `main.py` `apply_clean`: lower case, expand contractions, flag emoticons, remove punctuation, join negations (`NOT_cute`), flag elongated words (`heyyy` → `hey _elongated`), translate slang (`slang.json`), lemmatise verbs.
3. `utils/classifier.py`: 80/20 train/test split, train, test, confusion matrix.

## Results
Test set of 422 messages (`random_state=1`), plus the mean ± standard deviation over 10 different balance/split seeds:

| apply_clean | Accuracy (seed 1) | F1 (seed 1) | Accuracy, 10 seeds | F1, 10 seeds |
|---|---|---|---|---|
| lower case only | 74.6% | 0.752 | 74.4 ± 1.8% | 0.746 ± 0.020 |
| first version (removes stopwords) | 73.2% | 0.749 | 71.3 ± 1.4% | 0.721 ± 0.016 |
| first version + emoticons | 73.5% | 0.753 | 71.9 ± 1.3% | 0.727 ± 0.016 |
| **current: emoticons, keep stopwords** | **77.3%** | **0.777** | **75.1 ± 1.7%** | **0.750 ± 0.019** |

Current version on seed 1: precision 0.746, recall 0.811, specificity 0.736.

What the experiment showed:
- **Removing stopwords hurt** by about 3.7 points. In chat, small words carry the tone: "your", "me", "over" and "down" lean flirty, "he" and "they" lean friendly. Standard stopword lists were built for documents, not messages.
- **Emoticons are signal**: `;)` appears in 41 messages and every one is labelled flirty. Removing punctuation used to delete it.
- Most other steps (contractions, negation, elongation, slang, lemmatising) each moved accuracy by less than one standard deviation on their own. With about 2,100 messages, the test set is small, so differences under ~2 points are noise.

Next: word n-grams, and word embeddings that place similar words close together, which bag-of-words Naive Bayes cannot do.

## Run
```
pip install pandas numpy scikit-learn nltk contractions pyspellchecker matplotlib
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
python main.py
```
