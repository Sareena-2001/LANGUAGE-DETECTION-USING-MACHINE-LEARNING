# Language Detection Using Machine Learning

A machine-learning project that identifies the language of a text sample across **17 languages**, comparing five classical classifiers and two feature representations (word-level bag of words vs. character n-gram TF-IDF).

## Highlights

| Pipeline | Best model | Test accuracy | Macro F1 |
|---|---|---|---|
| A. Word-level bag of words (baseline) | Random Forest | 85.88% | 81.52% |
| B. Character n-gram (1–3) TF-IDF (proposed) | Logistic Regression | **98.69%** | **98.82%** |

5-fold stratified cross-validation: char n-gram + Linear SVM reaches **98.90% ± 0.16**.

**Key finding:** the baseline's `str.isalnum()` filter drops words that contain Unicode combining vowel signs. This removes about 95% of tokens in Kannada, Tamil and Malayalam (and 83% in Hindi), so the baseline nearly fails on those languages. Character n-grams on raw text fix this.

## Dataset

- File: `Language Detection.csv` (also zipped in `archive (2).zip`)
- Source: Kaggle "Language Detection" dataset
- 10,337 rows, columns `Text` and `Language`
- Languages: English, French, Spanish, Portuguese, Italian, Russian, Swedish, Malayalam, Dutch, Arabic, Turkish, German, Tamil, Danish, Kannada, Greek, Hindi
- Note: labels are spelled `Portugeese` and `Sweedish` in the raw data.

## Repository contents

| File | Description |
|---|---|
| `Language_detection.ipynb` | Original Colab notebook (baseline pipeline) |
| `language_detection.py` | Script export of the notebook |
| `Language Detection.csv` | Dataset |
| `archive (2).zip` | Zipped copy of the dataset |

## Methods

**Pipeline A (baseline):** remove URLs → NLTK tokenize → keep `isalnum()` tokens → lowercase → Porter stemming + WordNet lemmatization → `CountVectorizer`.

**Pipeline B (proposed):** raw text → lowercase → `TfidfVectorizer(analyzer="char_wb", ngram_range=(1,3), sublinear_tf=True, min_df=2)`.

**Classifiers:** Multinomial Naive Bayes, Logistic Regression, SVC (RBF), Decision Tree, Random Forest (scikit-learn). Split: 80/20, `random_state=42`.

## Results (test set, 2,068 samples)

| Classifier | Accuracy A | Accuracy B |
|---|---|---|
| Multinomial NB | 85.54% | 90.28% |
| Logistic Regression | 85.01% | 98.69% |
| SVC (RBF) | 81.38% | 98.79% |
| Decision Tree | 78.97% | 89.07% |
| Random Forest | 85.88% | 97.44% |

## Getting started

```bash
pip install pandas numpy scikit-learn nltk matplotlib
python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','wordnet','omw-1.4']]"
```

In `language_detection.py`, change the dataset path from `/content/sample_data/Language Detection.csv` to your local path, then run:

```bash
python language_detection.py
```

Or open `Language_detection.ipynb` in Jupyter or Google Colab.

### Quick start with the improved model

```python
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("Language Detection.csv")
model = make_pipeline(
    TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), sublinear_tf=True, min_df=2),
    LogisticRegression(max_iter=1000),
)
model.fit(df["Text"], df["Language"])
print(model.predict(["Bonjour, comment allez-vous ?", "Wie geht es dir?"]))
```

## Known issue in the original code

The classification report and confusion-matrix axes use `data['Language'].unique()` (order of appearance), while scikit-learn sorts labels alphabetically, so per-class names in those outputs are mismatched. Use `sorted(data['Language'].unique())` or `classifier.classes_` instead. Overall accuracy is unaffected.

## Future work

- Evaluate on short, informal and code-switched text
- Add confusable language pairs (e.g., Norwegian, Malay/Indonesian)
- Compare with langid.py, fastText and multilingual transformers
- Deploy as a lightweight web API

## Tech stack

Python · pandas · NumPy · scikit-learn · NLTK · Matplotlib
