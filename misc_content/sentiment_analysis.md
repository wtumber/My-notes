
# Sentiment Analysis with NLTK and Python
Note: All code in this document is from RealPython:[Python NLTK sentiment analysis](https://realpython.com/python-nltk-sentiment-analysis/#:~:text=Sentiment%20analysis%20is%20the%20practice,obtain%20insights%20from%20linguistic%20data.)

## Code snippets
Your input data should be a list of words, with stopwords removed via `words = [w for w in words if w.lower() not in stopwords]`, then tokenize words or create lemmas.

To build a freq dist of words:

```Python
words: list[str] = nltk.word_tokenize(text)
fd = nltk.FreqDist(words)
```

Use `fd.most_common(3)` and `fd.tabulate(3)` to view the results.

Convert text to lower with `lower_fd = nltk.FreqDist([w.lower() for w in fd])`.

### Collocation (N-grams)

```Python
finder = nltk.collocations.TrigramCollocationFinder.from_words(words)

finder.ngram_fd.tabulate(2)
```

### Pre-trained sentiment analysis finder

```Python
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
sia.polarity_scores("Wow, NLTK is really powerful!")

```

With full polarity:

```Python
from statistics import mean

def is_positive(review_id: str) -> bool:
    """True if the average of all sentence compound scores is positive."""
    text = nltk.corpus.movie_reviews.raw(review_id)
    scores = [
        sia.polarity_scores(sentence)["compound"]
        for sentence in nltk.sent_tokenize(text)
    ]
    return mean(scores) > 0

```

### Creating your own set of positive and negative terms

```Python
unwanted = nltk.corpus.stopwords.words("english")
unwanted.extend([w.lower() for w in nltk.corpus.names.words()])

def skip_unwanted(pos_tuple):
    word, tag = pos_tuple
    if not word.isalpha() or word in unwanted:
        return False
    if tag.startswith("NN"):
        return False
    return True

positive_words = [word for word, tag in filter(
    skip_unwanted,
    nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["pos"]))
)]
negative_words = [word for word, tag in filter(
    skip_unwanted,
    nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["neg"]))
)]

```

You can also use scikit-learn classifiers for sentiment analysis.

