import re
import pandas as pd


# quotes = pd.read_csv("quotes.txt", sep = "sbvjh")
# quotes_300 = quotes.sample(n=300)
# quotes_300['Quote'] = quotes_300['Quote'].apply(lambda quote: re.sub('[^A-Za-z0-9 ]+', '', quote.lower()))

# quotes_300.to_csv('quotes_300.csv', index=False)


def ngrams(word):
    ngrams = []
    for n in range(len(word) + 1):
        ngrams += [word[i:i + n] for i in range(len(word) - n + 1)]
    return list(filter(None, ngrams))


def niewiadomski(word1, word2):
    N = max(len(word1), len(word2))
    ngrams1 = ngrams(word1)
    h = [1 if ngram in ngrams1 else 0 for ngram in ngrams(word2)]
    return 2 * sum(h) / (pow(N, exp=2) + N)


# print(niewiadomski("lalka", "lalka")) # 1
# print(niewiadomski("outlier", "outlaw")) # 0.3751


def niewiadomski_sentence(sent1, sent2):
    sent1 = sent1.split()
    sent2 = sent2.split()
    N = max(len(sent1), len(sent2))
    if len(sent1) > len(sent2):
        g = [max([niewiadomski(word1, word2) for word2 in sent2]) for word1 in sent1]
    else:
        g = [max([niewiadomski(word1, word2) for word2 in sent1]) for word1 in sent2]
    return 1 / N * sum(g)


# print(niewiadomski_sentence("Hello you", "Hello you")) #1
# print(niewiadomski_sentence("WIDOCZNOSC NA DRODZE DOBRA", "WIDOCZNOSC DOBRA")) #0.5


quotes = pd.read_csv("quotes_300.csv")

similarities = [[row['Quote']] + [niewiadomski_sentence(row['Quote'], row2['Quote']) for _, row2 in quotes.iterrows()]
                for _, row in quotes.iterrows()]

df = pd.DataFrame(similarities)
df.to_excel("quotes_similarities.xlsx", sheet_name='Sheet_name_1')
