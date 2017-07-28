import collect_rss
import numpy as np
import string
from nltk import word_tokenize


def get_articles(url="rss_links.txt", filepath=None):
    """
    Obtains articles from RSS feed of Reuters.
    
    Parameters
    ----------
    url: String
        Filepath to URL of Reuters RSS feed.
    filepath: String
        Filepath of place to store articles.
    
    Returns
    -------
    List of strings
    """
    thing = []
    with open(url, 'r') as f:
        all_links = f.read().splitlines()
    for link in all_links:
        texts = collect_rss.collect(link, filepath)
        thing += list(texts.values())
    return thing


def tokenize(text, preserve_case=True, filter_text=True, stopwords=None, punc=string.punctuation):
    """
    Tokenizes a section of text and filters out punctuation and stopwords.

    Parameters
    ----------
    text: String
        The section of text to be processed.
    preserve_case: Boolean
        Whether or not capital letters should be preserved.
    filter_text: Boolean
        Whether or not stopwords and punctuation should be removed.
    stopwords: List
        List of stopwords to be filtered out. Leave as None if default should be used.
    punc: String or List
        String or List of punctuation to be filtered out.

    Return
    ------
    List of strings
    """
    if filter_text and stopwords is None:
        with open("stopwords.txt", 'r') as r:
            stops = []
            for line in r:
                stops += [i.strip() for i in line.split('\t')]
        stopwords = stops
    tokens = word_tokenize(text)
    indices = []
    if filter_text:
        for i in range(len(tokens)):
            word = tokens[i]
            # print(word, word in punc, word.lower() in stopwords)
            if word in punc or word.lower() in stopwords:
                indices.append(i)
        tokens = list(np.delete(np.array(tokens), np.array(indices)))
    if preserve_case:
        return tokens
    return [i.lower() for i in tokens]


def for_ner(url="rss_links.txt", filter_text=False):
    """
    Returns tokens as formatted for NER (capitalization preserved).

    Parameter
    ---------
    url: String
        URL for RSS feed from which to pull articles.

    Return
    ------
    List of strings
    """
    articles = get_articles(url)
    return [tokenize(tx, filter_text=filter_text) for tx in articles]


def for_search(url="rss_links.txt"):
    """
    Returns input formatted for search engine.

    Parameter
    ---------
    url: String
        URL for RSS feed from which to pull articles.

    Return
    ------
    List of tuples
        Tuples length 3:
            - String (raw text)
            - List of strings (tokens, lower case)
            - List of strings (unfiltered tokens)
    """
    articles = get_articles(url)
    tokens = [tokenize(tx, preserve_case=False) for tx in articles]
    unfiltered = [tokenize(tx, filter_text=False) for tx in articles]
    return list(zip(articles, tokens, unfiltered))

