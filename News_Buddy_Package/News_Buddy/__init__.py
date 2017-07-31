import feedparser
import justext
import pickle
import requests
import sys
import numpy as np
import string
from nltk import word_tokenize, sent_tokenize
from collections import defaultdict, Counter
from os import path, makedirs

_path = Path(path.dirname(path.abspath(__file__)))

__all__ = ['get_text', 'collect', 'get_articles', 'tokenize', 'for_ner'
'for_search', 'extract_single', 'create_database', 'update_single', 'new_database', 
'write database', 'retrieve_database', 'clear_database', 'top_related', 'update', 
'common_entities', 'collect_articles', 'get_data', 'initialize', 'clear_database',
'q1', 'q2', 'q3']

db = retrieve_database()
DATABASE_FR = "data/search_engine_data.txt"

def get_text(link):
    response = requests.get(link)
    paragraphs = justext.justext(response.content, justext.get_stoplist("English"))
    text = "\n\n".join([p.text for p in paragraphs if not p.is_boilerplate])
    return text


def collect(url, filename):
    # read RSS feed
    d = feedparser.parse(url)

    # grab each article
    texts = {}
    for entry in d["entries"]:
        link = entry["link"]
        print("downloading: " + link)
        text = get_text(link)
        texts[link] = text

    # pickle
    if filename is not None:
        pickle.dump(texts, open(filename, "wb"))

    return texts

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
        texts = collect(link, filepath)
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

class SearchEngine():
    def __init__(self):
        # Dict[str, str]: maps document id to original/raw text
        self.raw_text = {}

        # Dict[str, Counter]: maps document id to term vector (counts of terms in document)
        self.term_vectors = {}

        # Counter: maps term to count of how many documents contain term
        self.doc_freq = Counter()

        # Dict[str, set]: maps term to set of ids of documents that contain term
        self.inverted_index = defaultdict(set)

        self.filtered_tokens = {}
        self.unfiltered_tokens = {}

    def filtered_tokenize(self, text):
        """ Tokenizes the text, filters out the punctuation, and converts
            text to lower case.

            Parameters
            ----------
            text: str
                The text of the document to be tokenized.

            Returns
            -------
            the_tokens: list[str]
                A list of tokenized words.

        """

        tokens = word_tokenize(text)

        the_tokens = [token.lower() for token in tokens if not token in string.punctuation]

        return the_tokens

    def unfiltered_tokenize(self, text):
        """ Tokenizes the text, but keeps the punctuation and case of the original
            text

            Parameters
            ----------
            text: str
                The text of the document to be tokenized.

            Returns
            -------
            the_tokens: list[str]
                A list of tokenized words.

        """

        the_tokens = word_tokenize(text)
        return the_tokens

    def add(self, id, text, ftokens, uftokens):
        """ Adds document to index. Stores the filtered and unfiltered tokens
            as attributes of that document

            Parameters
            ----------
            id: str
                A unique identifier for the document to add, e.g., the URL of a webpage.
            text: str
                The text of the document to be indexed.
            ftokens: list
                The list of tokenized words WITH their punctuation removed
            uftokens: list
                The list of tokenized words WITHOUT their punctuation removed
        """

        self.filtered_tokens[id] = ftokens
        self.unfiltered_tokens[id] = uftokens
        # check if document already in collection and throw exception if it is
        if id in self.raw_text:
            raise RuntimeError("document with id [" + id + "] already indexed.")

        # store raw text for this doc id
        self.raw_text[id] = text

        # create term vector for document (a Counter over tokens)
        term_vector = Counter(ftokens)

        # store term vector for this doc id
        self.term_vectors[id] = term_vector

        # update inverted index by adding doc id to each term's set of ids
        for term in term_vector.keys():
            self.inverted_index[term].add(id)

        # update document frequencies for terms found in this doc
        # i.e., counts should increase by 1 for each (unique) term in term vector
        self.doc_freq.update(term_vector.keys())

    def remove(self, id):
        """ Removes document from index.

            Parameters
            ----------
            id: str
                The identifier of the document to remove from the index.
        """
        # check if document doesn't exists and throw exception if it doesn't
        if not id in self.raw_text:
            raise KeyError("document with id [" + id + "] not found in index.")

        # remove raw text for this document
        del self.raw_text[id]

        # remove filtered tokens for this document
        del self.filtered_tokens[id]

        # remove unfiltered tokens for this document
        del self.unfiltered_tokens[id]

        # update document frequencies for terms found in this doc
        # i.e., counts should decrease by 1 for each (unique) term in term vector
        self.doc_freq.subtract(self.term_vectors[id].keys())

        # update inverted index by removing doc id from each term's set of ids
        for term in self.term_vectors[id].keys():
            self.inverted_index[term].remove(id)

        # remove term vector for this doc
        del self.term_vectors[id]

    def get_text(self, id):
        """ Returns the original (raw) text of a document.

            Parameters
            ----------
            id: str
                The identifier of the document to return.
        """
        # check if document exists and throw exception if so
        if not id in self.raw_text:
            raise KeyError("document with id [" + id + "] not found in index.")

        return self.raw_text[id]

    def num_docs(self):
        """ Returns the current number of documents in index.
        """
        return len(self.raw_text)

    def get_ftokens_from_docid(self, id):
        """ Returns a list of filtered tokens (with punctuation removed) for the document
            specified by the doc id.
        """
        return self.filtered_tokens[id]

    def get_uftokens_from_docid(self, id):
        """ Returns a list of unfiltered tokens (without punctuation removed) for the document
            specified by the doc id.
        """
        return self.unfiltered_tokens[id]

    # ------------------------------------------------------------------------
    #  matching
    # ------------------------------------------------------------------------

    def get_matches_term(self, term):
        """ Returns ids of documents that contain term.

            Parameters
            ----------
            term: str
                A single token, e.g., "cat" to match on.

            Returns
            -------
            set(str)
                A set of ids of documents that contain term.
        """
        # note: term needs to be lowercased so can match output of tokenizer
        # look up term in inverted index
        return self.inverted_index[term.lower()]

    def get_matches_OR(self, terms):
        """ Returns set of documents that contain at least one of the specified terms.

            Parameters
            ----------
            terms: iterable(str)
                An iterable of terms to match on, e.g., ["cat", "hat"].

            Returns
            -------
            set(str)
                A set of ids of documents that contain at least one of the term.
        """
        # initialize set of ids to empty set
        ids = set()

        # union ids with sets of ids matching any of the terms
        for term in terms:
            ids.update(self.inverted_index[term])

        return ids

    def get_matches_AND(self, terms):
        """ Returns set of documents that contain all of the specified terms.

            Parameters
            ----------
            terms: iterable(str)
                An iterable of terms to match on, e.g., ["cat", "hat"].

            Returns
            -------
            set(str)
                A set of ids of documents that contain each term.
        """
        # initialize set of ids to those that match first term
        ids = self.inverted_index[terms[0]]

        # intersect with sets of ids matching rest of terms
        for term in terms[1:]:
            ids = ids.intersection(self.inverted_index[term])

        return ids

    def get_matches_NOT(self, terms):
        """ Returns set of documents that don't contain any of the specified terms.

            Parameters
            ----------
            terms: iterable(str)
                An iterable of terms to avoid, e.g., ["cat", "hat"].

            Returns
            -------
            set(str)
                A set of ids of documents that don't contain any of the terms.
        """
        # initialize set of ids to all ids
        ids = set(self.raw_text.keys())

        # subtract ids of docs that match any of the terms
        for term in terms:
            ids = ids.difference(self.inverted_index[term])

        return ids

    # ------------------------------------------------------------------------
    #  scoring
    # ------------------------------------------------------------------------

    def idf(self, term):
        """ Returns current inverse document frequency weight for a specified term.

            Parameters
            ----------
            term: str
                A term.

            Returns
            -------
            float
                The value idf(t, D) as defined above.
        """
        return np.log10(self.num_docs() / (1.0 + self.doc_freq[term]))

    def dot_product(self, tv1, tv2):
        """ Returns dot product between two term vectors (including idf weighting).

            Parameters
            ----------
            tv1: Counter
                A Counter that contains term frequencies for terms in document 1.
            tv2: Counter
                A Counter that contains term frequencies for terms in document 2.

            Returns
            -------
            float
                The dot product of documents 1 and 2 as defined above.
        """
        # iterate over terms of one document
        # if term is also in other document, then add their product (tfidf(t,d1) * tfidf(t,d2))
        # to a running total
        result = 0.0
        for term in tv1.keys():
            if term in tv2:
                result += tv1[term] * tv2[term] * self.idf(term) ** 2
        return result

    def length_of_doc(self, tv):
        """ Returns the length of a document (including idf weighting).

            Parameters
            ----------
            tv: Counter
                A Counter that contains term frequencies for terms in the document.

            Returns
            -------
            float
                The length of the document as defined above.
        """
        result = 0.0
        for term in tv:
            result += (tv[term] * self.idf(term)) ** 2
        result = result ** 0.5
        return result

    def cosine_similarity(self, tv1, tv2):
        """ Returns the cosine similarity (including idf weighting).

            Parameters
            ----------
            tv1: Counter
                A Counter that contains term frequencies for terms in document 1.
            tv2: Counter
                A Counter that contains term frequencies for terms in document 2.

            Returns
            -------
            float
                The cosine similarity of documents 1 and 2 as defined above.
        """
        return self.dot_product(tv1, tv2) / max(1e-7, (self.length_of_doc(tv1) * self.length_of_doc(tv2)))

    def add_all(self, input_list):

        raw_texts, f_tokens, uf_tokens = zip(*input_list)

        count = 1
        for i in range(len(raw_texts)):
            self.add(str(count), raw_texts[i], f_tokens[i], uf_tokens[i])

            count += 1

    def first_sent_highest_doc(self, q, k=1):
        """ Finds the top document that contains that most occurences of the query (q)
            and returns the first sentence of that document.

            *Raises an Exception if the query is not found in any of the documents the
            search engine contains*

            Parameters
            ----------
            q: str
                A string containing the query words to match on, e.g., "cat hat".

            Returns
            -------
            the_first_sent: str
                A first sentence of the highest ranked document.

        """
        # tokenize query
        # note: it's very important to tokenize the same way the documents were so that matching will work
        query_tokens = self.filtered_tokenize(q)

        # get matches
        # just support OR for now...
        ids = self.get_matches_OR(query_tokens)

        # raises an exception if the query is not found in any of the documents
        if len(ids) == 0:
            return None
            # raise Exception("Sorry, no matches were found with the query \' " + q + " \' ")

        # convert query to a term vector (Counter over tokens)
        query_tv = Counter(query_tokens)

        # score each match by computing cosine similarity between query and document
        scores = [(id, self.cosine_similarity(query_tv, self.term_vectors[id])) for id in ids]
        scores = sorted(scores, key=lambda t: t[1], reverse=True)

        # highest ranked document's id

        the_highest_id = scores[0][0]

        # converting the document's id to a raw text string
        the_highest_text = self.get_text(the_highest_id)

        # tokenizing the sentences in the highest ranked document and returing the
        # first element of that list, which represents the first sentence of that document

        the_first_sent = sent_tokenize(the_highest_text)[0]

        return the_first_sent

def extract_single(tokens, by_type=False):
"""
Gets the named entities in the list of tokens

Parameters:
-----------
tokens: list(str)
    a tokenized document containing named entities
by_type: bool, optional
    whether or not differentiate between types of named entities
    ***True currently not supported***
    
Returns:
--------
out: list(str)
    a list of the named entities in the document
"""
    entities = []
    pos = nltk.pos_tag(tokens) # label parts of speech
    named_entities = nltk.ne_chunk(pos, binary=not by_type) # identify named entities
    for i in range(0, len(named_entities)):
        ents = named_entities.pop()
        if getattr(ents, 'label', None) != None and ents.label() == "NE": 
            entities.append(([ne for ne in ents]))
    extracted = np.array(entities)
    if extracted.ndim == 3:
        final = extracted[:,:,0].tolist()
    else:
        final = []
        for entity in extracted:
            final.append(np.array(entity)[:,0].tolist())
    out = []
    for entity in final:
        entity = " ".join(entity)
        out.append(entity)
    return out

def create_database(filepath=DATABASE_FR):
    """
    Initializes the global variable db with the database stored in the specfied filepath
    
    Parameters:
    -----------
    filepath: str (optional)
        the directory and filename of the location of the database pickled txt file
        
    Returns:
    --------
        db: dict (str --> Counter)
            the database mapping a named entity to the Counter 
            of co-occurences with other entities in the database
    """
    global db
    db = retrieve_database(filepath=filepath)
    return db

def update_single(extracted):
    """
    Updates the database with the provided named entities in a single document
    
    Parameters:
    -----------
    extracted: list(str)
        a list of the named entities in the document
    
    Returns:
    --------
    db: dict (str --> Counter)
            the database mapping a named entity to the Counter 
            of co-occurences with other entities in the database
    """
    global db
    counts = Counter(extracted)
    for entity in extracted:
        if entity in db:
            db[entity.lower()].update(counts)
            for key, value in counts.items():
                if not key == entity:
                    db[entity.lower()][key] = value * db[entity.lower()][entity]
        else:
            db[entity.lower()] = Counter(extracted)
            for key, value in counts.items():
                if not key == entity:
                    db[entity.lower()][key] = value * db[entity.lower()][entity]
        del db[entity.lower()][entity]
    return db

def new_database(filepath=DATABASE_FR):
    """
    Creates a new text file and folder in the filepath 

    Parameters:
    -----------
    filepath: str (optional)
        the directory and filename of the location of the database pickled txt file 
    """
    if not os.path.exists(filepath):
        os.makedirs(str.partition(filepath, "/")[0])

def write_database(filepath=DATABASE_FR):
    """
    Saves the database to the text file in the specified file path

    Parameters:
    -----------
    filepath: str (optional)
        the directory and filename of the location of the database pickled txt file 
    """
    global db
    with open(filepath, "wb") as f:
        pickle.dump(db, f)

def retrieve_database(filepath=DATABASE_FR):
    """
    Retrieves the database from the text file in the specified file path

    Parameters:
    -----------
    filepath: str (optional)
        the directory and filename of the location of the database pickled txt file
        
    Returns:
    --------
    db: dict (str --> Counter)
            the database mapping a named entity to the Counter 
            of co-occurences with other entities in the database
    """
    global db
    with open(filepath, "rb") as f:
        db = pickle.load(f)
    return db

def clear_database(filepath=DATABASE_FR):
    """
    Removes all entries in the database in the text file in the specified file path

    Parameters:
    -----------
    filepath: str (optional)
        the directory and filename of the location of the database pickled txt file
    """
    global db
    db = defaultdict(Counter)
    with open(filepath, "wb") as f:
        pickle.dump(db, f)

def top_related(entity, k=None):
    """
    Finds the k entities most closely related to the query (an entity itself)
    
    Parameters:
    -----------
    entity: str
        the entity which the results will be related to 
    k: int (optional)
        the number of entities to be returned (default all matches)
        
    Returns:
    --------
    out: list(tuple(str, int))
        the k entities most closely related to the query and their count
    """
    global db
    if entity not in db:
        return None
    out = db[entity].most_common(k)
    return out

def update(tokenlists):
    """
    Updates the database with the provided named entities in multiple documents
    
    Parameters:
    -----------
    tokenlists: list(list(str))
        a list of the named entities in the documents
    
    Returns:
    --------
     db: dict (str --> Counter)
            the database mapping a named entity to the Counter 
            of co-occurences with other entities in the database
    """
    global db
    for doc in tokenlists:
        update_single(extract_single(doc))
    return db

def common_entities(tokenlists, k=None):
    """
    Given a list of token-lists, return the k most common entities among them
    
    Parameters:
    -----------
    tokenlists: list(list(str))
        a list of the named entities in the documents
    k: int (optional)
        the number of entities to be returned (default all matches)
        
    Returns:
    --------
    out: list(tuple(str, int))
        the k entities most common in the documents and their counts
    """
    matches = Counter()
    for tokens in tokenlists:
        extraction = extract_single(tokens)
        matches.update(Counter(extraction))
    out = matches.most_common(k)
    return out

def collect_articles():
    """
    Collects articles from Reuters and writes to database.
    """
    update(for_ner())
    write_database()


def get_data():
    global db
    db = retrieve_database()
    return db


def initialize():
    clear_database()
    collect_articles()
    s_e = SearchEngine()
    s_e.add_all(for_search())
    with open("data/search_engine_data.txt", "wb") as f:
        pickle.dump(s_e, f)


def clear_database():
    clear_database()


def q1(topic):
    topic = topic.lower()
    with open("data/search_engine_data.txt", "rb") as f:
        s_e = pickle.load(f)
    return s_e.first_sent_highest_doc(topic)


def q2(ent, k=3):
    """
    Returns the top k entities associated with ent.

    Parameters
    ----------
    ent: String
        The entity to be passed as a query.
    k: int
        The number of top entities to be returned.

    Returns
    -------
    List of strings
    """
    if k is None:
        k = 3
    else:
        k = int(k)
    return top_related(ent.lower(), k=k)


def q3(query, k=3):
    """
    Returns the top k entities associated with a query.

    Parameters
    ----------
    query: String
        The query to be passed to obtain documents.
    k: int
        The number of top entities to be returned.

    Returns
    -------
    List of strings
    """
    if k is None:
        k = 3
    else:
        k = int(k)
    with open("data/search_engine_data.txt", "rb") as f:
        s_e = pickle.load(f)
    query_tokens = s_e.filtered_tokenize(query)

    ids = s_e.get_matches_OR(query_tokens)

    # raises an exception if the query is not found in any of the documents
    if len(ids) == 0:
        return None
        # raise Exception("Sorry, no matches were found with the query \' " + q + " \' ")
    tokens = [s_e.unfiltered_tokens[id] for id in ids]
    return common_entities(tokens, k=k)