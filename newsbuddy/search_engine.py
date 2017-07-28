from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import numpy as np
import string


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