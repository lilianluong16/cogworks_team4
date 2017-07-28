import NER
import news_loader
import search_engine
import pickle
from collections import Counter


def collect_articles():
    """
    Collects articles from Reuters and writes to database.
    """
    NER.update(news_loader.for_ner())
    NER.write_database()


def get_data():
    NER.db = NER.retrieve_database()
    return NER.db


def initialize():
    clear_database()
    collect_articles()
    s_e = search_engine.SearchEngine()
    s_e.add_all(news_loader.for_search())
    with open("data/search_engine_data.txt", "wb") as f:
        pickle.dump(s_e, f)


def clear_database():
    NER.clear_database()


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
    return NER.top_related(ent.lower(), k=k)


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
    return NER.common_entities(tokens, k=k)