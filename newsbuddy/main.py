import NER
import news_loader
import search_engine


def collect_articles():
    """
    Collects articles from Reuters and writes to database.
    """
    NER.update(news_loader.for_ner())
    NER.write_database()


def clear_database():
    NER.clear_database()


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
    return NER.top_related(ent.lower(), k=k)


def q1(topic):
    topic = topic.lower()
    s_e = search_engine.SearchEngine()
    s_e.add_all(news_loader.for_search())
    return s_e.first_sent_highest_doc(topic)