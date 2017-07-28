import NER
import news_loader


def collect_articles():
    NER.update(news_loader.for_ner())
    NER.write_database()


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
    return NER.top_related(ent, k=k)

