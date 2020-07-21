def tokenize(docs, min_tokens=2, sep=' '):
    """ Tokenize document (list of sentences) into list of list of tokens
    """
    tokens = [x.split(sep) for x in docs]
    if min_tokens > 1:
        # Only keep sentence with at least 2 tokens
        tokens = [sentence for sentence in tokens if len(sentence) >= min_tokens]
    return tokens
