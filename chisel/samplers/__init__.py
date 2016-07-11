
def ngram(text, n, with_complete_tail=True, empty_word=u"\u0000"):
    assert isinstance(n, int), "n must to a an integer!"
    
    if len(text) > n:
        for k in range(len(text) - n):
            yield list(text[k:k + n])
    
        k = len(text) - n
    
        if with_complete_tail:
            for i in range(n):
                yield list(text[k+i:k + n]) + [empty_word for j in range(i)]
        else:
            yield list(text[k:k + n])
            
    else:
        if with_complete_tail:
            n_nones = n - len(text)
            for i in range(len(text)):
                yield list(text[i:]) + [empty_word for j in range(n_nones+i)]
        else:
            i = 0
            n_nones = n - len(text)
            yield list(text[i:]) + [empty_word for j in range(n_nones+i)]
        
        