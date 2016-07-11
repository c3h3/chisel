import pandas as pd
import hashlib
import json
from chisel.samplers import ngram
import numpy as np
import functools


def text2wordcountdf(text, n, var_name_pattern="w%s", with_complete_tail=True, with_n_docs=False):
    win_size = n
    col_names = map(lambda xx:"w%s" % xx, range(1,win_size+1))
    ngramdf = pd.DataFrame(ngram(text, win_size, with_complete_tail=with_complete_tail), columns=col_names)
    word_count_df = ngramdf.groupby(col_names).size().reset_index()
    word_count_df_colnames = []
    word_count_df_colnames.extend(col_names + ["n"] )
    word_count_df.columns = word_count_df_colnames
    word_count_df = word_count_df.sort("n",ascending=False)
    if with_n_docs:
        word_count_df["n_docs"] = 1
    return word_count_df


def get_wc_df_vars(wc_df):
    return filter(lambda col: col.startswith("w"), wc_df.columns)


def wc_df_add(*wc_dfs):    
    args_cols_hash = set(map(lambda df:hashlib.sha256(json.dumps(df.columns.tolist(),sort_keys=True)).hexdigest(),wc_dfs))
    assert len(list(args_cols_hash)) == 1, "dfs has different columns!"
    
    added_wc_df = pd.concat(wc_dfs)
    cols = get_wc_df_vars(added_wc_df)
    added_wc_df= added_wc_df.groupby(cols).sum().reset_index()
    added_wc_df = added_wc_df.sort("n",ascending=False)
    return added_wc_df


def pagination_generaotr(items, items_per_page = 10000):
    i = 0
    total_items = len(items)
    while i*items_per_page < total_items:
        yield items[i*items_per_page:(i+1)*items_per_page]
        i += 1


def compute_join_prob_df(wc_df):
    Xs = get_wc_df_vars(wc_df)
    prob_df = wc_df[Xs].copy()
    prob_df["p"] = wc_df.n / wc_df.n.sum()
    return prob_df


def compute_pmi_df(wc_df, X1 = None, X2 = None, k=1):
    Xs = get_wc_df_vars(wc_df)
    if X1 == None or X2 == None:
        X1 = Xs[:-1]
        X2 = [Xs[-1]]
    jp_df = compute_join_prob_df(wc_df).copy()
    p1_df = jp_df.groupby(X1).sum()
    p1_df.columns = ["p1"]
    p1_df = p1_df.reset_index()
    p2_df = jp_df.groupby(X2).sum()
    p2_df.columns = ["p2"]
    p2_df = p2_df.reset_index()
    pmi_df = pd.merge(pd.merge(jp_df,p1_df,on=X1,how="inner"),p2_df,on=X2,how="inner")
    pmi_df["pmi"] = pmi_df.p**k / (pmi_df.p1*pmi_df.p2)
    pmi_df = pmi_df.sort("pmi",ascending=False)
    return pmi_df




def check_triple_cuts(wc_df, X1=["w1"], X2=["w2"], X3=["w3"], k=1, 
                      cut_function=lambda pmi_1, pmi_2: pmi_1 >= 3*pmi_2,
                      combined_threshold=0.6):
    Xs = get_wc_df_vars(wc_df)
    
    X1_X2_pmi_df = compute_pmi_df(wc_df, X1=X1, X2=X2, k=k)[Xs + ["pmi"]]
    X1_X2_pmi_df_cols = X1_X2_pmi_df.columns.tolist()
    X1_X2_pmi_df_cols[-1] = "pmi_X1_X2"
    X1_X2_pmi_df.columns = X1_X2_pmi_df_cols
    
    X2_X3_pmi_df = compute_pmi_df(wc_df, X1=X2, X2=X3, k=k)[Xs + ["pmi"]]
    X2_X3_pmi_df_cols = X2_X3_pmi_df.columns.tolist()
    X2_X3_pmi_df_cols[-1] = "pmi_X2_X3"
    X2_X3_pmi_df.columns = X2_X3_pmi_df_cols
    
    pmi_df = pd.merge(X1_X2_pmi_df, X2_X3_pmi_df, on=Xs, how="inner")
    
    cut_between_X1_X2_condition = cut_function(pmi_df.pmi_X2_X3, pmi_df.pmi_X1_X2)
    cut_between_X2_X3_condition = cut_function(pmi_df.pmi_X1_X2, pmi_df.pmi_X2_X3)
    
    #return X1,X2,X3,pmi_df, cut_between_X1_X2_condition, cut_between_X2_X3_condition
    
    remainder_df = pmi_df[np.logical_and(np.logical_not(cut_between_X1_X2_condition),np.logical_not(cut_between_X2_X3_condition))]
    combined_threshold_value = ((remainder_df.pmi_X1_X2 + remainder_df.pmi_X2_X3) / 2).quantile(combined_threshold)
    remainder_df[(remainder_df.pmi_X1_X2 + remainder_df.pmi_X2_X3) / 2 > combined_threshold_value]
    
    res = {}
    res["X2X3"] = pmi_df[cut_between_X1_X2_condition][X2+X3].groupby(X2+X3).size().reset_index()[X2+X3].apply(lambda xx:u"".join(xx), axis=1).tolist()
    res["X1X2"] = pmi_df[cut_between_X2_X3_condition][X1+X2].groupby(X1+X2).size().reset_index()[X1+X2].apply(lambda xx:u"".join(xx), axis=1).tolist()
    res["X1X2X3"] = remainder_df[(remainder_df.pmi_X1_X2 + remainder_df.pmi_X2_X3) / 2 > combined_threshold_value][X1+X2+X3].groupby(X1+X2+X3).size().reset_index()[X1+X2+X3].apply(lambda xx:u"".join(xx), axis=1).tolist()

    return res
    
    

    
class NgramModelProbability(object):
    def __init__(self, n=3, eps=1e-9):
        self._n = n
        self._eps = eps
        self._ngram_wc_dfs = {}
        self._ngram_jp_dfs = {}
    
    
    def learning_from_texts(self, texts, with_complete_tail=True):
        self._texts = texts
        
        self._ngram_wc_dfs[self._n] = wc_df_add(*map(functools.partial(text2wordcountdf,n=self._n, with_complete_tail=with_complete_tail), self._texts))
        self._ngram_wc_dfs[self._n] = self._ngram_wc_dfs[self._n].set_index(get_wc_df_vars(self._ngram_wc_dfs[self._n]))["n"]
        #self._ngram_jp_dfs[m] = self._ngram_wc_dfs[m].set_index(get_wc_df_vars(self._ngram_wc_dfs[m]))["n"]
        self._ngram_jp_dfs[self._n] = self._ngram_wc_dfs[self._n] / self._ngram_wc_dfs[self._n].sum()
        
        Xs = get_wc_df_vars(self._ngram_wc_dfs[self._n].reset_index())
        
        for m in range(1, self._n):
            self._ngram_wc_dfs[self._n - m] = self._ngram_wc_dfs[self._n].groupby(level=Xs[:(self._n - m)]).sum()
            self._ngram_jp_dfs[self._n - m] = self._ngram_jp_dfs[self._n].groupby(level=Xs[:(self._n - m)]).sum()
            
        return self
    
    
    def __call__(self, *args):
        m = len(args)
        #assert m <= self._n, "len(args) > self._n"
        #assert m in self._ngram_jp_dfs, "need to learning prob from data!"
        
        if m <= self._n:
            try:
                if len(args) == 1:
                    return self._ngram_jp_dfs[m].__getitem__(args[0])
                else:
                    return self._ngram_jp_dfs[m].__getitem__(args)
            except KeyError:
                return self._eps
        
        else:
            return (self(*args[-self._n:])/self(*args[-self._n:-1]))*self(*args[:-1])
        
        