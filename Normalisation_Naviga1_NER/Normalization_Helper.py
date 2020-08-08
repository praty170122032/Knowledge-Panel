import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import json
import math
import copy
import random
from collections import defaultdict, Counter, OrderedDict
from itertools import combinations
import unicodedata
import re
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import wikipedia as wiki
import Levenshtein as lvst
from metaphone import doublemetaphone as dmt_phone
import networkx as nx
import nxviz as nv
from sparse_dot_topn import awesome_cossim_topn as cossim_topn


prefixes_all = ['lieutenant colonel', 'Lieutenant General', 'treasury secretary', 'lieutenant general', 'Lieutenant Colonel', 'Treasury Secretary', 'Defence Secratery', 
'defence secratery', 'defense secratery', 'Defense Secratery', 'defense minister', 'defence minister', 'Defence Minister', 'Defense Minister', 'Deputy Director', 
'general manager', 'deputy director', 'party secretary', 'General Manager', 'Party Secretary', 'prime minister', 'Vice President', 'Prime Minister', 'vice president', 
'Management MG', 'management mg', 'Major General', 'major general', 'Lieutenant', 'Management', 'lieutenant', 'ambassador', 'management', 'major gen.', 'Major Gen.', 
'Ambassador', 'Secratery', 'secretary', 'Secretary', 'army gen.', 'major gen', 'Major Gen', 'President', 'president', 'Army Gen.', 'secratery', 'director', 'marshall',
 'army gen', 'Director', 'treasury', 'Army Gen', 'Treasury', 'minister', 'Marshall', 'Minister', 'colonel', 'premier', 'Colonel', 'defense', 'speaker', 'Manager', 
 'General', 'manager', 'admiral', 'Speaker', 'Admiral', 'Defence', 'defence', 'general', 'Premier', 'Defense', 'Deputy', 'Maddam', 'deputy', 'maddam', "Ma'am", 
 'Major', 'Prime', "ma'am", 'party', 'major', 'Party', 'prime', 'sir.', 'mrs.', 'gen.', 'vice', 'army', 'Army', 'Mrs.', 'Sir.', 'miss', 'Gen.', 'Vice', 'Miss', 'sir',
  'Gen', 'mrs', 'ltg', 'Ms.', 'LTG', 'dr.', 'Sir', 'ms.', 'Mr.', 'mr.', 'Mrs', 'Dr.', 'gen', 'Ms', 'MG', 'dr', 'ms', 'Mr', 'Dr', 'mg', 'mr']

def distance_simliarity(entity_list):
    for name1, name2 in combinations(entity_list, 2):
        dist_lvst = lvst.distance(name1, name2)
        dist_jaro = lvst.jaro_winkler(name1, name2)
        edit_ops = lvst.editops(name1, name2)
        match_blocks = lvst.matching_blocks(edit_ops, name1, name2)
        
        yield ((name1, name2), dist_lvst, dist_jaro, edit_ops, match_blocks)

def filter_by_distance(simliarity, lower_lvst=1, upper_lvst=2, threshold_jaro=0.8):
    '''
    param:
        simliaritry -> generator: contains simliarity info
        lower_lvst -> int: lower bound of Levenshtein distance to filter (inclusive)
        upper_lvst -> int: upper bound of Levenshteindistance to filter (exclusive)
        threshold_jaro -> float: threhold of Jaro-Winkler distance (inclusive)
    
    return:
        res -> list: tuples of entity names such that lower_bound <= distance < upper_bound  
    '''
    # FACT: known max distance for person pairs is 69 (as of data ingestion on 04/10/2020)
    
    res = list()
    for tup in simliarity:
        dist_lvst, dist_jaro = tup[1], tup[2]
        if lower_lvst <= dist_lvst  and dist_lvst < upper_lvst and dist_jaro > threshold_jaro: 
            res.append(tup)
    
    return res        


def normalize_unicode_to_ascii(data, lowercasing=True):
    normal = unicodedata.normalize('NFKD', data).encode('ASCII', 'ignore')
    val = normal.decode("utf-8")
    if lowercasing:
        val = val.lower()
    # remove special characters
    val = re.sub('[^A-Za-z0-9 ]+', ' ', val)
    # remove multiple spaces
    val = re.sub(' +', ' ', val)
    return val




''' 4 functions for TFIDF SIMILARITY '''

def generate_ngrams(data, n=3, captialized_1st=True, prefix2remove=prefixes_all):
    '''
    generate subword-level ngrams
    '''
    # dealing non-ASCII characters
    normal = unicodedata.normalize('NFKD', data).encode('ASCII', 'ignore')
    val = normal.decode("utf-8") # no lovercasing yet
    
    # remove prefix
    if prefix2remove:  
        for prefix in prefix2remove:
            val = val.replace(prefix, '')
            #val = val[1:] if val[0] == ' ' else val # remove any starting white space
    
    # lowercasing
    val = val.lower()
    # remove special characters
    val = re.sub('[^A-Za-z0-9 ]+', ' ', val)
    # remove multiple spaces
    val = re.sub(' +', ' ', val)
    
    #print(val)
    
    # Capitalized 1st letter of every word
    if captialized_1st: 
        val = val.title()
    # padding
    padding = ' '
    val = padding + val + padding
    #string = re.sub(r'[,-./]|\sBD',r'', string)
    
    ngrams = zip(*[val[i:] for i in range(n)])
    
    return [''.join(ngram) for ngram in ngrams]

def get_matches_df(similarity_matrix, A, B):
        '''
        Takes a matrix with similarity scores and two arrays, A and B,
        as an input and returns the matches with the score as a dataframe.
        Args:
            similarity_matrix (csr_matrix)  : The matrix (dimensions: len(A)*len(B)) with the similarity scores
            A              (pandas.Series)  : The array to be matched (dirty)
            B              (pandas.Series)  : The baseline array (clean)
        Returns:
            pandas.Dataframe : Array with matches between A and B plus scores
        '''
        non_zeros = similarity_matrix.nonzero()

        sparserows = non_zeros[0]
        sparsecols = non_zeros[1]

        nr_matches = sparsecols.size

        in_text = np.empty([nr_matches], dtype=object)
        matched = np.empty([nr_matches], dtype=object)
        similarity = np.zeros(nr_matches)

        in_text = np.array(A)[sparserows]
        matched = np.array(B)[sparsecols]
        similarity = np.array(similarity_matrix.data)

        df_tuples = list(zip(in_text, matched, similarity))

        return pd.DataFrame(df_tuples, columns=['in_text', 'matched', 'similarity'])

def remove_duplicated_match(df, col1='in_text', col2='matched'):
    '''
    generate a temporary column of "pair" and use it for de-duping and remove it
    '''
    temp = list()
    names1 = df[col1].tolist()
    names2 = df[col2].tolist()
    for tp in zip(names1, names2):
        temp.append(str(sorted(list(tp))))
    
    df['pair'] = temp
    df.drop_duplicates(subset="pair", inplace=True)
    df.drop(columns=['pair'], axis=1, inplace=True)
    
    return df


def get_simliar_names_by_tfdif(names, topN=2, threshold=0.7, remove_selfMatch=True, verbose=0):
    '''
    To Do: function description
    '''
    # get subword-level tf-idf features for each names
    vectorizer = TfidfVectorizer(min_df=1, analyzer=generate_ngrams)
    tf_idf_mat = vectorizer.fit_transform(names)
    if verbose:
        print('tf_idf_mat size: {}'.format(tf_idf_mat.shape))
    
    # get top N simliar names by consine simliarity
    similarity_mat = cossim_topn(tf_idf_mat, tf_idf_mat.transpose(),
                                 topN, use_threads=True, n_jobs=4)
    
    # get dataframe of matched result
    df_matched = get_matches_df(similarity_mat, pd.Series(names), pd.Series(names))
    if verbose:
        print('df_matched_raw size: {}'.format(df_matched.shape))
    
    # filter by simliarity threshold and remove self-matching
    df_matched = df_matched[(df_matched.similarity > threshold)]
    
    # remove self-matching:
    if remove_selfMatch:
        df_matched = df_matched[(df_matched.in_text != df_matched.matched)]
    
    # retain only one copy of matched pairs
    #df_matched.drop_duplicates(subset="similarity", inplace=True)
    df_matched = remove_duplicated_match(df_matched)
    
    # reverse sort by simliarty
    df_matched.sort_values(by=['similarity'], ascending=False, inplace=True, ignore_index=True)
    
    if verbose:
        print('df_matched_final size: {}'.format(df_matched.shape))
    
    return df_matched 


''' For First letter matching 2 functions '''

def first_letters(word, sorting=False, prefix2remove=prefixes_all):
    if prefix2remove:  
        for prefix in prefix2remove:
            word = word.replace(prefix, '')
        #word = word[1:] if word[0] == ' ' else word # remove any starting white space
    #print(word)
    val = normalize_unicode_to_ascii(word, lowercasing=False)      
    parts = val if val.isupper() and ' ' not in val else val.split() # don't split if val itself is an acronym
    
    res = [p[0].lower() for p in parts]
    if sorting:
        res.sort()
    
    return "".join(res)

def get_1st_letters_match(A_iter, B_iter, sorting=False, partial_match=False, prefix2remove=prefixes_all):
    '''
    param:
        A_iter, B_iter: iterable with same length
    return:
        list of boolean values
    '''
    assert len(A_iter) == len(B_iter)
    
    res = list()
    for tp in zip(A_iter, B_iter):
        first_letters_A = first_letters(tp[0]
                                        , sorting=sorting
                                        , prefix2remove=prefix2remove
                                       )
        
        first_letters_B = first_letters(tp[1]
                                        , sorting=sorting
                                        , prefix2remove=prefix2remove
                                       )

        if len(first_letters_B) < len(first_letters_A):
            first_letters_A, first_letters_B = first_letters_B, first_letters_A
        
        #print(first_letters_A)
        #print(first_letters_B)
        
        if partial_match:
            if first_letters_A in first_letters_B:
                res.append(True)
            else:
                res.append(False)
        else:
            if first_letters_A == first_letters_B:
                res.append(True)
            else:
                res.append(False)
    
    return res


''' 3 functions for double metaphone match '''

def match_doublemetaphone(word_pair, normalize2Ascii=True, prefix2remove=prefixes_all):
    '''
    ToDo: function description
    '''
    w1, w2 = word_pair
    
    if prefix2remove:  
        for prefix in prefix2remove:
            w1 = w1.replace(prefix, '')
            w2 = w2.replace(prefix, '')
    
        # remove any starting white space
        #w1 = w1[1:] if w1[0] == ' ' else w1
        #w2 = w2[1:] if w2[0] == ' ' else w2
    
    match_types = ['ASCII_norm_match', 'strong_match', 'weak_match', 'minimal_match', 'no_match']
    
    if normalize2Ascii:
        w1, w2 = tuple(map(normalize_unicode_to_ascii, (w1, w2)))
        if w1 == w2:
            return (True, match_types[0], (None, None))
    else:
        w1, w2 = word_pair
    
    tp1 = dmt_phone(w1)
    tp2 = dmt_phone(w2)
    
    match = True
    if tp1[0] == tp2[0]: # primary_key match
        match_type = match_types[1]
    elif tp1[0] == tp2[1] or tp1[1] == tp2[0]: # secondary_key == primary_key or vise versa
        match_type = match_types[2]
    elif tp1[1] == tp2[1]: # secondary_key == secondary_key
        match_type = match_types[3]
    else:
        match, match_type = False, match_types[4]
    
    return (match, match_type, (tp1, tp2))

def doublemetaphone_simliarity(entity_list):
    match_types = ['ASCII_norm_match', 'strong_match', 'weak_match', 'minimal_match', 'no_match']
    
    # mutually exclusive match groups
    ascii_norm_match = list()
    strong_match = list()
    weak_match = list()
    minimal_match = list()
    
    for name_tp in combinations(entity_list, 2):
        matched, match_type, double_metaphones = match_doublemetaphone(name_tp)
        if matched:
            if match_type == match_types[0]:
                ascii_norm_match.append(name_tp)
            elif match_type == match_types[1]:
                strong_match.append(name_tp)
            elif match_type == match_types[2]:
                weak_match.append(name_tp)
            elif match_type == match_types[3]:
                minimal_match.append(name_tp)
    
    return ascii_norm_match, strong_match, weak_match, minimal_match    



def sigmoid(x, a=1):
    return 1/(1 + np.exp(-a*x))

def dmtph_match_score(x):
    '''
    This function gives rewards for "ascii_norm_match" & "strong_match", but penalties for the rest
    i.e. non-linear weighting
    '''
    return sigmoid(20*(x-0.7))

def get_dmtph_match(A_iter, B_iter, prefix2remove=prefixes_all):
    '''
    param:
        A_iter, B_iter: iterable with same length
    return:
        list of match scores
    '''
    assert len(A_iter) == len(B_iter)
    
    match_types = ['ASCII_norm_match', 'strong_match', 'weak_match', 'minimal_match', 'no_match']
    match_types.reverse()
    x = np.linspace(0, 1, 5)

    res = list()
    for tp in zip(A_iter, B_iter):
        match, match_type, dummy = match_doublemetaphone(tp, prefix2remove=prefix2remove)
        match_score = dmtph_match_score(x[match_types.index(match_type)])
        res.append(match_score)
    
    return res


''' Get string distance 1 function '''

def get_string_distance(A_iter, B_iter, normalize=True, prefix2remove=prefixes_all):
    '''
    param:
        A_iter, B_iter: iterable with same length
    return:
        lists of invese-Levenshtein distance & Jaro-Winkler distance, respectively 
    '''
    assert len(A_iter) == len(B_iter)
    
    res_inv_lvst = list()
    res_jw = list()
    for w1, w2 in zip(A_iter, B_iter):
        if prefix2remove:  
            for prefix in prefix2remove:
                w1 = w1.replace(prefix, '')
                w2 = w2.replace(prefix, '')
        
            # remove any starting white space
            #w1 = w1[1:] if w1[0] == ' ' else w1
            #w2 = w2[1:] if w2[0] == ' ' else w2
        
        if normalize:
            w1, w2 = tuple(map(normalize_unicode_to_ascii, (w1, w2)))
        else:
            pass
        
        inv_lvst_dist = 1/lvst.distance(w1, w2) if lvst.distance(w1, w2) else 1 # avoid dividing by zero
        jw_dist = lvst.jaro_winkler(w1, w2)
        res_inv_lvst.append(inv_lvst_dist)
        res_jw.append(jw_dist)
    
    return res_inv_lvst, res_jw


def get_ensemble_match(df, weights=[]):
    '''
    get a weighted average of matching scores
    '''
    row_num = df.shape[0]
    matchings = df.columns[2:]
    if weights:
        assert len(weights) == len(matchings)
        weights = np.array(weights)
    else:
        weights = np.ones(len(matchings))
    
    match_score = pd.Series(np.zeros(row_num), dtype='float64')
    for weight, matching in zip(weights, matchings):
        match_score += weight*df[matching]
    
    match_score = match_score/weights.sum()
    
    return match_score

def add_match_score(df, weights=[]):
    if 'match_score' in df.columns:
        df.drop(['match_score'], axis=1, inplace=True)
    df['match_score'] = get_ensemble_match(df, weights=weights)
    df.sort_values(by='match_score', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

def cut_off(df, q=None, lower_bound=0.0):
    assert lower_bound >= 0.0 and lower_bound <= 1.0
    if not lower_bound:
        if not q:
            return df[df.match_score > df.match_score.mean()]
        else:
            assert type(q) == float
            return df[df.match_score > df.match_score.quantile(q)]
    else: # given nonzero lowever bound, use it as a filter
        eps = 1e-10
        return df[df.match_score >= lower_bound-eps] # inclusive

''' The ENSEMBLE FUNCTION of the above functions '''

def get_df_match(entity_names, entity_type, entity_subtypes=[]
                 , prefix2remove=prefixes_all
                 , sorting_1st_letters=False
                 , partial_1st_letter_match=False
                 , co_doc_by_file=True       
                ):
    
    df = get_simliar_names_by_tfdif(entity_names)
    
    print("# of candidate pairs: {}".format(df.shape[0]))
    
    
    df["first_letter_match"] = get_1st_letters_match(df.in_text, df.matched
                                                     , sorting=sorting_1st_letters
                                                     , partial_match=partial_1st_letter_match
                                                     , prefix2remove=prefix2remove
                                                    )
    
    
    df["dmtph_match"] = get_dmtph_match(df.in_text, df.matched, prefix2remove=prefix2remove)
    
    inv_lvst_dist, jw_dist = get_string_distance(df.in_text, df.matched, prefix2remove=prefix2remove)
    df["inv_lvst_dist"] = inv_lvst_dist
    df["jw_sim"] = jw_dist # in python Levenshtein library, this is actually simlarity ranging [0, 1]
    
    return df


''' 4 graph functions to normalize '''

def get_connected_comp(df):
    edge_list = list(df[["in_text", "matched", "match_score"]].itertuples(index=False, name=None))
    graph_ = nx.Graph()
    graph_.add_weighted_edges_from(edge_list)

    # extract connected components and reverse sort by length of components
    connected_comp = [c for c in nx.connected_components(graph_)]
    connected_comp.sort(key=len, reverse=True)
    
    return connected_comp

def get_postProcessigUnits(connected_comp):
    connected_comp_short_ = sorted([list(comp) for comp in connected_comp if len(comp) == 2], key=lambda x:x[0])
    connected_comp_long_ = sorted([list(comp) for comp in connected_comp if len(comp) > 2], key=lambda x:x[0])
    connected_comp_all_ = sorted([list(comp) for comp in connected_comp], key=lambda x:x[0])
    
    connected_comp_short = [(i, comp) for i, comp in enumerate(connected_comp_short_)]
    connected_comp_long = [(i, comp) for i, comp in enumerate(connected_comp_long_)]
    connected_comp_all = [(i, comp) for i, comp in enumerate(connected_comp_all_)]
    
    return connected_comp_short, connected_comp_long, connected_comp_all 

def chunking(connected_comp_any, index2connect=[] ,index2exclude=[]):
    chunks = list()
    if index2connect:
        for tp in index2connect: 
            chunk = list()
            for idx in tp:
                chunk.extend(list(connected_comp_any[idx][1]))
                index2exclude.append(idx)

            chunks.append(chunk)
        
    return chunks, index2exclude

def get_comp_final(connected_comp, connected_comp_short, is_all=True
                   , *, index2connect=[], index2exclude=[], comp2add=[]):
    chunks, index2exclude = chunking(connected_comp, index2connect=index2connect, index2exclude=index2exclude)
    
    res = [list(a[1]) for i, a in enumerate(connected_comp) if i not in index2exclude]
    
    if chunks:
        res.extend(chunks)
    if comp2add:
        res.extend(comp2add)
    
    if not is_all:
        res += [list(a[1]) for a in connected_comp_short]
    
    return res

''' choose normalized function '''

def choose_normalized(connected_comp, choice='lvst'):
    '''
    choose one normalized term for each connected component by rule-based system
    NOTE: "longest = False" will choose the shortest
    '''
    res = list()
    for comp in connected_comp:
          
        # Rule-1: no period at the end
        temp = [name for name in comp if name[-1] != '.']
        comp = temp if temp else comp
        
        # Rule-2.1: no hyphen anywhere
        temp = [name for name in comp if '-' not in name]
        comp = temp if temp else comp
        
        # Rule-2.2: no '/' anywhere
        temp = [name for name in comp if '/' not in name]
        comp = temp if temp else comp
        
        # Rule-3: start with uppercase
        temp = [name for name in comp if name[0].isupper()]
        comp = temp if temp else comp
        
        # Rule-4: no unicode (no '\u2003')
        temp = []
        for name in comp:
            normal = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore')
            val = normal.decode("utf-8")
            temp.append(val)
            
        comp = list(set(temp)) if temp else comp     
        
        # Rule-5: choose by length or Levenshtein Median
        comp.sort(key=len)
        if choice == 'long':
            res.append(comp[-1])
        elif choice == 'short':
            res.append(comp[0])
        elif choice == 'lvst':
            res.append(lvst.median(comp))
        else:
            print("Not a valid choice methond")
            return None
    
    return res

''' final normalized dictionary to get the real mappings '''

def make_norm_dict(components, norms):
    raw2normalized = dict()
    normalized2raw = defaultdict(list)
    for comp, norm in zip(components, norms):
        for name in comp:
            raw2normalized[name] = norm
            normalized2raw[norm].append(name)
    
    return raw2normalized, normalized2raw

