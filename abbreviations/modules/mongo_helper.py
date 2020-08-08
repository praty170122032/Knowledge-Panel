import pymongo
import modules.common_helper as cm_help
import urllib
import re
import modules.config as config

DEFAULT_MC_KEY = 'master_concepts'
def get_mongo_client(params):
    user = params['user']
    passwd = urllib.parse.quote(params['password'])
    host = params['host']
    port = params['port']
    db_name = params['db']
    con_string = 'mongodb://%s:%s@%s:%d/%s' % (user, passwd, host, int(port), db_name)
    client = pymongo.MongoClient(con_string)
    return client

all_params = cm_help.get_params()
params = all_params[config.EARNING_CALL_SERVER]
client = get_mongo_client(params)
db = client[params['db']]
russel_coll = db[config.EARNING_CALL_DB]

def getRusselTickers():
    codes = ['AMZN-US','APTV-US','BEN-US','CE-US','CME-US','FLR-US','LB-US','NFLX-US','PSA-US','TRIP-US','TWTR-US','WYNN-US']
    codes = []
    factsetalredyadded = db["factset_xml_processed"].distinct("ticker")
    response= russel_coll.find({
                                  "$and":[{"code":{"$nin":factsetalredyadded}},{"code":{"$nin":codes}}]
                                }).distinct("code")
    return response

# returns the aggregate score for the entire call and also a per-concept list containing the concept_score and
# concept_importance
def helper_get_aggr_score_from_mc(master_concept, company, sector, concept_list=None,marketcapdata=None):
    # TODO: this assumes the weight of a concept is proportional to the number of snippets it has in this call
    # We need to check for the concept weightage for this company/sector/overall in DB and accordingly calculate
    total_score = 0.0
    total_count = 0
    concept_details = []    
    for k in master_concept.keys():
        if concept_list is not None and k not in concept_list:
            continue
        mck = master_concept[k]
        concept_score = 0.0
        concept_count = 0
        sector_cnt = 0.0
        industries_cnt = 0.0
        for x in mck:
            if x[1] != 0.0:
                concept_score += x[1]
                concept_count += 1
                # sector_cnt += double()
                # industries_cnt += double()
        total_score += concept_score
        total_count += concept_count
        if concept_count > 0:
            # concept_score = concept_score / concept_count
            concept_score = concept_score / concept_count * 5.0
            concept_details.append([k, concept_score, concept_count])
    if total_count > 0:
        # total_score = total_score / total_count
        total_score = total_score / total_count * 5.0
        for cd in concept_details:
            cd[2] = cd[2] / total_count
    return total_score, concept_details



def get_concept_wise_history( company_code, num_calls,concept_list=[], master_concepts_key=DEFAULT_MC_KEY):
    result = helper_recent_master_concepts(company_code, num_calls,[],master_concepts_key)
    data = {}
    response = {}
    all_concepts = set()
    concepts_per_date = {}
    for r in result:
        mc = r[master_concepts_key]
        aggr_score, concept_details = helper_get_aggr_score_from_mc(mc, r['code'],None)
        concepts_per_date[r['date']] = []
        if r['code'] not in data.keys():
            data[r['code']] = []
        for cd in concept_details:
            concept = cd[0]
            if len(concept_list)>0 and concept not in concept_list:
                continue
            score = cd[1]
            importance = cd[2]
            all_concepts.add(cd[0])
            concepts_per_date[r['date']].append(cd[0])
            mc_qna = r['master_concepts_qna']
            concept_data_qna = get_concept_data(mc_qna, False,True)
            datum = {'value': score, 'date': r['date'], 'concept_importance_qna':0 , 'concept': concept, 'importance': importance, 'concept_data_qna': concept_data_qna}
            if concept in r['concept_counts_qna'].keys():
                datum['concept_importance_qna'] = r['concept_counts_qna'][concept]
            if 'q_period' in r:
                datum['q_period'] = r['q_period']
            else:
                datum['q_period'] = q_period_from_date(r['date'])
            data[r['code']].append(datum)
    # Now we have all the concepts; just iterate again through everything and wherever the concept is missing, just
    # fill in with default 0 value.
    result.rewind()
    for rr in result:
        mc = rr[master_concepts_key]
        if rr['code'] not in data.keys():
            data[rr['code']] = []
        # present_concepts = set(mc.keys())
        present_concepts = set(concepts_per_date[rr['date']])
        for c in all_concepts:
            if c not in present_concepts: # or len(mc[c]) == 0:
                datum = {'value': 0.0, 'date': rr['date']}
                if 'q_period' in rr:
                    datum['q_period'] = rr['q_period']
                else:
                    datum['q_period'] = q_period_from_date(rr['date'])
                datum['concept'] = c
                datum['importance'] = 0.0
                data[rr['code']].append(datum)
    return data



def get_prev_earning_call_details(code,date):
    response= {}
    response['date'] = 0
    response['q_period'] = 0
    response['aggregate_sentiment'] = 0
    params = {'code':code,'get_one':True}
    params['sort'] = {'date':-1}
    params['extraconditions'] = {}
    params['extraconditions']['date'] = {'$lt':date}
    # params['q_period'] = q_period
    result = get_all_details(params)
    return result

def get_all_details(params={},collection_name=russel_coll):
    rx = re.compile('transcript$')
    reqparams = {}
    if "extraconditions" in params.keys():
        reqparams = params['extraconditions']
    reqparams['$or'] = [{'articlelink': rx}, {'is_transcript': True}]
    if "code" in params.keys():
        if not isinstance(params['code'],list):
            params['code'] = [params['code']]
        reqparams['code']= {'$in': params['code']}
    if "q_period" in params.keys():
        if not isinstance(params['q_period'],list):
            params['q_period'] = [params['q_period']]
        if len(params['q_period']) >0:
            reqparams['q_period']= {'$in': params['q_period']}
    if "calendar_q_period" in params.keys():
        if not isinstance(params['calendar_q_period'],list):
            params['calendar_q_period'] = [params['calendar_q_period']]
        reqparams['calendar_q_period']= {'$in': params['calendar_q_period']}

    if "master_concepts_exist" in params.keys():
        reqparams[DEFAULT_MC_KEY] = {'$exists': True}

    if "date" in params.keys():
        if not isinstance(params['date'],list):
            params['date'] = [params['date']]
        reqparams['date']= {'$in': params['date']}
    # if is_intermediate is True:
    #     field_returns['published'] = True
    sortDict = {"1":pymongo.ASCENDING,"-1":pymongo.DESCENDING}
    sortParams = []
    if "sort" in params.keys():
        for sortkey in params['sort']:
            sortParams.append([sortkey,sortDict[str(params['sort'][sortkey])]])
    if "get_one" in params.keys():
        response = collection_name.find_one(reqparams,sort=sortParams)
    else:
        response = collection_name.find(reqparams)
        if "distinct" in params.keys():
            response = response.distinct(params["distinct"])
        if "limit" in params.keys():
            response.limit(params['limit'])
    if "get_one" not in params.keys() and len(sortParams) > 0:
        response.sort(sortParams)
    if "skip" in params.keys():
        response.skip(params['skip'])
    return response
