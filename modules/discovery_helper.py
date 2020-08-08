import json
import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../modules/")
import modules.common_helper as cm_help
from ibm_watson import DiscoveryV1
import modules.config as config

import os
import time
from pprint import pprint
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

all_params = cm_help.get_params()
def discovery_init(COLLECTION_NAME=config.DISCOVERY_COLLECTION_NAME,default=all_params['discovery']['default']):
    response = {}
    
    discovery_param = all_params['discovery']
    # default = 
    authenticator = IAMAuthenticator(apikey=discovery_param[default]['apikey'])
    endurl = "https://gateway.watsonplatform.net/discovery/api"
    discovery = DiscoveryV1(
        version=discovery_param[default]['version'],
        authenticator = authenticator
        # iam_apikey=discovery_param[default]['apikey'],
        # url=discovery_param[default]['url']    
    )
    if default !="a1":
        discovery.set_service_url(discovery_param[default]['url'])

    environments = discovery.list_environments().get_result()
    # print(json.dumps(environments, indent=2))
    print("Discovery instance loaded "+default)
    js = json.dumps(environments)
    jres = json.loads(js)
    # print(jres['environments'][1]['environment_id'])
    env_id = jres['environments'][1]['environment_id']
    collections = discovery.list_collections(env_id).get_result()
    cols = json.dumps(collections, indent=2)
    colres = json.loads(cols)

    #print(colres['collections'])
    for item in colres['collections']:
        if item['name']== COLLECTION_NAME:
            print('COLLECTION ID:', item['collection_id'], 'COLLECTION NAME:', item['name'])
            col_id = item['collection_id']
    response['env_id'] = env_id
    response['col_id'] = col_id
    response['discovery'] = discovery
    return response

def processDirectory(DIRECTORY_NAME,otherparams):
    for i in range(2):
        pprint("attempt no "+str(i))

        for root, dirs, filenames in os.walk(DIRECTORY_NAME):
            print('\nProcessing dir ' + root + ' with ' + str(len(filenames)) + ' files\n')
            if len(filenames) >0:
                # pprint(str(len(filenames)))
                request_params = otherparams
                request_params['filenames'] = filenames
                request_params['root'] = root
                uploadFiles(request_params)
            time.sleep(50)


def uploadFiles(params):
    failed_docs = {}
    id_name_map = {}

    for f in params['filenames']:
        if f.endswith('.txt') or f.endswith('.xlsx') or f.endswith('.partial') or f.endswith('.xls'):
           continue
        f_path = params['root'] + os.sep + f
        file_processed_or_failed = False
        f_mode = 'r'
        if f_path.endswith('.doc') or f_path.endswith('.docx'):
            file_type = 'application/msword'
        elif f_path.endswith('.pdf'):
            file_type = 'application/pdf'
            f_mode = 'rb'
        else:
            file_type = 'application/json'
        while not file_processed_or_failed:
            re = None
            try:
                with open(f_path, f_mode) as file_data:
                    re = params['discovery'].add_document(params['env_id'], params['col_id'], file=file_data, file_content_type=file_type).get_result()
                    if f_path.endswith('.json'):
                        print('RESPONSE:', json.dumps(re,indent=2))
                    if re is not None:
                        file_processed_or_failed = True
                        doc_id = re['document_id']
                        id_name_map[doc_id] = [f_path]
                    else:
                        file_processed_or_failed = True
                        print(f + " did not upload")
                        failed_docs[f_path] = "got a 'None' type response from WDS api"
            except Exception as e:
                if "The service is busy processing" in str(e):
                    sleep(5)
                    print('.'),
                    continue

                file_processed_or_failed = True
                print(f + " did not upload, exception" + str(e))
                failed_docs[f_path] = "'add_document' WDS api request threw exception: " + str(e)
        if f_path not in failed_docs.keys():
            os.remove(f_path)
    if len(failed_docs)>0:
        cm_help.saveMDLogs(failed_docs,"discoveryuploadfailed.json")
    return True

def getDocumentDetails(params,reqparams):
    doc_info  = params['discovery'].get_document_status(params['env_id'], params['col_id'],reqparams['document_id']).get_result()
    return doc_info

def deleteDocument(params,reqparams):
    doc_info  = params['discovery'].delete_document(params['env_id'], params['col_id'],reqparams['document_id']).get_result()
    return doc_info

def getDiscoveryData(params,reqparams):
    return_fields = None
    aggr = None
    filters = None
    count = 1000
    offset = None
    q = ''
    if 'return_fields' in reqparams.keys():
        return_fields = reqparams['return_fields'] 
    if 'aggr' in reqparams.keys():
        aggr = reqparams['aggr']
    if 'q' in reqparams.keys():
        q = reqparams['q']
    if 'filters' in reqparams.keys():
        filters = reqparams['filters']
    if 'offset' in reqparams.keys():
        offset = reqparams['offset']
    if 'count' in reqparams.keys():
        count = reqparams['count']
    print("offset in query = "+str(offset))
    res = params['discovery'].query(params['env_id'], params['col_id'], filter=filters, query=q, natural_language_query=None,
                          passages=True, aggregation=aggr, count=count, return_fields=return_fields, offset=offset,
                          sort=None, highlight=True, passages_fields=None, passages_count=50,
                          passages_characters=None, deduplicate=False, deduplicate_field=None, collection_ids=None,
                          similar=None, similar_document_ids=None, similar_fields=None, bias=None, logging_opt_out=None).get_result()
    return res


# Initialize IBM Discovery instance
def initialize_discovery(collection_name=config.DISCOVERY_COLLECTION_NAME):

    # In the constructor, letting the SDK manage the token
    authenticator = IAMAuthenticator(apikey='LTk2AGh0Mjsu3G9bGGSkU92xeCBAYdfa4lN4fjaA-CR4')
    discovery = DiscoveryV1(version='2018-08-01',
                            authenticator=authenticator)
    # discovery.set_service_url('<url_as_per_region>')
    # discovery = DiscoveryV1(
    #     version='2018-08-01',
    #     url='https://gateway.watsonplatform.net/discovery/api',
    #     iam_apikey='LTk2AGh0Mjsu3G9bGGSkU92xeCBAYdfa4lN4fjaA-CR4'
    # )

    # get the environment id
    environments = json.dumps(discovery.list_environments().get_result())
    environments = json.loads(environments)
    environment_id = environments['environments'][1]['environment_id']

    # get the collections
    collections = json.dumps(discovery.list_collections(environment_id).get_result(), indent=2)
    collections = json.loads(collections)

    # iterate through collections in Discovery
    print("======================= AVAILABLE COLLECTIONS IN CORPUS =======================")
    for collection in collections['collections']:
        print('COLLECTION ID:', collection['collection_id'], 'COLLECTION NAME:', collection['name'])

        # find the collection the user wants, and extract the collection_id
        if str(collection['name']) == collection_name:
            collection_id = collection['collection_id']
        else:
            # raise Exception('ERROR: COLLECTION NAME %s DOES NOT EXIST.' % collection_name)
            pass

    print("--- USING COLLECTION ID: %s " % collection_id)

    return discovery, environment_id, collection_id

def updateDocument(params,reqparams):
    f_path = reqparams['folder'] + os.sep + reqparams['filename']
    f_mode = 'r'
    pprint(f_path)
    with open(f_path,f_mode) as fileinfo:
        add_doc = params['discovery'].update_document(
            params['env_id'], params['col_id'],reqparams['document_id'], 
            file=fileinfo).get_result()
    print(json.dumps(add_doc, indent=2))

def discovery_init(COLLECTION_NAME=config.DISCOVERY_COLLECTION_NAME,default=all_params['discovery']['default']):
    response = {}
    
    discovery_param = all_params['discovery']
    # default = 
    authenticator = IAMAuthenticator(apikey=discovery_param[default]['apikey'])
    endurl = "https://gateway.watsonplatform.net/discovery/api"
    discovery = DiscoveryV1(
        version=discovery_param[default]['version'],
        authenticator = authenticator
        # iam_apikey=discovery_param[default]['apikey'],
        # url=discovery_param[default]['url']    
    )
    if default =="a1":
        discovery.set_service_url(discovery_param[default]['url'])

    environments = discovery.list_environments().get_result()
    # print(json.dumps(environments, indent=2))
    print("Discovery instance loaded "+default)
    js = json.dumps(environments)
    jres = json.loads(js)
    # print(jres['environments'][1]['environment_id'])
    env_id = jres['environments'][1]['environment_id']
    collections = discovery.list_collections(env_id).get_result()
    cols = json.dumps(collections, indent=2)
    colres = json.loads(cols)

    #print(colres['collections'])
    for item in colres['collections']:
        if item['name']== COLLECTION_NAME:
            print('COLLECTION ID:', item['collection_id'], 'COLLECTION NAME:', item['name'])
            col_id = item['collection_id']
    response['env_id'] = env_id
    response['col_id'] = col_id
    response['discovery'] = discovery
    return response