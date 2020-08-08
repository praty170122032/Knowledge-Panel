
import requests
import json
import os
from time import sleep
from requests.auth import HTTPBasicAuth
from ibm_watson import DiscoveryV1
from pprint import pprint
import modules.discovery_helper as dh_help 

COLLECTION_NAME = 'Pratyush_Test'
# # COLLECTION_NAME = 'Earnings Calls Full Documents-enti-rel-se_role-key'
# # DIRECTORY_NAME =  "processeddocumentjson_r1"
DIRECTORY_NAME = "corpus"

# # COLLECTION_NAME = 'Earning Calls - Text'
# pprint("upload from "+DIRECTORY_NAME+" to "+COLLECTION_NAME)
# # data = dh_help.discovery_init(COLLECTION_NAME=COLLECTION_NAME)
# # pprint(data)
# # dh_help.processDirectory(DIRECTORY_NAME,data)

# discovery_data_r3 = dh_help.discovery_init(COLLECTION_NAME=COLLECTION_NAME,default="r3")
# dh_help.processDirectory(DIRECTORY_NAME,discovery_data_r3)





# COLLECTION_NAME = 'Earnings Calls Full Documents-enti-rel-se_role-key'
# DIRECTORY_NAME = "corpusd"
# DIRECTORY_NAME = "processedjsonupload"
# DIRECTORY_NAME = "processeddocumentjson"
# DIRECTORY_NAME = "processedjson"
# COLLECTION_NAME = 'Earning Calls - Text'
pprint("upload from "+DIRECTORY_NAME+" to "+COLLECTION_NAME)
discovery_data = dh_help.discovery_init(COLLECTION_NAME=COLLECTION_NAME)
dh_help.processDirectory(DIRECTORY_NAME,discovery_data)

