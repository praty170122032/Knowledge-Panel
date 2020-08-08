import json
import os
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
COLLECTION_NAME = config.DISCOVERY_COLLECTION_NAME
discovery_data = discovery_init(COLLECTION_NAME=COLLECTION_NAME)
processDirectory('corpus',discovery_data)