from pathlib import Path
import sys
home = str(Path.home())
DISCOVERY_COLLECTION_NAME = 'Test'
EARNING_CALL_DB = "factset_russell_earnings_call_new_16July"
EARNING_CALL_SERVER = "211_russel"
EARNING_XML_DB = "FACTSETEARNINGSCALL"
EARNING_XML_SERVER = "103"

if sys.platform == "linux" or sys.platform == "darwin":
    slash = "/"
else:
    slash = "\\"

# DISCOVERY_COLLECTION_NAME = 'Earning Calls - Snippet - v3'
# EARNING_CALL_DB = "factset_russell_earnings_call_intermediate_v3"

BASE_PATH = home+slash