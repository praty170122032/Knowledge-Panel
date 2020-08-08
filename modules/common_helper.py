import json
import os
import datetime
import sys
import modules.config as config
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../modules/")

def get_params(f='/home/pratyush/accrete/modules/params.json'):
    jf = open(f, 'r')
    jstr = jf.read()
    jf.close()
    params = json.loads(jstr)
    return params

#Create a function called "chunks" with two arguments, l and n:
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def saveInFile(tempData,path):
	with open(path, 'w') as fileOut:
		json.dump(tempData, fileOut, indent=4)

def key_format(strtext):
    strtext= strtext.replace(".","_")
    strtext = ''.join(e for e in strtext if e.isalnum())
    return strtext.replace(" ","_")

def saveMDLogs(tempData,path,OtherParams=''):
    filename = "logs/"+path
    if not os.path.exists(filename):
        open(filename, 'w+').close()
    with open(filename, 'a') as fileOut:
        json.dump(tempData, fileOut, indent=3)
        json.dump(OtherParams, fileOut, indent=3)
        json.dump(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), fileOut, indent=3)
        fileOut.write('\n')

def processMasterConcepts(master_conceptsParam,score,importance):
    responseRow = {}
    all_params = get_params()
    master_concepts = master_conceptsParam.copy()
    if master_concepts is None :
        return responseRow          
    for conceptname in master_concepts:
        if conceptname=="Unclassified Snippets":
            continue
        if len(master_concepts[conceptname])<1:
            continue
        try:
            responseRow[key_format(conceptname)] = {
                    # "Score":score[conceptname],
                    # "Importance":importance[conceptname],
                    "Concept":key_format(conceptname),
                    "Text":""#,
                    # "delta_score":0
                }
            if conceptname not in importance.keys():
                responseRow[key_format(conceptname)]['Importance'] = 1
            else:
                responseRow[key_format(conceptname)]['Importance'] = importance[conceptname]
            Scores = []
            for conceptsubrow in master_concepts[conceptname]:
                if len(conceptsubrow)>0:
                    Scores.append(conceptsubrow[all_params['SENTENCE_SCORE_INDEX']])
                    responseRow[key_format(conceptname)]['Text'] += conceptsubrow[all_params['SENTENCE_TEXT_INDEX']]+". "
            responseRow[key_format(conceptname)]['Scores'] = sum(Scores)/len(Scores)    
        except Exception as e:
            print(e)
            print("error in processMasterConcepts "+conceptname)
            continue
    return responseRow
