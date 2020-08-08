import pymysql
import modules.common_helper as cm_help

def connect(sql,field_names=False):
	params = cm_help.get_params()['mysql']
	# Open database connection
	db = pymysql.connect(params["host"],params["user"],params["password"],params["db"] )

	cursor = db.cursor()

	cursor.execute(sql)

	data = cursor.fetchall()

	if field_names is True:
		processed = []
		num_fields = len(cursor.description)
		field_names = [i[0] for i in cursor.description]
		for dtrow in data:
			pdrow = {}
			for idx,dt in enumerate(dtrow):
				pdrow[field_names[idx]] = dt
			processed.append(pdrow)
		return processed
	db.commit()
	# # disconnect from server
	db.close()
	return data

def getMerAcquisitionDetails():
	sql = "SELECT `ID`,`ACQUIREEID`,`ACQUIRERID`,`ESTIMATEDACQUISITIONONMKTCAP`,`ESTIMATEDACQUISITIONPRICE`,`ESTIMATEDPRICERANGE1`,`ESTIMATEDPRICERANGE2`,`BUBBLETEXT` as text,`SOURCEID`,`SOURCE`,`URL`,`CREDIBILITY`,`ArticleDate`,`ALPHAFACTOR`,`STARTTS`,`SICONTENTS`,`PREDICTIONPROBABILITY`,`PREDICTIONPROBABILITYORG`,`TARGETSYMBOLLTP`,`PREDICTIONINFO`,`RELIABILITY`,`RELIABILITYSTAR`,`REMAININGINFO`,`ESTIMATEDPERCENTAGEREMAINING1`,`ESTIMATEDPERCENTAGEREMAINING2`,`CATEGORY`,`HIGHLOWPRICE`,`PROGRESSPERCENTAGE`,`CURRENTPRICE`,`CURRENTSTATUS`,`SPOUTPERFORM`,`RUMORDECIDER`,`ESTPRICEIMPACT1`,`ESTPRICEIMPACT2`,`RUMORPRICE` FROM mer_acquisitiondetails_new"
	return connect(sql,True)