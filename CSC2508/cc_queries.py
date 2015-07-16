


import demo_init
import demo_init_argo3
import json
import dbms_postgres
import time
import sys
import pymongo
import numpy as np
import psycopg2
import dbms_postgres
#to plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_run_times(argo1_times, argo3_times, mongo_times, argo1_stds, argo3_stds, mongo_stds):


	# the x locations for the groups
	ind = np.arange(len(argo1_times))
	# the width of the bars
	width = 0.20

	fig, ax = plt.subplots()
	argo1 = ax.bar(ind-width, argo1_times, width, color='purple', yerr=argo1_stds)
	argo3 = ax.bar(ind, argo3_times, width, color='red', yerr=argo3_stds)
	mongo = ax.bar(ind+width, mongo_times[:len(argo1_times)], width, color='blue', yerr=mongo_stds)

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Time (Seconds)')
	ax.set_xlabel('Query Number')
	#ax.set_title('Scores by group and gender')
	ax.set_xticks(ind+width)
	ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9'])
	#ax.legend( (random[0], pca[0], ica[0]), ('Random', 'PCA', 'ICA') )
	ax.legend( (argo1[0], argo3[0], mongo[0]), ('Argo1', 'Argo3', 'MongoDB'))

	#add numbers
	for rect in argo1:
		height = rect.get_height()
		ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height), ha='center', va='bottom')
	for rect in argo3:
		height = rect.get_height()
		ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height), ha='center', va='bottom')
	for rect in mongo:
		height = rect.get_height()
		ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height), ha='center', va='bottom')

	plt.savefig('bar_plot_run_times.pdf')
	print 'Saved plot'



def run_psql_query(query, numb):

	print "Beginnig PSQL Query " + str(numb)
	start = time.time()
	db.execute("""SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'""")
	for table in db.fetchall():
		print(table)
	qwer
	#db.execute("SELECT str1, num FROM test")
	end = time.time()
	print 'Query ' + str(numb) + ' time= ' + str(end - start)
	return end - start


def run_argo_query(query, numb):

	times = []
	print "Beginnig Argo Query " + str(numb)
	#run 10 times
	for iter1 in range(10):
		start = time.time()
		db.execute_sql(query)
		end = time.time()
		times.append(start-end)

	times.sort()
	#remove longest and shortest
	del times[0]
	del times[-1]
	mean = np.mean(times)
	std = np.std(times)

	print 'Argo Query ' + str(numb) + ' time= ' + str(end - start)
	return end - start


def run_mongo_query(query, numb):

	times = []
	print "Beginnig Mongo Query " + str(numb)
	#run 10 times
	for iter1 in range(10):
		start = time.time()
		query_output = query
		for j in query_output:
			print j
		end = time.time()
		times.append(start-end)

	times.sort()
	#remove longest and shortest
	del times[0]
	del times[-1]
	mean = np.mean(times)
	std = np.std(times)

	print 'Mongo Query ' + str(numb) + ' time= ' + str(mean)
	return mean, std


argo_queries = [
					"SELECT str1, num FROM nobench_1mil",
					"SELECT nested_obj.str1, nested_obj.num FROM nobench_1mil",
					"SELECT sparse_XX0, sparse_XX9 FROM nobench_1mil",
					"SELECT sparse_XX0, sparse_YY0 FROM nobench_1mil",
					"SELECT * FROM nobench_1mil WHERE str1 = \"hello\"",
					"SELECT * FROM nobench_1mil WHERE keystr = \"num\" and valnum > 1234 AND valnum < 5678",
					"SELECT * FROM nobench_1mil WHERE keystr = \"dyn1\" and valnum > 1234 AND valnum < 5678",
					"SELECT * FROM nobench_1mil WHERE \"hello\" = ANY nested_arr",
					"SELECT * FROM nobench_1mil WHERE sparse_123 = \"hello\"",
					
				]
#"SELECT count(*) FROM test WHERE keystr = \"num\" and valnum > 1234 AND valnum < 5678 GROUP BY thousandth"


client = pymongo.MongoClient()
db = client.noBench
collection = db.million

mongo_queries = [
					db.collection.find({}, ["str1", "num"]),
					db.collection.find({}, ["nested_obj.str", "nested_obj.num"]),
					db.collection.find(
   						{ "$or" : [ { "sparse_XX0" : {"$exists" : True} },
                		{ "sparse_XX9" : {"$exists" : True} } ] },
   						["sparse_XX0", "sparse_XX9"]),
					db.collection.find(
        				{ "$or" : [ { "sparse_XX0" : {"$exists" : True} },
                    	{ "sparse_YY0" : {"$exists" : True} } ] },
        				["sparse_XX0", "sparse_YY0"]),
					db.collection.find({ "str1" : 'hello' }),
					db.collection.find({ "$and": [{ "num" : {"$gte" : 1234 } },
                                       { "num" : {"$lt"  : 5678 } }]}),
					db.collection.find({ "$and": [{ "dyn1" : {"$gte" : 1234 } },
                                       { "dyn1" : {"$lt"  : 5678 } }]}),
					db.collection.find({ "nested_arr" : 'hello' }),
					db.collection.find({ "sparse_123" : "hello" }),
					db.collection.group(
				        {"thousandth" : True},
				        {"$and": [{"num" : { "$gte" : 1234 } },
				                  {"num" : { "$lt"  : 5678 } }]},
				        { "total" : 0 },
				        "function(obj, prev) { prev.total += 1; }")
				]


if __name__ == "__main__":

	print 'Get info about PostgreSQL database'
	conn = psycopg2.connect("user=ccremer dbname=argo")
	db = conn.cursor()
	db.execute("""SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'""")
	for table in db.fetchall():
		print(table)


	# print '\nStarting PSQL\n'
	# conn = psycopg2.connect("user=ccremer dbname=argo")
	# db = conn.cursor()
	# for i in range(1, len(argo_queries)+1):
	# 	time1 = run_psql_query(argo_queries[i-1], i)
	# 	argo1_times.append(time1)

	argo1_times = []
	argo3_times = []
	mongo_times = []
	argo1_stds = []
	argo3_stds = []
	mongo_stds = []

	print '\nStarting Argo1\n'
	db = demo_init.get_db()
	for i in range(1, len(argo_queries)+1):
		time1, std = run_argo_query(argo_queries[i-1], i)
		argo1_times.append(time1)
		argo1_stds.append(std)


	print '\nStarting Argo3\n'
	db = demo_init_argo3.get_db()
	for i in range(1, len(argo_queries)+1):
		time1, std = run_argo_query(argo_queries[i-1], i)
		argo3_times.append(time1)
		argo3_stds.append(std)


	print '\nStarting Mongo\n'
	for i in range(1, len(mongo_queries)+1):
		time1, std = run_mongo_query(mongo_queries[i-1], i)
		mongo_times.append(time1)
		mongo_stds.append(std)


	plot_run_times(argo1_times, argo3_times, mongo_times, argo1_stds, argo3_stds, mongo_stds)



 
 
 
	# db = demo_init.get_db()
	 
	# connection = db.dbms.connection
	# cursor = connection.cursor()
	 
	# run_times = []
	 
	 
	# sql_list = ["SELECT * FROM (SELECT valstr, objid FROM argo_onemill_fivedeep_str WHERE keystr = 'nested_obj.str') a INNER JOIN (SELECT valnum, objid FROM argo_nobench_main_num WHERE keystr = 'nested_obj.num') b ON (a.objid = b.objid)",
	# "SELECT * FROM (SELECT valstr, objid FROM argo_onemill_fivedeep_str WHERE keystr = '5_nested_obj.deep.deep.deep.deep.str') a INNER JOIN (SELECT valnum, objid FROM argo_nobench_main_num WHERE keystr = '5_nested_obj.deep.deep.deep.deep.num') b ON (a.objid = b.objid)",
	# "SELECT * FROM argo_onemill_fivedeep_num WHERE keystr = 'nested_obj.num' and valnum = 245072",
	# "SELECT * FROM argo_onemill_fivedeep_num WHERE keystr = '5_nested_obj.deep.deep.deep.deep.num' and valnum = 245072",
	# "SELECT * FROM argo_onemill_fivedeep_num WHERE keystr = 'nested_obj.num' and valnum > 10000 AND valnum < 99999999",
	# "SELECT * FROM argo_onemill_fivedeep_num WHERE keystr = '5_nested_obj.deep.deep.deep.deep.num' and valnum > 10000 AND valnum < 99999999",
	# "(SELECT objid from argo_onemill_fivedeep_str where keystr = 'nested_obj.str' and valstr != '') UNION (SELECT objid from argo_onemill_fivedeep_num where keystr = 'nested_obj.num' and valnum > 25)",
	# "(SELECT objid from argo_onemill_fivedeep_str where keystr = '5_nested_obj.deep.deep.deep.deep.str' and valstr != '') UNION (SELECT objid from argo_onemill_fivedeep_num where keystr = '5_nested_obj.deep.deep.deep.deep.num' and valnum > 25)",
	# "DROP TABLE IF EXISTS argo_im; CREATE TEMP TABLE argo_im AS SELECT objid FROM argo_onemill_fivedeep_num WHERE keystr = 'nested_obj.num' and valnum BETWEEN 0 AND 99999999; SELECT argo_join_left.objid, argo_join_right.objid FROM argo_onemill_fivedeep_str AS argo_join_left, argo_onemill_fivedeep_str AS argo_join_right WHERE argo_join_left.keystr = 'nested_obj.str' AND argo_join_right.keystr = 'str1' AND argo_join_left.valstr = argo_join_right.valstr AND argo_join_left.objid IN (SELECT objid FROM argo_im);",
	# "DROP TABLE IF EXISTS argo_im; CREATE TEMP TABLE argo_im AS SELECT objid FROM argo_onemill_fivedeep_num WHERE keystr = '5_nested_obj.deep.deep.deep.deep.num' and valnum BETWEEN 0 AND 99999999; SELECT argo_join_left.objid, argo_join_right.objid FROM argo_onemill_fivedeep_str AS argo_join_left, argo_onemill_fivedeep_str AS argo_join_right WHERE argo_join_left.keystr = '5_nested_obj.deep.deep.deep.deep.str' AND argo_join_right.keystr = 'str1' AND argo_join_left.valstr = argo_join_right.valstr AND argo_join_left.objid IN (SELECT objid FROM argo_im);"]
	 
	# f = open('deep_comparison_ARGO3_runtimes.txt','w')
	 
	# for sql_text in sql_list:
	#     run_times = []
	#     for iteration in range(0,10):
	#         start_time = time.time()
	#         cursor.execute(sql_text)
	#         runtime = time.time() - start_time
	#         run_times.append(runtime)
	 
	# f.close()