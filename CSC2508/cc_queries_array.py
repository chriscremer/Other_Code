


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
import subprocess

def plot_run_times(argo1_times, argo3_times, mongo_times, argo1_stds, argo3_stds, mongo_stds):

	# the x locations for the groups
	ind = np.arange(len(argo1_times))
	# the width of the bars
	width = 0.20

	fig, ax = plt.subplots()
	argo1_rects = ax.bar(ind-width, argo1_times, width, color='purple', yerr=argo1_stds)
	argo3_rects = ax.bar(ind, argo3_times, width, color='red', yerr=argo3_stds)
	mongo_rects = ax.bar(ind+width, mongo_times, width, color='blue', yerr=mongo_stds)

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Time (Seconds)')
	ax.set_xlabel('Query Number')
	#ax.set_title('Scores by group and gender')
	ax.set_xticks(ind+width)
	ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
	#ax.legend( (random[0], pca[0], ica[0]), ('Random', 'PCA', 'ICA') )
	ax.legend( (argo1_rects[0], argo3_rects[0], mongo_rects[0]), ('Argo1', 'Argo3', 'MongoDB'))

	#add numbers
	# for rect in argo1_rects:
	# 	height = rect.get_height()
	# 	ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%f'%float(height), ha='center', va='bottom')
	# for rect in argo3_rects:
	# 	height = rect.get_height()
	# 	ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%f'%float(height), ha='center', va='bottom')
	# for rect in mongo_rects:
	# 	height = rect.get_height()
	# 	ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%f'%float(height), ha='center', va='bottom')

	plt.savefig('bar_plot_run_times_array.pdf')
	print 'Saved plot'



# def run_psql_query(query, numb):

# 	print "Beginnig PSQL Query " + str(numb)
# 	start = time.time()
# 	db.execute("""SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'""")
# 	for table in db.fetchall():
# 		print(table)
# 	qwer
# 	#db.execute("SELECT str1, num FROM test")
# 	end = time.time()
# 	print 'Query ' + str(numb) + ' time= ' + str(end - start)
# 	return end - start


def run_argo_query(query, numb):

	times = []
	print "Beginnig Argo Query " + str(numb)
	#run 10 times
	for iter1 in range(10):
		start = time.time()
		cursor.execute(query)
		query_output = cursor.fetchall()
		for outp in query_output:
			continue
		end = time.time()
		times.append(end-start)
		# for outp in query_output:
		# 	print outp
		# print query_output.count()

	times.sort()
	#remove longest and shortest
	del times[0]
	del times[-1]
	mean = np.mean(times)
	std = np.std(times)

	print 'Argo Query ' + str(numb) + ' time= ' + str(mean)
	f.write('Argo Query ' + str(numb) + ' time= ' + str(mean) + '\n')
	return mean, std


def run_argo3_query(query, numb):

	times = []
	print "Beginning Argo3 Query " + str(numb)
	#run 10 times
	for iter1 in range(10):
		start = time.time()
		cursor.execute(query)
		query_output = cursor.fetchall()
		for outp in query_output:
			continue
		end = time.time()
		times.append(end-start)

	times.sort()
	#remove longest and shortest
	del times[0]
	del times[-1]
	mean = np.mean(times)
	std = np.std(times)

	print 'Argo3 Query ' + str(numb) + ' time= ' + str(mean)
	f.write('Argo3 Query ' + str(numb) + ' time= ' + str(mean) + '\n')
	return mean, std


def run_mongo_query(query, numb):

	times = []
	print "Beginning Mongo Query " + str(numb)

	# print 'Number of docs in this collection'
	# print db.collection.count()

	#run 10 times
	for iter1 in range(10):
		start = time.time()
		if numb == 1:
			query_output = db.array.find({}, ["str1", "num"])
			for outp in query_output:
				continue
		elif numb == 2:
			query_output = db.array.find({}, ["nested_obj.str", "nested_obj.num"])
			for outp in query_output:
				continue
		elif numb == 3:
			query_output = db.array.find(
						{ "$or" : [ { "sparse_110" : {"$exists" : True} },
						{ "sparse_119" : {"$exists" : True} } ] },
						["sparse_110", "sparse_119"])
			for outp in query_output:
				continue
		elif numb == 4:
			query_output = db.array.find(
						{ "$or" : [ { "sparse_110" : {"$exists" : True} },
						{ "sparse_220" : {"$exists" : True} } ] },
						["sparse_110", "sparse_220"])
			for outp in query_output:
				continue
		elif numb == 5:
			query_output = db.array.find({ "str1" : 'join' })
			for outp in query_output:
				continue
		elif numb == 6:
			query_output = db.array.find({ "$and": [{ "num" : {"$gte" : 1234 } }, { "num" : {"$lt"  : 5678 } }]})
			for outp in query_output:
				continue
		elif numb == 7:
			query_output = db.array.find({ "$and": [{ "dyn1" : {"$gte" : 1234 } }, { "dyn1" : {"$lt"  : 5678 } }]})
			for outp in query_output:
				continue
		elif numb == 8:
			query_output = db.array.find({ "nested_arr" : 'join' })
			for outp in query_output:
				continue
		elif numb == 9:
			query_output = db.array.find({ "sparse_123" : 'join' })
			for outp in query_output:
				continue
		elif numb == 10:
			query_output = db.array.group(
						{"thousandth" : True},
						{"$and": [{"num" : { "$gte" : 1234 } },
						          {"num" : { "$lt"  : 5678 } }]},
						{ "total" : 0 },
						"function(obj, prev) { prev.total += 1; }")

		else:
			print 'somehting wrong'

		end = time.time()

		#print query_output.count()
		# for j in query_output[:1]:
		# 	print j
		
		times.append(end-start)

	times.sort()
	#remove longest and shortest
	del times[0]
	del times[-1]
	mean = np.mean(times)
	std = np.std(times)

	print 'Mongo Query ' + str(numb) + ' time= ' + str(mean)
	f.write('Mongo Query ' + str(numb) + ' time= ' + str(mean) + '\n')
	return mean, std


argo1_queries = [
				"SELECT * FROM (SELECT valstr, objid FROM argo_a1array_data WHERE keystr = 'str1') a INNER JOIN (SELECT valnum, objid FROM argo_a1array_data WHERE keystr = 'num') b ON (a.objid = b.objid)",
				"SELECT * FROM (SELECT valstr, objid FROM argo_a1array_data WHERE keystr = 'nested_obj.str') a INNER JOIN (SELECT valnum, objid FROM argo_a1array_data WHERE keystr = 'nested_obj.num') b ON (a.objid = b.objid)",
				"SELECT valstr, objid FROM argo_a1array_data WHERE keystr = 'sparse_110' or keystr = 'sparse_119'",
				"SELECT valstr, objid FROM argo_a1array_data WHERE keystr = 'sparse_110' or keystr = 'sparse_220'",
				"SELECT * FROM argo_a1array_data WHERE keystr = 'str1' and valstr = 'join'",
				"SELECT * FROM argo_a1array_data WHERE keystr = 'num' and valnum > 1234 AND valnum < 5678",
				"SELECT * FROM argo_a1array_data WHERE keystr = 'dyn1' and valnum > 1234 AND valnum < 5678",
				"SELECT objid FROM argo_a1array_data WHERE keystr SIMILAR TO 'nested_arr\[[0123456789]+\]' AND valstr = 'join'",
				"SELECT * FROM argo_a1array_data WHERE keystr = 'sparse_111' and valstr = 'join'",
				"DROP TABLE IF EXISTS temp_argo; CREATE TEMP TABLE temp_argo AS SELECT objid FROM argo_a1array_data WHERE keystr = 'num' and valnum > 1234 AND valnum < 5678; SELECT count(*) FROM argo_a1array_data WHERE objid in (SELECT objid FROM temp_argo) AND keystr = 'thousandth' GROUP BY valnum"
				]



# "SELECT str1, num FROM argo_a1_data",
# "SELECT nested_obj.str1, nested_obj.num FROM argo_a1_data",
# "SELECT sparse_110, sparse_119 FROM argo_a1_data",
# "SELECT sparse_110, sparse_220 FROM argo_a1_data",
# "SELECT * FROM argo_a1_data WHERE str1 = \"join\"",
# "SELECT * FROM argo_a1_data WHERE keystr = \"num\" and valnum > 1234 AND valnum < 5678",
# "SELECT * FROM argo_a1_data WHERE keystr = \"dyn1\" and valnum > 1234 AND valnum < 5678",
# "SELECT * FROM argo_a1_data WHERE \"join\" = ANY nested_arr",
# "SELECT * FROM argo_a1_data WHERE sparse_111 = \"join\""
# "SELECT count(*) FROM test WHERE keystr = \"num\" and valnum > 1234 AND valnum < 5678 GROUP BY thousandth"

argo3_queries = [
					"SELECT * FROM (SELECT valstr, objid FROM argo_a3array_str WHERE keystr = 'str1') a INNER JOIN (SELECT valnum, objid FROM argo_a3array_num WHERE keystr = 'num') b ON (a.objid = b.objid)",
					"SELECT * FROM (SELECT valstr, objid FROM argo_a3array_str WHERE keystr = 'nested_obj.str') a INNER JOIN (SELECT valnum, objid FROM argo_a3array_num WHERE keystr = 'nested_obj.num') b ON (a.objid = b.objid)",
					"SELECT valstr, objid FROM argo_a3array_str WHERE keystr = 'sparse_110' or keystr = 'sparse_119'",
					"SELECT valstr, objid FROM argo_a3array_str WHERE keystr = 'sparse_110' or keystr = 'sparse_220'",
					"SELECT * FROM argo_a3array_str WHERE keystr = 'str1' and valstr = 'join'",
					"SELECT * FROM argo_a3array_num WHERE keystr = 'num' and valnum > 1234 AND valnum < 5678",
					"SELECT * FROM argo_a3array_num WHERE keystr = 'dyn1' and valnum > 1234 AND valnum < 5678",
					"SELECT objid FROM argo_a3array_str WHERE keystr SIMILAR TO 'nested_arr\[[0123456789]+\]' AND valstr = 'join'",
					"SELECT * FROM argo_a3array_str WHERE keystr = 'sparse_111' and valstr = 'join'",
					"DROP TABLE IF EXISTS temp_argo; CREATE TEMP TABLE temp_argo AS SELECT objid FROM argo_a3array_num WHERE keystr = 'num' and valnum > 1234 AND valnum < 5678; SELECT count(*) FROM argo_a3array_num WHERE objid in (SELECT objid FROM temp_argo) AND keystr = 'thousandth' GROUP BY valnum"

				]

# client = pymongo.MongoClient()
# db = client.noBench
# collection = db.million

# mongo_queries = [
# 					db.collection.find({}, ["str1", "num"]),
# 					db.collection.find({}, ["nested_obj.str", "nested_obj.num"]),
# 					db.collection.find(
#    						{ "$or" : [ { "sparse_XX0" : {"$exists" : True} },
#                 		{ "sparse_XX9" : {"$exists" : True} } ] },
#    						["sparse_XX0", "sparse_XX9"]),
# 					db.collection.find(
#         				{ "$or" : [ { "sparse_XX0" : {"$exists" : True} },
#                     	{ "sparse_YY0" : {"$exists" : True} } ] },
#         				["sparse_XX0", "sparse_YY0"]),
# 					db.collection.find({ "str1" : 'join' }),
# 					db.collection.find({ "$and": [{ "num" : {"$gte" : 1234 } },
#                                        { "num" : {"$lt"  : 5678 } }]}),
# 					db.collection.find({ "$and": [{ "dyn1" : {"$gte" : 1234 } },
#                                        { "dyn1" : {"$lt"  : 5678 } }]}),
# 					db.collection.find({ "nested_arr" : 'join' }),
# 					db.collection.find({ "sparse_123" : "join" }),
# 				]




if __name__ == "__main__":

	#Connect to PostgreSQL
	conn = psycopg2.connect("user=ccremer dbname=argo")
	cursor = conn.cursor()
	print 'PostgreSQL databases:'
	cursor.execute("""SELECT datname from pg_database""")
	rows = cursor.fetchall()
	for row in rows:
		print row[0]
	print 'PostgreSQL argo database tables:'
	cursor.execute("""SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'""")
	for table in cursor.fetchall():
		print(table)

	#Connect to MongoDB
	client = pymongo.MongoClient()
	db = client.noBench
	#collection = db.million
	print '\nMongo databases:'
	print client.database_names()
	print 'Mongo noBench collections:'
	print db.collection_names()

	# print db.million.count()
	# qwer = db.collection.find()
	# print '111'
	# for thign in qwer:
	# 	print thign
	# print '2222'
	# print db.command("dbstats")


  

	# # start_time = time.time()
	# # subprocess.call("/data1/morrislab/ccremer/mongodb/mongodb-linux-x86_64-rhel62-3.0.4/bin/mongoimport -d test_db -c test_collection /data1/morrislab/ccremer/CSC2508/nobench/nobench_data.json --jsonArray", shell=True)
	# # runtime = time.time() - start_time
	# # print str(runtime) + "seconds"

	# # db = client.test_db
	# collection = db.test_collection
	# print '111'
	# for thign in qwer:
	# 	print thign
	# print '2222'
	# print db.collection.count()

	# az

	argo1_times = []
	argo3_times = []
	mongo_times = []
	argo1_stds = []
	argo3_stds = []
	mongo_stds = []



	f = open('runtimes_nobench_array.txt', 'w')

	print '---------------------------------'
	print 'Starting Queries'
	print '---------------------------------'

	print '\nStarting Argo1\n'
	#argo_cursor = demo_init.get_db()
	for i in range(1, len(argo1_queries)+1):
		time1, std = run_argo_query(argo1_queries[i-1], i)
		argo1_times.append(time1)
		argo1_stds.append(std)


	print '\nStarting Argo3\n'
	#argo_cursor = demo_init_argo3.get_db()
	#argo_cursor = cursor

	# db1 = demo_init_argo3.get_db()
	# connection1 = db1.dbms.connection
	# cursor1 = connection1.cursor()

	for i in range(1, len(argo3_queries)+1):
		time1, std = run_argo3_query(argo3_queries[i-1], i)
		argo3_times.append(time1)
		argo3_stds.append(std)


	print '\nStarting Mongo\n'
	for i in range(1, len(argo3_queries)+1):
		time1, std = run_mongo_query(argo3_queries[i-1], i)
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