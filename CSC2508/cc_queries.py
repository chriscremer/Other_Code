


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


def run_psql_query(query, numb):

	print "Beginnig Query " + str(numb)
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

	print "Beginnig Query " + str(numb)
	start = time.time()
	db.execute_sql(query)
	end = time.time()
	print 'Query ' + str(numb) + ' time= ' + str(end - start)
	return end - start

def run_mongo_query(query, numb):

	print "Beginnig Query " + str(numb)
	start = time.time()
	a = query
	for j in a:
		print j
	end = time.time()
	print 'Query ' + str(numb) + ' time= ' + str(end - start)
	return end - start


argo1_times = []
argo3_times = []
mongo_times = []

argo_queries = [
					"SELECT str1, num FROM test",
					"SELECT nested_obj.str1, nested_obj.num FROM test",
					"SELECT sparse_XX0, sparse_XX9 FROM test",
					"SELECT sparse_XX0, sparse_YY0 FROM test",
					"SELECT * FROM test WHERE str1 = \"hello\"",
					"SELECT * FROM test WHERE keystr = \"num\" and valnum > 1234 AND valnum < 5678",
					"SELECT * FROM test WHERE keystr = \"dyn1\" and valnum > 1234 AND valnum < 5678",
					"SELECT * FROM test WHERE \"hello\" = ANY nested_arr",
					"SELECT * FROM test WHERE sparse_123 = \"hello\"",
					"SELECT count(*) FROM test WHERE keystr = \"num\" and valnum > 1234 AND valnum < 5678 GROUP BY thousandth"
				]

# mongo_queries = [
# 					db.collection.find({}, ["str1", "num"]),
# 					find({}, ["nested_obj.str", "nested_obj.num"]),
# 					find(
#    						{ "$or" : [ { "sparse_XX0" : {"$exists" : True} },
#                 		{ "sparse_XX9" : {"$exists" : True} } ] },
#    						["sparse_XX0", "sparse_XX9"]),
# 					find(
#         				{ "$or" : [ { "sparse_XX0" : {"$exists" : True} },
#                     	{ "sparse_YY0" : {"$exists" : True} } ] },
#         				["sparse_XX0", "sparse_YY0"]),
# 					find({ "str1" : 'hello' }),
# 					find({ "$and": [{ "num" : {"$gte" : 1234 } },
#                                        { "num" : {"$lt"  : 5678 } }]}),
# 					find({ "$and": [{ "dyn1" : {"$gte" : 1234 } },
#                                        { "dyn1" : {"$lt"  : 5678 } }]})
# 					find({ "nested_arr" : 'hello' }),
# 					find({ "sparse_123" : "hello" }),
# 					group(
# 				        {"thousandth" : True},
# 				        {"$and": [{"num" : { "$gte" : 1234 } },
# 				                  {"num" : { "$lt"  : 5678 } }]},
# 				        { "total" : 0 },
# 				        "function(obj, prev) { prev.total += 1; }")
# 				]


if __name__ == "__main__":

	print 'Get info about PostgreSQL database'
	conn = psycopg2.connect("user=ccremer dbname=argo")
	db = conn.cursor()
	db.execute("""SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'""")


	print '\nStarting PSQL\n'
	conn = psycopg2.connect("user=ccremer dbname=argo")
	db = conn.cursor()
	for i in range(1, len(argo_queries)+1):
		time1 = run_psql_query(argo_queries[i-1], i)
		argo1_times.append(time1)


	print '\nStarting Argo1\n'
	db = demo_init.get_db()
	for i in range(1, len(argo_queries)+1):
		time1 = run_argo_query(argo_queries[i-1], i)
		argo1_times.append(time1)


	print '\nStarting Argo3\n'
	db = demo_init_argo3.get_db()
	for i in range(1, len(argo_queries)+1):
		time1 = run_argo_query(argo_queries[i-1], i)
		argo3_times.append(time1)


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


	print '\nStarting Mongo\n'
	# client = pymongo.MongoClient()
	# db = client.noBench
	# collection = db.million
	for i in range(1, len(mongo_queries)+1):
		time1 = run_mongo_query(mongo_queries[i-1], i)
		mongo_times.append(time1)


	#to plot
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt

	# the x locations for the groups
	ind = np.arange(len(argo1_times))
	# the width of the bars
	width = 0.20

	fig, ax = plt.subplots()
	argo1 = ax.bar(ind-width, argo1_times, width, color='purple', yerr=.0001)
	argo3 = ax.bar(ind, argo3_times, width, color='red', yerr=.0001)
	mongo = ax.bar(ind+width, mongo_times[:len(argo1_times)], width, color='blue', yerr=.0001)



	# add some text for labels, title and axes ticks
	ax.set_ylabel('Time (Seconds)')
	ax.set_xlabel('Query Number')
	#ax.set_title('Scores by group and gender')
	ax.set_xticks(ind+width)
	ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9'])
	#ax.legend( (random[0], pca[0], ica[0]), ('Random', 'PCA', 'ICA') )
	ax.legend( (argo1[0], argo3[0], mongo[0]), ('Argo1', 'Argo3', 'MongoDB'))


	plt.savefig('bar_plot.pdf')
	print 'Saved plot'