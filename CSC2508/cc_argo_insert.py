 
import demo_init_argo3
import demo_init
import json
import dbms_postgres
import time
import sys

 
db = demo_init.get_db()


start_time = time.time()
 

count =0

with open('/data1/morrislab/ccremer/CSC2508/nobench/nobench_data_array.json') as f:

    for line in f:

    	sql_text = line

        sql_text = sql_text.strip()
        sql_text = sql_text.rstrip(",")
        if sql_text == "[" or sql_text == "]":
            continue


        sql_text = "insert into a1array OBJECT " + sql_text
        #print sql_text
        #print count
        count += 1

        db.execute_sql(sql_text)


        # counter = counter + 1;
        # sql_text = "insert into extra_json OBJECT " + sql_text
        # # print sql_text
        # # sys.exit()
        # if counter == 1000:
        #     for item in db.execute_sql(sql_text):
        #         print json.dumps(item)
        # else:
        #     exec_list.append(db.execute_sql(sql_text))
             
 
#sql_text = lines[-1]
#sql_text = sql_text.rstrip(",")
# sql_text = "SELECT * FROM test"

# a = db.execute_sql(sql_text)

# # for i in a:
# # 	print i


# sql_text = "SELECT COUNT(*) FROM test"

# a = db.execute_sql(sql_text)

# for i in a:
#  	print i


# for item in db.execute_sql(sql_text):
#     print json.dumps(item)



# sql_text = "SELECT str1, num FROM test"

# a = db.execute_sql(sql_text)

# for i in a:
#  	print i

# SELECT nested_obj.str1, nested_obj.num FROM test;



print "DONE"
print("--- %s seconds ---" % str(time.time() - start_time))