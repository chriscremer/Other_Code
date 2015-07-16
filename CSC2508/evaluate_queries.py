


import pymongo

import time


client = pymongo.MongoClient()

db = client.noBench

collection = db.million



print "Beginnig Query 1"
start = time.time()
db.collection.find({}, ["str1", "num"])
end = time.time()
print 'Query 1 time= ' + str(end - start)


print "Beginnig Query 2"
start = time.time()
db.collection.find({}, ["nested_obj.str", "nested_obj.num"])
end = time.time()
print 'Query 2 time= ' + str(end - start)


print "Beginnig Query 3"
start = time.time()
db.collection.find(
    { "$or" : [ { "sparse_XX0" : {"$exists" : True} },
                { "sparse_XX9" : {"$exists" : True} } ] },
    ["sparse_XX0", "sparse_XX9"])
end = time.time()
print 'Query 3 time= ' + str(end - start)


print "Beginnig Query 4"
start = time.time()
db.collection.find(
        { "$or" : [ { "sparse_XX0" : {"$exists" : True} },
                    { "sparse_YY0" : {"$exists" : True} } ] },
        ["sparse_XX0", "sparse_YY0"])
end = time.time()
print 'Query 4 time= ' + str(end - start)


print "Beginnig Query 5"
start = time.time()
db.collection.find({ "str1" : 'hello' })
end = time.time()
print 'Query 5 time= ' + str(end - start)


print "Beginnig Query 6"
start = time.time()
db.collection.find({ "$and": [{ "num" : {"$gte" : 1234 } },
                                       { "num" : {"$lt"  : 5678 } }]})
end = time.time()
print 'Query 6 time= ' + str(end - start)


print "Beginnig Query 7"
start = time.time()
db.collection.find({ "$and": [{ "dyn1" : {"$gte" : 1234 } },
                                       { "dyn1" : {"$lt"  : 5678 } }]})
end = time.time()
print 'Query 7 time= ' + str(end - start)


print "Beginnig Query 8"
start = time.time()
db.collection.find({ "nested_arr" : 'hello' })
end = time.time()
print 'Query 8 time= ' + str(end - start)


print "Beginnig Query 9"
start = time.time()
db.collection.find({ "sparse_123" : "hello" })
end = time.time()
print 'Query 9 time= ' + str(end - start)


print "Beginnig Query 10"
start = time.time()
db.collection.group(
        {"thousandth" : True},
        {"$and": [{"num" : { "$gte" : 1234 } },
                  {"num" : { "$lt"  : 5678 } }]},
        { "total" : 0 },
        "function(obj, prev) { prev.total += 1; }")
end = time.time()
print 'Query 10 time= ' + str(end - start)

