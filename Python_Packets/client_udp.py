import socket
import time
import datetime

#ip address of server
ip = "172.31.35.230"
#choose any port number, make it around 5000
port = 5005
#bind a socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#get time to find difference in time between each packet
last = time.time()
#open a file to write the data
fo = open("packet_stats.txt", "w")

#send 15000 messages
for i in range(0,15000): 
	try:
		#the message
		msg = "short msg %d" % (i)
		#send message
		bytes_sent = sock.sendto(msg, (ip, port))
		theTime = time.time()
		#find difference in time between each packet
		dif = theTime - last
		now = datetime.datetime.now()
		now1 = now.second
		now2 = now.microsecond
		#print what it just sent
		a = "sent %d sec: %d usec:%d dif: %f bytes:%d" % (i, now1, now2, dif, bytes_sent)
		print a		
		fo.write(a)
		fo.write("\n")		
		last = theTime
	#if there is a error when sending, it will print caught
	except socket.error:
		print "caught"

#close file
fo.close()

