import socket
import time
import datetime

#ip address of this computer (the server)
ip = "172.31.35.21"
#choose any port number, around 5000
port = 5005
#bind to a socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((ip, port))
#record the time for
last = time.time()
#open a file to save the stats
fo = open("packet_stats.txt", "w")

#infinite loop to continously accept packets
while True:
	#get time to compare vs last time a packet was received
	theTime = time.time()
	#reception of packet
	data, addr = sock.recvfrom(1024)
	dif = theTime - last
	#get time it is now
	now = datetime.datetime.now()
	now1 = now.second
	now2 = now.microsecond
	#write to the file 
	a = "%s sec: %d usec: %d dif: %f" % (data, now1, now2, dif)
	fo.write(a)
	fo.write("\n")
	#print to stdout
	print "received:", data, " ", time.time(), " dif:", dif
	last = theTime
	
	

