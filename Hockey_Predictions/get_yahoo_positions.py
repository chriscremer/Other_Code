


import numpy as np
from lxml import html
import requests
import scipy
import os, inspect, sys
from scipy import ndimage

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from os.path import expanduser
home = expanduser("~")


# driver = webdriver.Chrome()
# driver.get("https://hockey.fantasysports.yahoo.com/hockey/68895/players")
# driver.get("google.com")

# print driver.title
# print driver.current_url



# import urllib2
# response = urllib2.urlopen('http://python.org/')
# html = response.read()



url = "https://hockey.fantasysports.yahoo.com/hockey/68895/players"

url  = 'http://python.org/'
url = 'https://hockey.fantasysports.yahoo.com/hockey/68895/players?status=A&pos=P&cut_type=33&stat1=S_S_2016&myteam=0&sort=OR&sdir=1&count=50'

s = requests.Session()
r = s.get(url)
print r.url
aa =  r.text
print aa
if 'Hoff' in aa:
	print True
else:
	print False








# https://hockey.fantasysports.yahoo.com/hockey/68895/players?status=A&pos=P&cut_type=33&stat1=S_S_2016&myteam=0&sort=OR&sdir=1&count=75


