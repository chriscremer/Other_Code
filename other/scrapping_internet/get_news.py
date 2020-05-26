


# import os
# import time
# import datetime
# import subprocess

# from selenium import webdriver

# from os.path import expanduser
# home = expanduser("~")


# driver = webdriver.Chrome(home +'/Downloads/chromedriver')

# driver.get("https://ca.reuters.com/")
# print (driver.title)
# print (driver.current_url)



# news = driver.find_element_by_xpath('//*[@id="maincontent"]/div[2]/div[2]/div[1]/div[4]/div/div/div[1]/div/div/div/div/h5/a')
# news.click()


# print (news)
# print(driver)



def getstuff(list1):
	if len(list1) == 0:
		return  ''

	current_text = ''
	for i in range(len(list1)):
		# current_text += str(list1[i].tag) + ' '
		current_text += str(list1[i].text) + ' '
		current_text += getstuff(list1[i])

	return current_text





from lxml import html, etree
import requests

page = requests.get('https://finance.yahoo.com/')
tree = html.fromstring(page.content)
thing = tree.xpath('//*[@id="slingstoneStream-0-Stream"]/ul/li[2]')
print (thing[0].tag)
print (thing[0].keys())
print (thing[0].text)



print (getstuff(thing))


# print ('Johnson' in str(page.content))


# print (str(page.text))
# print (str(page.content))
print ('launches' in str(page.text))
print ('launches' in str(page.content))
print ('virus' in str(page.text))
print ('virus' in str(page.content))
print ('survey' in str(page.text))
print ('survey' in str(page.content))

print ('react-text: 69' in str(page.text))
print ('data-reactid="68"' in str(page.content))

# print(etree.tostring(thing[0], pretty_print=True))

# thing = tree.xpath('//*[@id="slingstoneStream-0-Stream"]/ul/li[2]/div/div/div[2]/div[2]')


# print (thing)











