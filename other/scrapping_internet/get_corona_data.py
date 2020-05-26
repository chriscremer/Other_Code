



# go through wayback machine dates 


# collect the table data

# plot the stuff I want 



from os.path import expanduser
home = expanduser("~")


from lxml import html
import requests

import pandas as pd

import numpy as np

import pickle



# url = 'https://web.archive.org/web/*/https://www.worldometers.info/coronavirus/'


# page = requests.get(url)
# webpage = html.fromstring(page.content)
# # print (webpage.xpath('//a/@href'))
# print ('/web/20200319/https://www.worldometers.info/coronavirus/' in str(page.content))


# # <a href="/web/20200319/https://www.worldometers.info/coronavirus/">19</a>


url = 'https://web.archive.org/web/20200406/https://www.worldometers.info/coronavirus/'
# page = requests.get(url)
# print (page.content)

# print ('Canada' in str(page.content))

print ('getting', url)

raw_html_tbl_list = pd.read_html(url)

# print(len(raw_html_tbl_list))
# # print ()
# # print (raw_html_tbl_list[1])
# # print ()
# # print (raw_html_tbl_list[2])
# # print ()
# # print ()
# print (raw_html_tbl_list[3])

table = raw_html_tbl_list[3]
# print (table.columns)
# print (table['Country,Other'])
# print()

# print (table.iloc[1])
# print (table.iloc[1]['Country,Other'])

# for i, val in enumerate(table):
# 	print (i, val)


# 	if i ==5:
# 		break

data = {}
data_cases = {}
data_tests = {}
for i in range(1,15):

	country = table.iloc[i]['Country,Other']
	if country == 'China':
		continue

	data[country] = []
	data_cases[country] = []
	data_tests[country] = []


for i in range(1,15):
	country = table.iloc[i]['Country,Other']
	if country == 'China':
		continue
	tests = table.iloc[i]['TotalTests'] 
	cases = table.iloc[i]['TotalCases'] 

	data[country].append(float(cases)/ float(tests))
	data_cases[country].append(float(cases))
	data_tests[country].append(float(tests))

	# print (country, cases, tests)

# print (data)



# for i in range(7, 8):
for i in range(7, 19):

	day = str(i)
	if len(day)==1:
		day= '0' + day

	url = 'https://web.archive.org/web/202004'+day+'/https://www.worldometers.info/coronavirus/'
	print ('getting', url)
	raw_html_tbl_list = pd.read_html(url)
	table = raw_html_tbl_list[3]
	
	for i in range(1,15):
		country = table.iloc[i]['Country,Other']
		if country == 'China' or country not in data:
			continue
		tests = table.iloc[i]['TotalTests'] 
		cases = table.iloc[i]['TotalCases'] 

		data[country].append(float(cases)/ float(tests))
		data_cases[country].append(float(cases))
		data_tests[country].append(float(tests))

print (data)


print()


pickle.dump( [data, data_cases, data_tests], open( home +"/Downloads/rona_data.p", "wb" ) )
print ('saved data to',"/Downloads/rona_data.p" )

# from plotly.subplots import make_subplots


# fig = make_subplots(rows=1, cols=1)
# for key, val in data:
# 	fig.add_trace(
# 	            go.Scatter(x=list(range(6,19)), y=val, mode='lines', name=key,),
# 	            row=1, col=1, 
# 	            )


# fig.show()















