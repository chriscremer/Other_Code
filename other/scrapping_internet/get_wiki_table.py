
# code for getting the plot section of wiki articles


url = 'https://en.wikipedia.org/wiki/List_of_American_films_of_2019'




# import requests
# import json
# from bs4 import BeautifulSoup

# r = requests.get(url)
# soup = BeautifulSoup(r.text, 'html.parser')
# print (soup.title.text)
# # print (soup.prettify())






# import lxml.html
# import requests

# url =  "https://en.wikipedia.org/wiki/List_of_American_films_of_2019"
# response = requests.get(url, stream=True)
# response.raw.decode_content = True
# tree = lxml.html.parse(response.raw)








# import pandas as pd

# raw_html_tbl_list = pd.read_html(url)

# print (len(raw_html_tbl_list))
# # print (raw_html_tbl_list[0])
# # print (raw_html_tbl_list[1])
# # print (raw_html_tbl_list[2])

# print (raw_html_tbl_list[3].columns)
# # print ('Title' in raw_html_tbl_list[3].columns and 
# #             'Production company' in raw_html_tbl_list[3].columns)
# ismoviestable = lambda x: ('Title' in x.columns and 
#           'Production company' in x.columns)
# # print (ismoviestable(raw_html_tbl_list[3]))

# # fasdf
# print ([ismoviestable(x) for x in raw_html_tbl_list])

# # print (raw_html_tbl_list[4])
# # print (raw_html_tbl_list[5])



# print (raw_html_tbl_list[3]['Title'][0])





import pandas as pd
import requests
from bs4 import BeautifulSoup



response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# print (soup.children)


# table = soup.find('table')

# print (list(table.children))
# fsa

















# this seems useufl
# https://srome.github.io/Parsing-HTML-Tables-in-Python-with-BeautifulSoup-and-pandas/
# and this
# https://stackoverflow.com/questions/56757261/extract-href-using-pandas-read-html









# OPTION 1


df = pd.read_html(url)[3]
# print (df)
# print (df.columns)
# print (df['Title'])
# fds


response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
# table = soup.find('table')
table = soup.findAll("table")[3]
# print (len (table))
# print (table)
# fsdfa


links = []
for i, tr in enumerate(table.findAll("tr")):
    # print (i, tr)
    # fds

    # table header
    ths = tr.findAll("th")
    if ths != []:
        pass
        # for each in ths:
        #     columns.append(each.text)
    else:
        # table elements
        trs = tr.findAll("td")
        # print (trs[0])
        # print (trs[1])
        # print (trs[2])
        # print (trs[3])
        # fdsa
        try:
            link = trs[0].find('a')['href']
            links.append(link)
        except:
            print (trs[0])
            fsa
            pass
        # print (link)
        # fsa
        # for each in trs:
        #     try:
        #         link = each.find('a')['href']
        #         links.append(link)
        #         print (i, link)

        #     except:
        #         pass


print (df)
print (len (links))
df['Link'] = links
print (df)



fdsa














# # OPTION 2


# tables = soup.find_all('table')
# table = tables[3]

# records = []
# columns = []
# for tr in table.findAll("tr"):
#     ths = tr.findAll("th")
#     if ths != []:
#         for each in ths:
#             columns.append(each.text)
#     else:
#         trs = tr.findAll("td")
#         print (trs[0])
#         print (trs[1])
#         print (trs[2])
#         fdsa
#         record = []
#         for each in trs:
#             try:
#                 link = each.find('a')['href']
#                 text = each.text
#                 record.append(link)
#                 record.append(text)
#             except:
#                 text = each.text
#                 record.append(text)
#         records.append(record)

# columns.insert(1, 'Link')
# df = pd.DataFrame(data=records, columns = columns)


# print (df.columns)
# print (df["Title\n"])

# fsd
















print (len(tables))

table = tables[3]
for tr in table.findAll("tr")[:5]:
    # ths = tr.findAll("th")
    # if ths != []:
        # print (tr, ths)

    # print (tr)
    # print ()

    # if tr !=[]:
    # print (tr)
    tds = tr.findAll("td")
    # print (tds)
    # fdsa
    # print (len(tds))
    if len (tds) > 0:
        print (tds)
        fasf
        print (tds[1])
        print ()
        print (tds[1].find("a"))




fasd


records = []
columns = []
for tr in table.findAll("tr"):
    ths = tr.findAll("th")
    if ths != []:
        for each in ths:
            columns.append(each.text)
    else:
        trs = tr.findAll("td")
        record = []
        for each in trs:
            try:
                link = each.find('a')['href']
                text = each.text
                record.append(link)
                record.append(text)
            except:
                text = each.text
                record.append(text)
        records.append(record)

columns.insert(1, 'Link')
df = pd.DataFrame(data=records, columns = columns)
















