



from os.path import expanduser
home = expanduser("~")


import pandas as pd

import numpy as np

import pickle



import plotly.graph_objects as go
from plotly.subplots import make_subplots


data_list = pickle.load( open( home+"/Downloads/rona_data.p", "rb" ) )
print (len(data_list))
data = data_list[0]
data_cases = data_list[1]
data_tests = data_list[2]

print ()

fig = make_subplots(rows=1, cols=1)
for key, val in data.items():

    n = len(list(range(6,19)))
    if len(val) < n:
        continue

    # print (val, key)
    # fsdfa
    fig.add_trace(
                go.Scatter(x=list(range(6,19)), y=val, mode='lines', name=key,),
                row=1, col=1, 
                )

    # fig.add_trace(
    #             go.Scatter(x=list(range(6,19)), y=data_cases[key], mode='lines', name=key,),
    #             row=2, col=1, 
    #             )

    # fig.add_trace(
    #             go.Scatter(x=list(range(6,19)), y=data_tests[key], mode='lines', name=key,),
    #             row=3, col=1, 
    #             )


fig.show()

fig.write_html(home +"/Downloads/plotly_rona.html")
print ('saved ',"/Downloads/plotly_rona.html" )






