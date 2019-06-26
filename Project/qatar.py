# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 02:13:25 2019

@author: dell
"""

from bs4 import BeautifulSoup
import requests
import os

os.chdir('E:/ML/Project')

url="http://pesdb.net/pes2019/?sort=nationality&page=110"

source = requests.get(url).text
soup = BeautifulSoup(source)
players = soup.findAll('tr')[1:33]
#players[1].findAll('td')[4].text

f=open("qatar.csv","w",encoding='utf-8')
headers="Full Name,Age,Height,Nationality,Overall \n"
f.write(headers)

for player in players:
    data = player.findAll('td')
    name = data[1].text
    age = data[6].text
    height = data[4].text
    nation = data[3].text
    overall = data[8].text
    if(nation=='QATAR'):
        det = (name + ',' + age + ',' + height + ',' + nation + ',' + overall + "\n")
        f.write(det)
    
f.close()


import pandas as pd
import numpy as np
    
q = pd.read_csv('qatar.csv')
q.columns = ['Full Name', 'Age', 'Height', 'Nationality', 'Overall']
q['Nationality'] = 'Qatar'
q['Potential'] = 0
q['Potential'] = q['Potential'].apply(lambda x: x + np.random.randint(5,8)) + q['Overall']
squads_names = q['Full Name'].str.split(" ", n=1, expand=True)
q['First Name'] = squads_names[0]
q['Last Name'] = squads_names[1]
q['Name'] = q['First Name'].apply(lambda x: x[0:1]) + ". " + q['Last Name']
q=q.fillna('#')
col = q.columns.tolist() 
col = col[-1:] + col[:-1]
q = q[col]
           
fifa = pd.read_csv('fifa_19.csv')
def get_inches(el):
    if el=='-':
        return
    f = int(el[0:1])*12
    i = int(el[2:])
    inch = f+i
    return(round(inch*2.54))
fifa = fifa[['Name','Full Name', 'Age', 'Height', 'Nationality', 'Overall','Potential','First Name','Last Name']]
#fifa['Height'] = np.where(fifa['Height']=='-',"5'10",fifa['Height'])
fifa['Height'] = fifa['Height'].apply(lambda x: get_inches(x))
fifa['Height'] = fifa.groupby('Nationality').transform(lambda x: x.fillna(round(x.mean())))['Height']
fifa = pd.concat([fifa,q]).drop_duplicates().reset_index(drop=True)
fifa = fifa.sort_values(['Overall','Potential','Full Name'],ascending=[False,False,True]).reset_index(drop=True)
fifa.to_csv('fifa19_cleaned.csv',encoding='utf-8',index=False)