# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 14:31:17 2019

@author: dell
"""

from bs4 import BeautifulSoup
import requests
import os

os.chdir('E:/ML/Data/International')

myurl="https://www.fifa.com/fifa-world-ranking/ranking-table/men/"

source = requests.get(myurl).text
soup = BeautifulSoup(source)

#print(soup.prettify())

containers= soup.find_all('tr')

#container = containers[1]

f=open("8.csv","w")
headers="rank_date,rank,country_full,total_points,previous_points,confederation\n"
f.write(headers)
date='14-06-2019'

for i in range(1,len(containers)):
    data = containers[i].findAll('span')
    rank = data[0].text
    name = data[2].text
    points = data[4].text
    ppoints = data[5].text
    confed = data[8].text.replace("#","")
    det= (date + ',' + rank + "," + name + "," + points + "," + ppoints + "," + confed + "\n")
    f.write(det)
               
f.close()