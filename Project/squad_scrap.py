# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 03:28:58 2019

@author: dell
"""

from bs4 import BeautifulSoup
import requests
import os

os.chdir('E:/ML/Data/International')

myurl="https://en.wikipedia.org/wiki/2019_Copa_América_squads"

source = requests.get(myurl).text
soup = BeautifulSoup(source)

#print(soup.prettify())

countries= soup.find_all('h3')
squads= soup.find_all('table')

"""
country = countries[0].findAll('span',{'class':'mw-headline'})[0].text
squad = squads[0].findAll('tr')
pos = squad[1].findAll('a')[0].text
name = squad[1].findAll('a')[1].text
"""

f=open("squads.csv","w")
headers="group,country,player,position\n"
f.write(headers)

for i in range(12):
    country = countries[i].findAll('span',{'class':'mw-headline'})[0].text
    squad = squads[i].findAll('tr')
    if i<4:
        group = 'A'
    elif i<8:
        group = 'B'
    else:
        group = 'C'
    for j in range(1,len(squad)):
        data = squad[j].findAll('a')
        pos = data[0].text
        name = data[1].text            
        det= (group + ',' + country + "," + name + "," + pos + "\n")
        f.write(det)
               
f.close()