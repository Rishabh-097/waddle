# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:03:28 2019

@author: dell
"""

from bs4 import BeautifulSoup
import requests
import os

os.chdir('E:/ML/Data/fifa19')

base_url="https://www.fifaindex.com/players/fifa19_299/"

#source = requests.get(base_url).text
#soup = BeautifulSoup(source)
#body = soup.find_all('tbody')
#players = str(soup.find_all('tr')[29].findAll('td')[3]).split(">")[2].split("<")[0]


f=open("names.csv","w",encoding='utf-8')
headers="ID,Full Name\n"
f.write(headers)

for off in range(1,610):
    url = base_url + str(off)
    source = requests.get(url).text
    soup = BeautifulSoup(source)
    players = soup.find_all('tr')
    for player in players[1:]:
        id = str(player).split(">")[0].split('"')[1]
        name = str(player.findAll('td')[3]).split(">")[2].split("<")[0]
        det = (id + "," + name + "\n")
        f.write(det)
        
f.close()

