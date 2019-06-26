# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:29:06 2019

@author: RIshabh
"""

import pandas as pd
import numpy as np
df=pd.read_csv("spi.csv")
players=pd.read_csv("fifa19_cleaned.csv")
squads=pd.read_csv('squads_upd.csv')
stats=pd.read_csv('stats.csv')
spi=pd.merge(left=df,right=stats,how='left',left_on=['Name'],right_on=['Nationality']).dropna().drop(['Nationality'],axis=1)
team=pd.read_csv('history team.csv',delimiter='\t')
spi=pd.merge(left=spi,right=team,how='left',left_on=['Name'],right_on=['Team']).drop(['Team'],axis=1)
spi.to_csv('team.csv',encoding='utf-8')
team=pd.read_csv('team.csv')
fixtures=pd.read_csv('copa_fixtures.csv')
fixtures=pd.merge(left=fixtures,right=team,how='left',left_on=['Team'],right_on=['Name']).drop(['Name','Unnamed: 0'],axis=1).fillna(0)
fixtures.to_csv('fixtures.csv',encoding='utf-8')

fixtures["avg score"] = round(fixtures['GF'] / fixtures['GP'],2)
fixtures["avg conceded"] = round(fixtures['GA'] / fixtures['GP'],2)

#for i in ["SPI","Age","Height","Part.","Potential"]:
#    fixtures[i] = fixtures[i].apply(lambda x: (x - fixtures[i].mean())/fixtures[i].std())
features=fixtures.iloc[0:,[5,6,7,9,-2,-1]].fillna(0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features)
