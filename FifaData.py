#IMPORTS
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import eli5
from eli5.sklearn import PermutationImportance
from collections import Counter
import missingno as msno

import warnings
warnings.filterwarnings('ignore')
import plotly
sns.set_style('darkgrid')

df=pd.read_csv(r'C:\Users\Ciara\PycharmProjects\pythonProject1\data.csv')
print('\n')
print('\n')
print('_____________________________________\n\n')
df.head()
print('\n')
print('\n')
print('_____________________________________\n\n')
df.columns
print('\n')
print('\n')
print('_____________________________________\n\n')
df.info()
print('\n')
print('\n')
print('_____________________________________\n\n')
df.describe()
print('\n')
print('\n')
print('_____________________________________\n\n')
df.drop(['Unnamed: 0','Photo','Flag','Club Logo'],axis=1,inplace=True)
print('\n')
print('\n')
print('_____________________________________\n\n')
df.isnull().sum()
print('\n')
print('\n')
print('_____________________________________\n\n')
missing_height = df[df['Height'].isnull()].index.tolist()
missing_weight = df[df['Weight'].isnull()].index.tolist()
if missing_height == missing_weight:
    print('SAME')
else:
    print('DIFFERENT')

df.drop(df.index[missing_height], inplace=True)
df.isnull().sum()
df.drop(['Loaned From','Release Clause','Joined'],axis=1,inplace=True)
print('\n')
print('\n')
print('_____________________________________\n\n')
print('TOTAL NUMBER OF COUNTRIES WITHIN DATASET: {0}'.format(df['Nationality'].nunique()))
print(df['Nationality'].value_counts().head(5))
print('--'*40)
print('_____________________________________\n\n')
print('TOTAL NUMBER OF CLUBS WITHIN DATASET: {0}'.format(df['Club'].nunique()))
print(df['Club'].value_counts().head(5))
print('\n')
print('\n')
print('_____________________________________\n\n')
print('PLAYER WITH THE MOST POTENTIAL: '+str(df.loc[df['Potential'].idxmax()][1]))
print('PLAYER WITH BEST OVERALL PREFORMANCE : '+str(df.loc[df['Overall'].idxmax()][1]))
print('\n')
print('\n')
print('_____________________________________\n\n')
pr_cols=['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']

print('\n')
print('\n')
print('BEST PLAYER PER STAT:')
print('_____________________________________\n\n')
i=0
while i < len(pr_cols):
    print('Best {0} : {1}'.format(pr_cols[i],df.loc[df[pr_cols[i]].idxmax()][1]))
    i += 1

def value_to_int(df_value):
    try:
        value = float(df_value[1:-1])
        suffix = df_value[-1:]

        if suffix == 'M':
            value = value * 1000000
        elif suffix == 'K':
            value = value * 1000
    except ValueError:
        value = 0
    return value

df['Value'] = df['Value'].apply(value_to_int)
df['Wage'] = df['Wage'].apply(value_to_int)

sns.jointplot(x=df['Age'],y=df['Potential'],
              joint_kws={'alpha':0.1,'s':5,'color':'red'},
              marginal_kws={'color':'red'})

player_features = (
    'Acceleration', 'Aggression', 'Agility',
    'Balance', 'BallControl', 'Composure',
    'Crossing', 'Dribbling', 'FKAccuracy',
    'Finishing', 'GKDiving', 'GKHandling',
    'GKKicking', 'GKPositioning', 'GKReflexes',
    'HeadingAccuracy', 'Interceptions', 'Jumping',
    'LongPassing', 'LongShots', 'Marking', 'Penalties')

from math import pi

idx = 1
plt.figure(figsize=(15, 45))
for position_name, features in df.groupby(df['Position'])[player_features].mean().iterrows():
    top_features = dict(features.nlargest(5))


    categories = top_features.keys()
    N = len(categories)

    values = list(top_features.values())
    values += values[:1]


    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax = plt.subplot(10, 3, idx, polar=True)


    plt.xticks(angles[:-1], categories, color='grey', size=8)

    ax.set_rlabel_position(0)
    plt.yticks([25, 50, 75], ["25", "50", "75"], color="grey", size=7)
    plt.ylim(0, 100)

    plt.subplots_adjust(hspace=0.5)

    ax.plot(angles, values, linewidth=1, linestyle='solid')

    ax.fill(angles, values, 'b', alpha=0.1)
    plt.title(position_name, size=11, y=1.1)
    idx += 1

sns.lmplot(x='BallControl', y='Dribbling', data=df, col='Preferred Foot',
           scatter_kws={'alpha': 0.1, 'color': 'orange'},
           line_kws={'color': 'red'})

plt.show()
plt.close()

