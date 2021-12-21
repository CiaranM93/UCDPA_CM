import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#real-world scenario
#webscraping and importing a csv
club_data = pd.read_csv(r"C:\Users\Ciara\PycharmProjects\pythonProject1\club.csv")
club_data.head()

players_data = pd.read_csv(r"C:\Users\Ciara\PycharmProjects\pythonProject1\players.csv")
players_data.head()



#Removing blank column data
club_data = club_data.drop(columns = ['Unnamed: 0'])

#Loading csv Data information
print('\n')
club_data.info()
print('\n')
players_data.info()

print('\n')

#merging two .csv files and printing all columns
merged = pd.concat([club_data, players_data], axis=1)
print(merged.columns.tolist())

print('\n')

#Checking for null values in merged data
merged.shape
print(merged.isnull().sum())


#Showing the number of teams in each legue
league_chart = sns.countplot(x='Competition Name', data = merged)
plt.xlabel('Competition Name')
plt.title('Count of Teams per League')
plt.show()
plt.close()

#Showing the number of players per country
country_chart = sns.countplot(x='Country', data = merged)
plt.xlabel('Country')
plt.title('Count of Players per Country')
plt.show()
plt.close()

#Top10 Clubs by Market Value
dataset1 = club_data.loc[0:10, :]
plt.figure(figsize=(10,6))
sns.barplot(x= 'Club Name', y = 'Market Value Of Club In Millions(£)', data = club_data)
plt.xticks(rotation = 90)
plt.title('Top10 Clubs by Market Value')
plt.show()

#Average Age of Player by Position
PlayersAge = players_data.groupby(['Position']).agg({'Age':'mean'})
plt.figure(figsize = (10,6))
sns.barplot(players_data['Position'], players_data['Age'], ci = None)
plt.xticks(rotation = 90)
plt.title('Average Age of Player by Position')
plt.show()


