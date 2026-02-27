
#%% 
# First you will see the initial code I used to get the clustering.
# Then I will give examples of players that are not good choices, good coices, and some that could work
# At the bottom is my write up.

#%% 
#Code from Lab-Clustering.ipynb
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
salary = pd.read_csv("2025_salaries.csv", header=1, encoding='latin-1')
stats = pd.read_csv("nba_2025.txt", sep=",", encoding='latin-1')
merged_data = pd.merge(salary, stats, on='Player')
duplicates = merged_data[merged_data.duplicated(subset='Player', keep=False)]
merged_data = merged_data.drop_duplicates(subset='Player', keep='first')
merged_data = merged_data.dropna(subset=['2025-26'])
merged_data = merged_data.drop(columns=['3P%', '2P%'])
merged_data['Awards'] = merged_data['Awards'].fillna(0)
merged_data = merged_data.drop(columns=['Rk', 'Team', 'Pos', 'Age', 'GS', 'MP', 'Player-additional', 'Tm'])
merged_data = merged_data.drop(columns=['FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB',  'DRB', 'TOV', 'PF'])
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = merged_data[["PTS", "TRB"]]
scaled_features = scaler.fit_transform(features)
mymodel = KMeans(n_clusters=3, random_state=42 )
mymodel.fit(scaled_features)
predictions = mymodel.predict(scaled_features)
mymodel.score(scaled_features)
merged_data['Cluster'] = predictions



markers = {0: "o", 1: "s", 2: "^", 3: "D", 4: "P"}
color_map = {'Low': 'blue', 'Medium': 'orange', 'High': 'red'}

merged_data['2025-26'].dtypes
merged_data['2025-26'] = pd.to_numeric(
merged_data['2025-26'].astype(str).str.strip().str.replace('[$,]', '', regex=True), errors='coerce')
merged_data['Salary_Grouping'] = merged_data['2025-26'].apply(
    lambda x: 'Low' if x < 3000000 else ('Medium' if x < 6500000 else 'High')
)

plt.figure(figsize=(10, 6))
for cluster_id, grp in merged_data.groupby("Cluster"):
    plt.scatter(grp['FG%'], grp['PTS'],
                marker=markers[cluster_id],
                c=grp['Salary_Grouping'].map(color_map),
                label=f"Cluster {cluster_id}",
                alpha=0.6, edgecolors="white")

plt.xlabel('FG%')
plt.ylabel('PTS')
plt.title('Clusters of NBA Players based on FG% and PTS\n(shape = cluster, color = salary)')
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

#%%
# Based on the visualization you can see that players with high points and a solid field goal percentage tend to be in clusters 1 or 2.


# %%
# Subset players

# So based on my model I can see that cluster 1 and 2 result in better players than cluster 0. 
# I want to find players with medium or low salaries so I find out who these players are by subsetting them.

# Good Choice — high performing clusters (1 and 2) with low or medium salary
# These are underpaid good players that we want to recruit. 
# group by being in cluster 1 or 2 and having salary be medium. sort by PTS with ascending = False so that the people with the most points appear first
good_choices_medium = merged_data[(merged_data['Cluster'].isin([1, 2])) & (merged_data['Salary_Grouping'].isin(['Medium']))].sort_values('PTS', ascending=False)
# do the same thing just change it so that their salary is considered low
good_choices_low = merged_data[(merged_data['Cluster'].isin([1, 2])) & (merged_data['Salary_Grouping'].isin(['Low']))].sort_values('PTS', ascending=False)


# Supporting visualization
print("GOOD CHOICES")
# this will print the players who fit the criteria above and list their name, points, total rebounts, field goal percentage, what cluster they are in, and what salary group they are in.
# use head(4) to show the first 4 entries, convert everything to a string though so it will appear cleanly and do index=False so i do not get 0 1 2 etc next to players name
print(good_choices_medium[['Player', 'PTS', 'TRB', 'FG%', '2025-26', 'Cluster', 'Salary_Grouping']].head(4).to_string(index=False))
print(good_choices_low[['Player', 'PTS', 'TRB', 'FG%', '2025-26', 'Cluster', 'Salary_Grouping']].head(4).to_string(index=False))


# Not Good Choices — low performing cluster with low salary
# These are cheap bad players we should avoid
# sort by ascending = True so that people with the lowest points appear first
# group by being in cluster 0 and having salary be low. sort by PTS with ascending = True so that the people with the least points appear first
not_good_choices = merged_data[(merged_data['Cluster'] == 0) & (merged_data['Salary_Grouping'] == 'Low')].sort_values('PTS', ascending=True)

# Supporting visualization
print("\nNOT GOOD CHOICES")
# this will print the players who fit the criteria above and list their name, points, total rebounts, field goal percentage, what cluster they are in, and what salary group they are in.
# use head(4) to show the first 4 entries, convert everything to a string though so it will appear cleanly and do index=False so i do not get 0 1 2 etc next to players name
print(not_good_choices[['Player', 'PTS', 'TRB', 'FG%', '2025-26', 'Cluster', 'Salary_Grouping']].head(4).to_string(index=False))


# Ok Choices — high performing cluster (1) with high salary but the cheapest of that group
# These could work if we can't get our good choices, they are expensive though but very good
# For this I will have to sort salary by ascending so I get the lowest salary.
# group by being in cluster 1 and having salary be high. sort by the salary column with ascending = True so that the people with the lowest salary out of being classified as high appear first
ok_choices = merged_data[(merged_data['Cluster'] == 1) & (merged_data['Salary_Grouping'] == 'High')].sort_values('2025-26', ascending=True)  

# Supporting visualization
print("\n OK CHOICES")
# this will print the players who fit the criteria above and list their name, points, total rebounts, field goal percentage, what cluster they are in, and what salary group they are in.
# use head(4) to show the first 4 entries, convert everything to a string though so it will appear cleanly and do index=False so i do not get 0 1 2 etc next to players name
print(ok_choices[['Player', 'PTS', 'TRB', 'FG%', '2025-26', 'Cluster', 'Salary_Grouping']].head(4).to_string(index=False))


#%% 
#Write up

# The best basketball players will be in Cluster 1, then really solid players will be in Cluster 2, and bad players will be in Cluster 0. This is becuase players are sorted by PTS and FG%.
# Then, each player is color coded according to their salary, blue is low, medium is orange, and red is high.
# I found players that were in cluster 1 or 2 since they are good players but with medium or low salaries. This is because we do not want really expeisive players as they are usually the top players. We also do not want really cheap players as they are usually not very good.
# Ideally, we want to find players that have good stats and medium salary,

# 3 players that I think are good choices are Russell Westbrook, Keyonte George, and Tim Hardaway Jr. 
# These players have high points and feild goal percentages which contribute greatly to them being good players.
# They also have accessable salarys of $4278960, $2296274, and $2296274 all which we can finance to have them join our team.
# They also have solid stats with good points, TBD, and FG%. 
# Russell Westbrook has 796 points, 296 total rebounds, and a 0.43% field goal percentage. His salary is grouped as low. All this combine make him a solid choice.
# Keyonte George has 1141 points, 187 total rebounds, and a 0.45% field goal percentage. His salary is grouped as medium, however all of his stats combine make him a solid pick.
# Tim Hardaway Jr. has 763 points, 142 total rebounds, and a 0.45% field goal percentage.  His salary is grouped as low. All this combine make him a solid choice.

# 3 players that are not good choices are N-Faly Dante, Garrett Temple, and Isaac Jones. These players have very low points, and are in the lowest cluster group which means that they are the worst in terms of basketball skills. 
# N-Faly Dante has 3 points, 7 total rebounds, and a 0.25% field goal percentage. His salary is grouped as low. All this combine make him not a good choice.
# Garrett Temple has 4 points, 4 total rebounds, and a 0.14% field goal percentage. His salary is grouped as low. All this combine make him not a good choice.
# Isaac Jones has 5 points, 2 total rebounds, and a 0.5% field goal percentage. His salary is grouped as low. All this combine make him not a good choice.
# These players also have low salaries so we should not choose them because they are cheap for a reaon, they are not good at basketball.

# 3 players that are ok choices are Donovan Clingan, Payton Pritchard, and Jeremiah Fears. These players have good stats but still high salaries. However, these are the lowest paid out of the high salary group.
# Donovan Clingan has 604 points, 597 total rebounds, and a 0.52% field goal percentage. His salary is grouped as high though. All this combine make him an ok choice.
# Payton Pritchard has 913 points, 221 total rebounds, and a 0.46% field goal percentage. His salary is grouped as high though. All this combine make him an ok choice.
# Jeremiah Fears has 741 points, 206 total rebounds, and a 0.42% field goal percentage. His salary is grouped as high though. All this combine make him an ok choice.

# Overall this breakdown allows us to find good athletes who are not the most expensive so we can recruit them.


# %%
