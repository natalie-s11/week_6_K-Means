
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




# %%
# Subset players

# So based on my model I can see that cluster 1 and 2 result in better players than cluster 0. 
# I want to find players with medium or low salaries so I find out who these players are by subsetting them.

# Good Choice — high performing clusters (1 and 2) with low or medium salary
# These are underpaid good players that we want to recruit. 
good_choices = merged_data[
    (merged_data['Cluster'].isin([1, 2])) & 
    (merged_data['Salary_Grouping'].isin(['Low', 'Medium']))
].sort_values('PTS', ascending=False)

# Supporting visualization
print("GOOD CHOICES")
print(good_choices[['Player', 'PTS', 'TRB', 'FG%', '2025-26', 'Cluster', 'Salary_Grouping']].head(4).to_string(index=False))


# Not Good Choices — low performing cluster with low salary
# These are cheap bad players we should avoid
not_good_choices = merged_data[
    (merged_data['Cluster'] == 0) & 
    (merged_data['Salary_Grouping'] == 'Low')
].sort_values('PTS', ascending=False)

# Supporting visualization
print("\nNOT GOOD CHOICES")
print(not_good_choices[['Player', 'PTS', 'TRB', 'FG%', '2025-26', 'Cluster', 'Salary_Grouping']].head(4).to_string(index=False))


# Ok Choices — high performing cluster (1) with high salary but the cheapest of that group
# These could work if we can't get our good choices, they are expensive though but very good
# For this I will have to sort salary by ascending so I get the lowest salary.
ok_choices = merged_data[
    (merged_data['Cluster'] == 1) & 
    (merged_data['Salary_Grouping'] == 'High')
].sort_values('2025-26', ascending=True)  

# Supporting visualization
print("\n OK CHOICES")
print(ok_choices[['Player', 'PTS', 'TRB', 'FG%', '2025-26', 'Cluster', 'Salary_Grouping']].head(4).to_string(index=False))


#%% 
#Write up the results in a separate notebook with supporting visualizations and  an overview of how and why you made the choices you did. This should be at least  500 words and should be written for a non-technical audience.

