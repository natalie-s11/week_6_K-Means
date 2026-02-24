#%%
# Imports
import pandas as pd
import numpy as np
import sklearn as sk

# %%
# Load the data and view the first few rows
salary = pd.read_csv("2025_salaries.csv", header=1, encoding='latin-1')
salary.head()

stats = pd.read_csv("nba_2025.txt", sep=",", encoding='latin-1')
stats.head()


# %%
# Merge
#help(pd.merge)
merged_data = pd.merge(salary, stats, on='Player')
merged_data.head()

# %%
# Duplicates
duplicates = merged_data[merged_data.duplicated(subset='Player', keep=False)]
print(duplicates)

# Salary is not a feature, put is as the color 
# Lower left = lower performance, upper right = higher performance
# Need to choose variables in data frame to put on x and y axis, can be anything
# Should best display the clusters with a scatterplot, need to explore first
# Use unique values in dataset to cluster withm 
# Want people that are good but not the most expensive 
# Find people with high performance but have low salaries = goal
# points, minutes played, rebounds etc are all good, experiment 
# Work well becuase high distribution, want variables with high variance to tell good from bad


# %%
# Sklearn
mymodel = KMeans(n_clusters=3)
mymodel.fit(merged_data[["Salary", "Points"]])
predictions = mymodel.predict(merged_data[["Salary", "Points"]])
mymodel.score(merged_data[["Salary", "Points"]])

mymodel = KMeans(n_clusters=3)
mymodel.fit(X) #Test data
predictions = mymodel.predict(X) #Get prediction
mymodel.score(X) #Get evaluation 

#%%
# Lambda Functions 
merged_data['Salary_in_Thousands'] = merged_data['Salary'].apply(lambda x: x / 1000)
merged_data['High_Salary'] = merged_data['Salary'].apply(lambda x: True if x > 100000 else False)






#%%
# Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(merged_data[["Salary", "Points"]])
    wcss.append(kmeans.inertia_)
    import matplotlib.pyplot as plt
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()



