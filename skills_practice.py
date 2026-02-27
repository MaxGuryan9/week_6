# %%
import pandas as pd
import numpy as np
import sklearn as sk

# %%
# loading data 
salary_data = pd.read_csv("2025_salaries.csv" , header=1)

stats = pd.read_csv("nba_2025.txt" , sep = "," , encoding="latin-1")

# %%
salary_data.head()


# %%
merged_data = pd.merge(salary_data, stats, on = "Player")
# %%
duplicates = merged_data[merged_data.duplicated(subset="Player", keep=False)]

# %%
# 1. create an instance of the model example: mymodel = KMeans(n_clusters = 3)
# 2. fit the model to the data example: mymodel.fit(X)
# 3. make predictions using the model example: predictions = mymodel.predict(X)
# 4. evaluate the model's performance example: performance = mymodel.score(X)

# for kmeans you don't need to predict, you can just use the labels_ attribute
# to get the clust assignments for each data point after fitting the model

# %%
merged_data["Salary_in_thousands"] = merged_data["Salary"].apply(lambda x: x/1000)

merged_data["High_Salary"] = merged_data["Salary_in_thousands"].apply(lambda x: True if x > 1000000 else False)