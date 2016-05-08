import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns


f = open(".\movies.txt", "a")
#==============================================
#==============================================
# Script for question 4a
# .\ is shorthand for "this directory" where the movies.py file is, ..\ means "the parent folder" based on the movies.py file
df_data = pd.read_table(".\data\u.data", sep="\t", names=["user id", "movie id", "rating", "timestamp"])
Number_of_unique_movies = (str) (len(df_data.drop_duplicates(subset="movie id")))
Number_of_unique_users = (str) (len(df_data.drop_duplicates(subset="user id")))

df_movie = pd.read_table(".\data\u.item", sep="\t", names=["movie id", "movie title", "release date", "IMDb URL", "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])

df_user = pd.read_table(".\data\u.user", sep="\|", names=["user id", "age", "gender", "occupation", "zip code"], engine="python")

# Write Number_of_unique_movies to txt file
f.write(Number_of_unique_movies + "\n")

# Write Number_of_unique_users to txt file
f.write(Number_of_unique_users + "\n")
f.write("\n")
f.close()
# Create
f = open("./movies.txt", "a")
#==============================================
#==============================================
# Script for question 4b

# Selection of "movie id" and "movie titile"
mi_t = df_movie[["movie id", "movie title"]]

# Selection of "movie id" and "rating"
mi_r = df_data[["movie id", "rating"]]

# Join mi_r and mi_t on key "movie id" and select "movie title" and "rating"
mt_r_join = mi_r.merge(mi_t, on="movie id")[["movie title", "rating"]]

# Group by "movie title" and sum up the number of "rating"
mt_r_count = mt_r_join[["movie title", "rating"]].groupby("movie title").count(axis="rating").sort("rating", ascending=0).reset_index()

# Limit result from the top 5
mt_r_count_5 = mt_r_count[:5]

# Write mt_r_count_5 to txt file
mt_r_count_5.to_csv(r'movies.txt', header=None, index=None, sep='\t', mode='a')
f.write("\n")
f.close()

f = open("./movies.txt", "a")
#==============================================
#==============================================
# Script for question 4c

# Selection of movie titles whose number of "rating" are greater and equal to 100
mt_r_100 = mt_r_count.ix[mt_r_count["rating"] >= 100, :]
# print mt_r_100

# Selection of "user id" and "age"
ui_a = df_user[["user id", "age"]]

# Selection of "user id", "movie id" and "rating"
ui_mi_r = df_data[["user id", "movie id", "rating"]]

# Join ui_mi_r, mi_t and ui_a on key "movie id" and select "movie title" and "age"
mt_a = ui_mi_r.merge(mi_t, on="movie id")[["movie title", "user id"]].merge(ui_a, on="user id")[["movie title", "age"]]

# Find the average ages of user and sort them ascendingly.
mt_a_avg = mt_a.groupby("movie title").mean().sort("age", ascending=1).reset_index()

# Join(merge) mt_r_100 and mt_a_avg on key "movie title" and select "movie title", "rating" and "age"
mt_r_a = mt_r_100.merge(mt_a_avg, on="movie title")[["movie title", "rating", "age"]].sort("age", ascending=1)[:5]

# Write mt_r_a to txt file
mt_r_a.to_csv(r'movies.txt', header=False, index=None, sep='\t', mode='a')
f.close()

#==============================================
#==============================================
# Script for question 4d

# Join(merge) mt_r_100 and mt_a_avg on key "movie title" and select "movie title", "rating" and "age"
r_a_merge = df_user.merge(df_data, on="user id")[["rating", "age"]]
r_avg_a = r_a_merge.groupby("age").mean().reset_index()
# print r_avg_a

sns.jointplot(x="age", y="rating", data=r_avg_a, color="g")
sns.plt.show()

#==============================================
#==============================================
# Script for question 4e
g = sns.jointplot(x="age", y="rating", data=r_avg_a, kind="kde", color="g")
g.plot_joint(sns.plt.scatter)
sns.plt.show()