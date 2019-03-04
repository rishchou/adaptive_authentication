import numpy as np
from sklearn.cluster import MeanShift, KMeans
from sklearn import preprocessing, model_selection as cross_validation
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
from sklearn import preprocessing,model_selection as cross_validation, neighbors


# https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
df = pd.read_excel('input1.xlsx')

pd.options.mode.chained_assignment = None  # default='warn'
#print(df)
original_df = pd.DataFrame.copy(df)
nr_df = df[['Source','Destination', 'NAS-IP-Address','Vendor-ID']]


# #original_df = original_df.filter(items=['Source', 'Destination', 'NAS-IP-Address', 'User-Name'])
# #network_df = original_df[['Source', 'Destination', 'NAS-IP-Address', 'User-Name']]
# print(original_df)
# df.drop(['body','name'], 1, inplace=True)
df.fillna(0,inplace=True)

def handle_non_numerical_data(df):

    # handling non-numerical data: must convert.
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        #print(column,df[column].dtype)
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:

            column_contents = df[column].values.tolist()
            #finding just the uniques
            unique_elements = set(column_contents)
            # great, found them.
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    # creating dict that contains new
                    # id per unique string
                    text_digit_vals[unique] = x
                    x+=1
            # now we map the new "id" vlaue
            # to replace the string.
            df[column] = list(map(convert_to_int,df[column]))

    return df

nr_df = handle_non_numerical_data(nr_df)
# df.drop(['ticket','home.dest'], 1, inplace=True)

# X = np.array(df.drop(['survived'], 1).astype(float))
X = np.array(nr_df.astype(float))
X = preprocessing.scale(X)
# y = np.array(df['survived'])

# clf = MeanShift()
clf = KMeans(n_clusters=3)
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

nr_df = df[['Source','Destination', 'NAS-IP-Address','Vendor-ID']]
nr_df['Network reputation']=np.nan

for i in range(len(X)):
    nr_df['Network reputation'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))
# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k, col in zip(range(n_clusters_), colors):
#     my_members = labels == k
#     cluster_center = cluster_centers[k]
#     plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
#     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)
# plt.title('Estimated number of clusters for NR: %d' % n_clusters_)
# plt.show()
colors = 10*['r','g','b','c','k','y','m']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')

ax.scatter(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],
            marker="x",color='k', s=150, linewidths = 5, zorder=10)

plt.show()
# survival_rates = {}
# for i in range(n_clusters_):
#     temp_df = original_df[ (original_df['cluster_group']==float(i)) ]
#     #print(temp_df.head())
#
#     survival_cluster = temp_df[  (temp_df['survived'] == 1) ]
#
#     survival_rate = len(survival_cluster) / len(temp_df)
#     #print(i,survival_rate)
#     survival_rates[i] = survival_rate
#
# print(survival_rates)
print(" number of clusters for NR: " ,n_clusters_)

nr_df.to_csv('nr.csv', index = False)

print(nr_df[ (nr_df['Network reputation']==1) ])
print(nr_df[ (nr_df['Network reputation']==0) ].describe())


geo_df = original_df[['Location','Timezone','Timestamp','Callback-Number']]

geo_df = handle_non_numerical_data(geo_df)
# df.drop(['ticket','home.dest'], 1, inplace=True)

# X = np.array(df.drop(['survived'], 1).astype(float))
X = np.array(geo_df.astype(float))
X = preprocessing.scale(X)
# y = np.array(df['survived'])

# clf = MeanShift()
clf = KMeans(n_clusters=3)
clf.fit(X)


labels = clf.labels_
cluster_centers = clf.cluster_centers_
geo_df = original_df[['Location','Timezone','Timestamp','Callback-Number']]

geo_df['Geo-location authenticity'] = np.nan

for i in range(len(X)):
    geo_df['Geo-location authenticity'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))
# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k, col in zip(range(n_clusters_), colors):
#     my_members = labels == k
#     cluster_center = cluster_centers[k]
#     plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
#     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)
# plt.title('Estimated number of clusters for GL: %d' % n_clusters_)
# plt.show()
colors = 10*['r','g','b','c','k','y','m']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')

ax.scatter(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],
            marker="x",color='k', s=150, linewidths = 5, zorder=10)

plt.show()

print(" number of clusters for GL: " ,n_clusters_)

geo_df.to_csv('geo.csv',index = False)


dp_df = original_df[['Device-MAC','Device-Type','Authen-method','Vendor-ID','Software-Version','Service-Type','User-Name','Protocol']]

dp_df = handle_non_numerical_data(dp_df)
# df.drop(['ticket','home.dest'], 1, inplace=True)

# X = np.array(df.drop(['survived'], 1).astype(float))
X = np.array(dp_df.astype(float))
X = preprocessing.scale(X)
# y = np.array(df['survived'])

# clf = MeanShift()
clf = KMeans(n_clusters=3)
clf.fit(X)


labels = clf.labels_
cluster_centers = clf.cluster_centers_
dp_df = original_df[['Device-MAC','Device-Type','Authen-method','Vendor-ID','Software-Version','Service-Type','User-Name','Protocol']]

dp_df['Device fingerprinting'] = np.nan

for i in range(len(X)):
    dp_df['Device fingerprinting'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))
# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k, col in zip(range(n_clusters_), colors):
#     my_members = labels == k
#     cluster_center = cluster_centers[k]
#     plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
#     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)
# plt.title('Estimated number of clusters for DP: %d' % n_clusters_)
# plt.show()
colors = 10*['r','g','b','c','k','y','m']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')

ax.scatter(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],
            marker="x",color='k', s=150, linewidths = 5, zorder=10)

plt.show()

print(" number of clusters for DP: " ,n_clusters_)

dp_df.to_csv('dp.csv',index = False)

ta_df = original_df[['Location','Timezone','Timestamp']]

ta_df = handle_non_numerical_data(ta_df)
# df.drop(['ticket','home.dest'], 1, inplace=True)

# X = np.array(df.drop(['survived'], 1).astype(float))
X = np.array(ta_df.astype(float))
X = preprocessing.scale(X)
# y = np.array(df['survived'])

# clf = MeanShift()
clf = KMeans(n_clusters=3)
clf.fit(X)


labels = clf.labels_
cluster_centers = clf.cluster_centers_
ta_df = original_df[['Location','Timezone','Timestamp','Callback-Number']]

ta_df['Time Anamolies'] = np.nan

for i in range(len(X)):
    ta_df['Time Anamolies'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))
# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k, col in zip(range(n_clusters_), colors):
#     my_members = labels == k
#     cluster_center = cluster_centers[k]
#     plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
#     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)
# plt.title('Estimated number of clusters for GL: %d' % n_clusters_)
# plt.show()
colors = 10*['r','g','b','c','k','y','m']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')

ax.scatter(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],
            marker="x",color='k', s=150, linewidths = 5, zorder=10)

plt.show()

print(" number of clusters for TA: " ,n_clusters_)

ta_df.to_csv('ta.csv',index = False)











# df = pd.read_csv('nr')

# X = np.array(df.drop(['Network reputation'], 1))
# y = np.array(df['Network reputation'])

# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# clf = neighbors.KNeighborsClassifier()
# clf.fit(X_train, y_train)

# accuracy = clf.score(X_test, y_test)
# print("Accuracy",accuracy)

# example_df = X_test[2:3]
# example_df = handle_non_numerical_data(example_df)
# X = np.array(example_df.astype(float))
# X = preprocessing.scale(X)
# #example_measures = np.array([4,2,1,1,1,2,3,2,1]).reshape(1, -1)
# prediction = clf.predict(X)
# print("Predicted Network reputation for new user:",prediction)




