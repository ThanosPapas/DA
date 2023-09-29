from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score, roc_auc_score
import keras_tuner
import numpy as np
import pandas as pd
import sklearn.cluster
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense, Dropout
from keras_tuner import HyperModel
from sklearn.feature_selection import VarianceThreshold,chi2, SelectKBest, f_classif
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN
from kneed import KneeLocator
from sklearn.svm import SVC
from sklearn.utils import class_weight


def binary_encoding(df):
    ''' encoding the binary features '''

    for feat in binary_feats:
        df[feat] = df[feat].str.strip('"').replace({'yes': 1, 'no': 0})
    return df

def dtype_conversion(df):
    '''epeidi ta numeric features exun dtype object, ta kanw na exun dtype int'''

    for i in numeric_feats:
        df[i] = df[i].astype(int)
    return df

def outlier_detection_and_removal(df):
    ''' xrisimopoiw tin z-method gia na vrw ta outliers. Arxika thetw ta oria kai meta vriskw tis times pou ksefevgun ap auta'''

    for i in numeric_feats:
        upper_limit = df[i].mean() + 3 * df[i].std()
        lower_limit = df[i].mean() - 3 * df[i].std()
        # new_df = df.loc[(df[i] < upper_limit) & (df[i] > lower_limit)][i]
        # print(len(new_df))

        # kanw capping sta outliers, diladi thetw tis times tus ises me to upper i to lower limit analogws. Epeleksa autin tin methodo anti gia trimming
        # epeidi ta outliers itan arketa se kathe feature. An xrisimopoiusa trimming tha xanontan megalos ogkos data
        df.loc[(df[i] > upper_limit), i] = upper_limit
        df.loc[(df[i] < lower_limit), i] = lower_limit
    return df

def variance_threshold(df):
    '''check for constant variables to be removed, with a threshold of 0.01'''

    columns_to_add_back = list(categorical_feats) + ['y']
    df_var_threshold = df.drop(columns='y').drop(columns=categorical_feats)

    var_thr = VarianceThreshold(threshold=0.01)
    var_thr.fit(df_var_threshold)

    concol = [column for column in df_var_threshold.columns
              if column not in df_var_threshold.columns[var_thr.get_support()]]
    df_var_threshold.drop(concol, axis=1)
    return  pd.concat([df_var_threshold, df[columns_to_add_back]], axis=1)

def correlation_threshold(df):
    '''will filter feature if correlation > 0.9'''

    df_corr_threshold = df[numeric_feats]
    corr = df_corr_threshold.corr()
    # sns.heatmap(corr, annot=True, cmap='coolwarm')
    # plt.show()

def chi_sq(df):
    '''Chi2 test for categorical input and categorical output'''

    df_chi = df.copy().drop(columns=numeric_feats, axis=1)
    label_encoder = LabelEncoder()
    for i in categorical_feats:
        df_chi[i] = label_encoder.fit_transform(df_chi[i])

    X = df_chi.drop(columns=['y'], axis=1)
    y = df_chi['y']
    chi_scores = chi2(X,y)

    for i in range(2):
        values = pd.Series(chi_scores[i], index=X.columns)
        values.sort_values(ascending=False, inplace=True)
        # values.plot.bar()
        # plt.show()
    return df.drop(columns=['default', 'marital', 'month', 'education', 'poutcome'], inplace=True)

def anova(df):
    '''Anova for numerical input and categorical output'''

    X = df[numeric_feats]
    print(X)
    y = df['y']
    fs = SelectKBest(score_func=f_classif, k='all')
    fs.fit(X, y)

    # fs_scores_df = pd.DataFrame({'Feature': numeric_feats, 'Score': fs.scores_})
    # fs_scores_df.sort_values('Score', ascending=False, inplace=True)
    # fs_scores_df.plot(x='Feature', y='Score', kind='bar')
    # plt.xlabel('Features')
    # plt.ylabel('Feature Selection Scores')
    # plt.title('Feature Selection Scores')
    # plt.tight_layout()
    # plt.show()

    # Vasei twn parapanw, epilegw ta features duration kai previous kai ws ta pio simantika (me fthinusa seira simantikotitas)
    return df.drop(columns=['age', 'balance', 'day', 'campaign', 'pdays'], inplace=True)

def visualize(df):
    '''Displays histograms and basic data points of the features'''

    print(df.describe())
    sns.pairplot(df[df.columns])
    plt.show()
    for i in df.columns:
        plt.hist(df[i], bins=20)
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title(i.upper())
        plt.show()
    exit()

df = pd.read_csv('bank-full.csv', delimiter=';')

numeric_feats = ['age','balance','day','duration','campaign','pdays','previous']
binary_feats = ['default','housing','loan', 'y']
categorical_feats = ['job','marital','education','contact','month','poutcome']

# check for NaN/null values
print("\n", print(df.isnull().sum()), "\n")

df = binary_encoding(df)

df = dtype_conversion(df)

df = outlier_detection_and_removal(df)

# scaling numerical data
rs = RobustScaler()
df[numeric_feats] = pd.DataFrame(rs.fit_transform(df[numeric_feats]), columns=numeric_feats)


# # #                                     FEATURE SELECTION                                                     # # #

df = variance_threshold(df)

correlation_threshold(df)

chi_sq(df)

anova(df)
# visualize(df)

subset_df = pd.DataFrame(df[['duration', 'previous']], columns=['duration', 'previous'])
# # #                                     CLUSTERING                                                     # # #

# Try different values of k to find the optimal number of clusters
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(subset_df)
    wcss.append(kmeans.inertia_)

kl = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")

# Plot the elbow curve
# plt.plot(range(1, 11), wcss)
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('WCSS')
# plt.title('Elbow Curve')
# plt.show()


kmeans = KMeans(n_clusters=kl.elbow, random_state=42)  # Set the number of clusters as desired
kmeans.fit(subset_df)
labels = kmeans.labels_

subset_df['Cluster'] = labels

# Visualize the clusters
# plt.scatter(subset_df['duration'], subset_df['previous'], c=subset_df['Cluster'])
# plt.xlabel('Duration')
# plt.ylabel('Previous')
# plt.title('K-means Clustering')
# plt.show()


df['Cluster'] = kmeans.labels_
print(kmeans.cluster_centers_)
print(df['Cluster'].value_counts())

print(df.columns)
print(df.groupby('Cluster')['contact'].describe())
print(df.groupby('Cluster')['job'].describe())
print(df.groupby('Cluster')['y'].describe())
print(df.groupby('Cluster')['duration'].describe())
print(df.groupby('Cluster')['previous'].describe())

df.drop(['Cluster', 'contact', 'job', 'loan'],axis=1,inplace=True)


subset_df.drop('Cluster', axis=1, inplace=True)
# labels = df['Cluster']
# # silhouette_avg = silhouette_score(X, labels)
# #
# # print("The silhouette score is:", silhouette_avg)

subset_df = subset_df.drop_duplicates()

dbscan = DBSCAN(eps = 0.5, min_samples = 4).fit(subset_df) # fitting the model
labels = dbscan.labels_ # getting the labels
print(set(labels))


subset_df['Cluster'] = labels
X = subset_df.drop('Cluster', axis=1)
silhouette_avg = silhouette_score(X, labels)

print("The silhouette score is:", silhouette_avg)

# plt.scatter(subset_df['duration'], subset_df['previous'], c=subset_df['Cluster'])
# plt.xlabel('Duration')
# plt.ylabel('Previous')
# plt.title('DBSCAN')
# plt.show()
# ohe = OneHotEncoder()
# df = pd.get_dummies(df, prefix=['job'], columns=['job'], drop_first=True,)

# # #                                     CLASSIFICATION                                                     # # #

y = pd.DataFrame(df['y'], columns=['y'])
x = df.drop(columns=['y'], axis=1)

x = x.values
y = y.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
class_weights = dict(zip([0, 1], [(1 / len(y_train[y_train == 0])), (1 / len(y_train[y_train == 1]))]))



classifier = SVC(kernel='linear', class_weight=class_weights)
classifier.fit(X_train, y_train)

# Predict on the testing set
y_pred = classifier.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("Accuracy:", accuracy)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_pred)

print("F1 Score:", f1)
print("ROC-AUC Score:", roc_auc)


# GaussianNB
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("ROC-AUC Score:", roc_auc)

sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

# Set labels, title, and ticks
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Display the plot
plt.show()


# # #                                     NN                                                     # # #

class MyHyperModel(HyperModel):
     def build(self, hp):
         model2 = Sequential()
         model2.add(Flatten())
         for i in range(hp.Int("num_layers", 2, 6)):
             model2.add(
                 Dense(
                     # Tune number of units separately.
                     units=hp.Int(f"units_{i}", min_value=10, max_value=200, step=4), #80 128
                     activation="linear",
                 )
             )
         if hp.Boolean("dropout"):
             model2.add(Dropout(rate=0.25))

         model2.add(Dense(1, activation='linear'))
         model2.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])


         return model2


tunerr = keras_tuner.Hyperband(
                        hypermodel=MyHyperModel(),
                        objective ='val_mean_squared_error',
                        max_epochs=50,
                         directory="my_dir",
                         project_name="helloworld",
                         overwrite=True,
)

stop_early = EarlyStopping(monitor='val_loss', patience=15)

# tunerr.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
# best_hps = tunerr.get_best_hyperparameters(num_trials=1)[0]
#
#
# best_hyperparameters = tunerr.get_best_hyperparameters()[0].values
# best_model_config = tunerr.get_best_models()[0].get_config()
# print("Best Hyperparameters:")
# print(best_hyperparameters)
# print("\nBest Model Configuration:")
# print(best_model_config)


model = Sequential()
model.add(Flatten())
model.add(Dense(units=170, activation="relu"))
model.add(Dense(units=154, activation="relu"))
model.add(Dense(units=162, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100,validation_data=(X_test, y_test), class_weight=class_weights,  batch_size=32)


y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.79).astype(int)

confusion_mtx = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Confusion matrix:")
print(confusion_mtx)
print("Accuracy:", accuracy)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)

sns.heatmap(confusion_mtx, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()





