#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from mamsi.mamsi_pls import MamsiPls
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


# In[ ]:


#load the raw data
LD = pd.read_csv('C:\\Users\\hxr\\Desktop\\Comic_data\\LD.csv', sep=',')
LD = LD.select_dtypes(include=[float, int]).T

BM = pd.read_csv('C:\\Users\\hxr\\Desktop\\Comic_data\\BM.csv', sep = ',')
BM = BM.select_dtypes(include=[float, int]).T
BM.replace(0, np.nan, inplace=True)

HDF= pd.read_csv('C:\\Users\\hxr\\Desktop\\Comic_data\\HDF.csv', sep=',')
HDF = HDF.select_dtypes(include=[float, int]).T

PM = pd.read_csv('C:\\Users\\hxr\\Desktop\\Comic_data\\PM.csv', sep = ',')
PM = PM.select_dtypes(include=[float, int]).T

#outcome variable
sample_data = pd.read_csv('C:\\Users\\hxr\\Desktop\\Comic_data\\sample_data.csv', sep = ',')
#classification for DIAB and SEX
Y1 = sample_data.set_index('SampleId')["DIAB"].apply(lambda x: 1 if x == 'Non-diabetic' else 0)
Y2= sample_data.set_index('SampleId')["SEX"].apply(lambda x: 1 if x == 'Female' else 0)
#regression for AGE and BMI
Y3= sample_data.set_index('SampleId')["AGE"]
Y4= sample_data.set_index('SampleId')["BMI"]


# In[ ]:


# check the data
LD.describe()


# In[ ]:


def analyze_missing_data(df):

    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    num_columns_over_30 = len(missing_percentage[missing_percentage > 30])
    num_columns_under_30 = len(missing_percentage[missing_percentage <= 30])

    print('Number of columns with missing values over 30%:', num_columns_over_30)

    plt.figure(figsize=(8, 8))
    plt.pie([num_columns_over_30, num_columns_under_30],
            labels=['> 30% missing', '<= 30% missing'],
            autopct='%1.1f%%',
            colors=['lightcoral', 'lightgreen'])
    plt.title('Proportion of Columns with Missing Values Over 30%', fontsize=16)
    plt.tight_layout()
    plt.show()


# In[ ]:


analyze_missing_data(HDF)
analyze_missing_data(LD)
analyze_missing_data(BM)
analyze_missing_data(PM)


# In[ ]:


#comparision of different imputing methods( using HDF as an example)
#KNN
knn_imputer = KNNImputer(n_neighbors=5)  
HDF_knn_imputed = knn_imputer.fit_transform(HDF_missing)
mse_knn = mean_squared_error(HDF_complete.fillna(0), HDF_knn_imputed)
print(mse_knn)

#mean 
HDF_missing1=HDF_missing.replace(0, pd.NA)
HDF_mean_imputed=HDF_missing1.fillna(HDF_missing.mean())
mse_mean = mean_squared_error(HDF_complete.fillna(0), HDF_mean_imputed)
print(mse_mean)

#zero
HDF_zero_imputed = HDF_missing.fillna(0)
mse_zero = mean_squared_error(HDF_complete.fillna(0), HDF_zero_imputed)
print(mse_zero)
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

#PCA_imputation
def pca_impute(X_miss, n_components=2):
    pca = PCA(n_components=n_components)
    imputer = SimpleImputer(strategy='mean')  # use mean

    # use mean then apply PCA
    X_filled = imputer.fit_transform(X_miss)
    pca.fit(X_filled)
    X_pca = pca.inverse_transform(pca.transform(X_filled))
    return X_pca

HDF_pca_imputed = pca_impute(HDF_missing, n_components=3)
mse_pca = mean_squared_error(HDF_complete.fillna(0), HDF_pca_imputed)
print(mse_pca)


from sklearn.impute import IterativeImputer

MAI_imputer = IterativeImputer(random_state=42)
HDF_MAI_imputed = MAI_imputer.fit_transform(HDF_missing) 
mse_MAI=mean_squared_error(HDF_complete.fillna(0), HDF_MAI_imputed)
print(mse_MAI)

#iterative imputer
imputer = IterativeImputer(random_state=30)
HDF_iterative_imputed=imputer.fit_transform(HDF_missing)
mse_iterative=mean_squared_error(HDF_complete.fillna(0), HDF_iterative_imputed)
print(mse_iterative)

#visualization( choosing the best one)
import matplotlib.pyplot as plt
import numpy as np

mses_HDF = [mse_knn, mse_mean, mse_zero, mse_pca, mse_iterative]
x_labels = ["KNN", "Mean", "Zero", "PCA", "Iterative"]
n_bars = len(mses_HDF)  

colors = ["r", "g", "b", "orange", "purple"]

xval = np.arange(n_bars)

plt.figure(figsize=(12, 6))
ax1 = plt.subplot(111) 
for j in xval:
    ax1.barh(
        j,  
        mses_HDF[j],  
        color=colors[j], 
        alpha=0.6,  
        align="center"  
    )

ax1.set_title("Imputation Techniques with HDF Dataset")
ax1.set_xlim(left=np.min(mses_HDF) * 0.9, right=np.max(mses_HDF) * 1.1)  
ax1.set_yticks(xval)  
ax1.set_xlabel("MSE")  
ax1.invert_yaxis()  
ax1.set_yticklabels(x_labels)  

plt.show()


# In[ ]:


# comparision of scaling methods
from sklearn.preprocessing import StandardScaler
import seaborn as sns

scaler = StandardScaler()
# scaling
# 1. log + standard scaling
HDF_log_scaled = np.log1p(HDF_filled)   
HDF_log_standard_scaled = pd.DataFrame(scaler.fit_transform(HDF_log_scaled), columns=HDF_filled.columns)

# 2. standard scaling
HDF_standard_scaled = pd.DataFrame(scaler.fit_transform(HDF_filled), columns=HDF_filled.columns)

def plot_boxplots(df, n = 25):
    boxplot_data = []
    for i in range(0, df.shape[1],n):
        sampled_feature = df.iloc[:,i:i+n].sample(n=1, axis=1)
        boxplot_data.append(sampled_feature)

    boxplot_df = pd.concat(boxplot_data, axis = 1)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=boxplot_df)
    plt.title('Boxplots of Features in HDF Dataset-Standard Scaled')
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    


# In[ ]:


plot_boxplots(HDF_log_standard_scaled)
plot_boxplots(HDF_standard_scaled)


# In[ ]:


#data preprocessing
# impute the missing value
HDF_cleaned = HDF.dropna(thresh=len(HDF) * 0.7, axis=1)
BM_cleaned = BM.dropna(thresh=len(BM) * 0.7, axis=1)
LD_cleaned = LD.dropna(thresh=len(LD) * 0.7, axis=1)
PM_cleaned = PM.dropna(thresh=len(PM) * 0.7, axis=1)

knn_imputer = KNNImputer(n_neighbors=5)
HDF_knn_imputed = knn_imputer.fit_transform(HDF_cleaned)
BM_knn_imputed = knn_imputer.fit_transform(BM_cleaned)
LD_knn_imputed = knn_imputer.fit_transform(LD_cleaned)
PM_knn_imputed = knn_imputer.fit_transform(PM_cleaned)

HDF_filled= pd.DataFrame(HDF_knn_imputed, columns=HDF_cleaned.columns, index=HDF_cleaned.index)
BM_filled= pd.DataFrame(BM_knn_imputed, columns=BM_cleaned.columns, index=BM_cleaned.index)
LD_filled= pd.DataFrame(LD_knn_imputed, columns=LD_cleaned.columns, index=LD_cleaned.index)
PM_filled= pd.DataFrame(PM_knn_imputed, columns=PM_cleaned.columns, index=PM_cleaned.index)


# In[ ]:


#scaling
#scaling: log+standard
scaler = StandardScaler()

HDF_log_scaled = np.log1p(HDF_filled)
HDF_log_standard_scaled = pd.DataFrame(scaler.fit_transform(HDF_log_scaled),
                                       columns=HDF_filled.columns,
                                       index=HDF_filled.index)

BM_log_scaled = np.log1p(BM_filled)
BM_log_standard_scaled = pd.DataFrame(scaler.fit_transform(BM_log_scaled),
                                       columns=BM_filled.columns,
                                       index=BM_filled.index)

LD_log_scaled = np.log1p(LD_filled)
LD_log_standard_scaled = pd.DataFrame(scaler.fit_transform(LD_log_scaled),
                                       columns=LD_filled.columns,
                                       index=LD_filled.index)

PM_log_scaled = np.log1p(PM_filled)
PM_log_standard_scaled = pd.DataFrame(scaler.fit_transform(PM_log_scaled),
                                       columns=PM_filled.columns,
                                       index=PM_filled.index)


# In[ ]:


#def Cleaned_X(X, na_threshold=0.7, n_neighbors=5):

    # Step 1: Drop columns with too many missing values
    X_cleaned = X.dropna(thresh=len(X) * na_threshold, axis=1)

    # Step 2: Perform KNN imputation
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    X_knn_imputed = knn_imputer.fit_transform(X_cleaned)

    # Step 3: Recreate DataFrame with original column names and indices
    X_filled = pd.DataFrame(X_knn_imputed, columns=X_cleaned.columns, index=X_cleaned.index)
    
    return X_filled


# In[ ]:


LD_cleaned = Cleaned_X(LD)
BM_cleaned = Cleaned_X(BM)
PM_cleaned = Cleaned_X(PM)
HDF_cleaned=Cleaned_X(HDF)


# In[ ]:


#mapping
y.index = y.index.astype(str)  
common_index = set(y.index).intersection(set(BM_cleaned.index)).intersection(set(PM_cleaned.index).intersection(set(HDF_cleaned.index)))
y_aligned_final = y[y.index.isin(common_index)]  # Filter y by common indices
BM_aligned_final = BM_cleaned[BM_cleaned.index.isin(common_index)]
PM_aligned_final = PM_cleaned[PM_cleaned.index.isin(common_index)]
HDF_aligned_final = HDF_cleaned[HDF_cleaned.index.isin(common_index)]
y_aligned_final.value_counts()
#save the mapping files


# In[ ]:


#PCA algorithm exploration
#Variance
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
x=np.array([1,2,3,4,5,6,7,8,9,10])
x = x-x.mean()
print(x)
print(np.var(x,ddof=1))
print(np.dot(x,x)/(len(x)-1)) 

#Covariance
x=pd.Series([1,2,3,4,5])
y=pd.Series([29,45,36,78,58])
x=x-x.mean()
y=y-y.mean()
print(x.cov(y))
print(np.dot(x,y)/(len(x)-1))

#Covariance Matirx
np.random.seed(0)
X=np.random.random(size=(3,4))
X=X-X.mean(axis=0)
print(np.cov(X,rowvar=False))
print(np.dot(X.T,X)/(len(X)-1))

#geometric interpretation of covariance matrix
#rotation
v=np.array([4,4])
theta= 1/6 * np.pi
A= np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta), np.cos(theta)]])
v2=np.dot(A,v)
print(v2)

def plot_point(*points):
    for point in points:
        plt.plot(*point)
        
plt.figure(figsize=(5,5))
origin=(0,0)
plot_point(origin,v,v2)
plt.axis("equal")
a1=plt.arrow(*origin,*v,head_width=0.1,color="yellow")
a2=plt.arrow(*origin,*v2,head_width=0.1,color="blue")
plt.legend([a1,a2],["v","v2"])

origin=(0,0)
v=np.array([2,3])
w1=np.array([-1,1])
w2=np.array([-1,-1])
norm_w1=w1/np.linalg.norm(w1)
norm_w2=w2/np.linalg.norm(w2)
w1_coordinate=np.dot(v,norm_w1)
w2_coordinate=np.dot(v,norm_w2)
print(w1_coordinate,w2_coordinate)
plt.figure(figsize=(5,5))
plot_point(origin,v,w1,w2)
plt.axis("equal")
plt.arrow(*origin,*(1,0),head_width=0.1,color="black")
plt.arrow(*origin,*(0,1),head_width=0.1,color="black")
a1=plt.arrow(*origin,*v,head_width=0.1,color="yellow")
a2=plt.arrow(*origin,*w1,head_width=0.1,color="blue")
a3=plt.arrow(*origin,*w2,head_width=0.1,color="red")
plt.legend([a1,a2,a3],["v","norm_w1","norm_w2"])


# In[ ]:


#PCA--outlier detection
# HDF
pca = PCA(n_components=5)
pca_features = pca.fit_transform(HDF)

df_pairplot = pd.DataFrame(pca_features, columns = [f'PC{i+1}' for i in range(5)])
df_pairplot['DIAB Group'] = df_pairplot['DIAB'].map({0: 'Non-DIAB', 1: 'DIAB'})

sns.pairplot(df_pairplot, diag_kind = 'kde', hue = 'ID Group',palette = custom_palette)
plt.suptitle('HDF Pair Plots: PC1 to PC5')
plt.show()

#BM LD PM
pca_features2 = pca.fit_transform(BM)
pca_features3 = pca.fit_transform(LD)
pca_features4 = pca.fit_transform(PM)

df_pairplot2 = pd.DataFrame(pca_features2, columns = [f'PC{i+1}' for i in range(5)])
df_pairplot2['ID Group'] = pd.qcut(df_pairplot2.index, q=2, labels=["A","B"])
custom_palette = {
    'A': 'green',
    'B': '#3399ff'
}
sns.pairplot(df_pairplot2, diag_kind = 'kde', hue = 'ID Group',palette = custom_palette)
plt.suptitle('BM Pair Plots: PC1 to PC5')
plt.show()


df_pairplot3 = pd.DataFrame(pca_features3, columns = [f'PC{i+1}' for i in range(5)])
df_pairplot3['ID Group'] = pd.qcut(df_pairplot3.index, q=2, labels=["A","B"])
custom_palette = {
    'A': 'green',
    'B': '#3399ff'
}
sns.pairplot(df_pairplot3, diag_kind = 'kde', hue = 'ID Group',palette = custom_palette)
plt.suptitle('LD Pair Plots: PC1 to PC5')
plt.show()


df_pairplot4 = pd.DataFrame(pca_features4, columns = [f'PC{i+1}' for i in range(5)])
df_pairplot4['ID Group'] = pd.qcut(df_pairplot4.index, q=2, labels=["A","B"])
custom_palette = {
    'A': 'green',
    'B': '#3399ff'
}
sns.pairplot(df_pairplot4, diag_kind = 'kde', hue = 'ID Group',palette = custom_palette)
plt.suptitle('PM Pair Plots: PC1 to PC5')
plt.show()


# In[ ]:


#PCA
#raw data
pca=PCA(n_components=2)
HDF_pca=pca.fit_transform(HDF.fillna(0)[:297])
_, ax = plt.subplots()  
scatter = ax.scatter(HDF_pca[:, 0], HDF_pca[:, 1], c=y_DIAB, cmap='viridis') 

ax.set(xlabel='PC1', ylabel='PC2')
handles, labels = scatter.legend_elements()
ax.legend(handles, labels, loc="lower right", title="DIAB")
plt.title('PCA analysis in raw dataset')
plt.show()

#preprocessed data
pca=PCA(n_components=2)
HDF_scaling=HDF_log_standard_scaled_cleaned.fillna(0)
HDF_pca=pca.fit_transform(HDF_scaling)
_, ax = plt.subplots()  
scatter = ax.scatter(HDF_pca[:, 0], HDF_pca[:, 1], c=y_DIAB, cmap='viridis') 

ax.set(xlabel='PC1', ylabel='PC2')
handles, labels = scatter.legend_elements()
ax.legend(handles, labels, loc="lower right", title="DIAB")
plt.title('PCA analysis in HDF_scaled dataset')
plt.show()


# In[ ]:


# differnet combination of blocks with MAMSI


# In[ ]:


#load the preprocessed data
HDF = pd.read_csv('C:\\Users\\hxr\\Desktop\\Permutation test\\HDF.csv', sep=',', index_col=0)
BM = pd.read_csv('C:\\Users\\hxr\\Desktop\\Permutation test\\BM.csv', sep=',', index_col=0)
LD = pd.read_csv('C:\\Users\\hxr\\Desktop\\Permutation test\\LD.csv', sep=',', index_col=0)
PM = pd.read_csv('C:\\Users\\hxr\\Desktop\\Permutation test\\PM.csv', sep=',', index_col=0)
Y = pd.read_csv('C:\\Users\\hxr\\Desktop\\Permutation test\\Y.csv', sep=',', index_col=0)


# In[ ]:


#four blocks 
HDF_train, HDF_test, Y_train, Y_test = train_test_split(HDF, Y, test_size=0.2, random_state=42)
BM_train = BM.loc[HDF_train.index]
BM_test = BM.loc[HDF_test.index]
LD_train = LD.loc[HDF_train.index]
LD_test = LD.loc[HDF_test.index]
PM_train = PM.loc[HDF_train.index]
PM_test = PM.loc[HDF_test.index]
#fit model
mamsipls = MamsiPls(n_components=1)
mamsipls.fit([HDF_train, BM_train, LD_train, PM_train], Y_train)
# predict
y_pred = mamsipls.evaluate_class_model([HDF_test, BM_test, LD_test, PM_test], Y_test)


# In[ ]:


mamsipls.estimate_lv([HDF_train, BM_train, LD_train, PM_train], Y_train, metric="auc")


# In[ ]:


print(mamsipls.n_components)


# In[ ]:


y_pred = mamsipls.evaluate_class_model([HDF_test, BM_test, LD_test, PM_test], Y_test)


# In[ ]:


mamsipls.block_importance(block_labels=["HDF","BM","LD","PM"],normalised=True)


# In[ ]:


#single blocks, pair blocks and three blocks are in other files


# In[ ]:


#permutation tests(HPC)


# In[ ]:


#Selected features


# In[ ]:


p_vals_MB = pd.read_csv("C:\\Users\\hxr\\Desktop\\Permutation final\\50MBpvals.csv", header=None).values
#p_vals_MB2 = pd.read_csv("C:\\Users\\hxr\\Desktop\\Permutation final\\MBpvals2.csv").values
p_vals_10MB = pd.read_csv("C:\\Users\\hxr\\Desktop\\Permutation final\\10MBpvals.csv",header=None).values
p_vals_50HDF= pd.read_csv("C:\\Users\\hxr\\Desktop\\Permutation final\\50HDFpvals.csv",header=None).values
#p_vals_BM= pd.read_csv("C:\\Users\\hxr\\Desktop\\Permutation final\\BMpvals.csv").values
p_vals_50BM= pd.read_csv("C:\\Users\\hxr\\Desktop\\Permutation final\\50BMpvals.csv",header=None).values
p_vals_10PM= pd.read_csv("C:\\Users\\hxr\\Desktop\\Permutation final\\10PMpvals.csv",header=None).values
p_vals_50PM= pd.read_csv("C:\\Users\\hxr\\Desktop\\Permutation final\\50PMpvals.csv",header=None).values
p_vals_50LD= pd.read_csv("C:\\Users\\hxr\\Desktop\\Permutation final\\50LDpvals.csv",header=None).values
p_vals_50HDF_BM_PM= pd.read_csv("C:\\Users\\hxr\\Desktop\\Permutation final\\50HDF_BM_PMpvals.csv",header=None).values
#p_vals_PM_HDF=pd.read_csv("C:\\Users\\hxr\\Desktop\\Permutation final\\PM_HDFpvals.csv").values
#p_vals_PM_BM=pd.read_csv("C:\\Users\\hxr\\Desktop\\Permutation final\\PM_BMpvals.csv").values


# In[ ]:


def calculate_zero_p_value_ratio(p_vals):
    ratio = np.sum(p_vals == 0) / len(p_vals) if len(p_vals) > 0 else 0
    print(ratio)
    return ratio


# In[ ]:


# determine the best times of permutation tests
calculate_zero_p_value_ratio(p_vals_10MB)
calculate_zero_p_value_ratio(p_vals_MB)
calculate_zero_p_value_ratio(p_vals_50BM)
calculate_zero_p_value_ratio(p_vals_50LD)
calculate_zero_p_value_ratio(p_vals_50HDF)


# In[ ]:


# features selected 
#four blocks 
x = pd.concat([HDF, BM, LD, PM], axis=1)
mask = np.where( p_vals< 0.01)
selected_MB = x.iloc[:, mask[0]]

# in single block
# HDF
mask = np.where(p_vals_10HDF < 0.01)
selected_10HDF = HDF.iloc[:, mask[0]]
#LD
mask = np.where(p_vals_10LD < 0.01)
selected_10LD = LD.iloc[:, mask[0]]
selected_10LD.columns.intersection(selected_MB.columns)
#PM
mask = np.where(p_vals_10PM < 0.01)
selected_10PM = PM.iloc[:, mask[0]]
selected_10PM.columns.intersection(selected_MB.columns)
#BM
mask = np.where(p_vals_10BM < 0.01)
selected_10BM = BM.iloc[:, mask[0]]


# In[ ]:


#visualize the mapping of features selected
import pandas as pd
from upsetplot import from_memberships, UpSet
import matplotlib.pyplot as plt

data_sets = [selected_10HDF.columns, selected_10BM.columns, selected_10LD.columns, selected_10PM.columns, selected_MB.columns]
all_columns = set().union(*data_sets)
data_dict = {col: [col in dataset for dataset in data_sets] for col in all_columns}

upset_data = pd.DataFrame.from_dict(data_dict, orient='index', columns=['HDF', 'BM', 'LD', 'PM', 'MB'])
print(upset_data)


# In[ ]:


upset_data.columns = upset_data.columns.astype(str)
upset_data.index = upset_data.index.astype(str)

memberships_with_index = [
    (tuple(upset_data.columns[upset_data.loc[feature] == 1]), feature)
    for feature in upset_data.index
]

memberships_df = pd.DataFrame(memberships_with_index, columns=["Memberships", "Feature"])
membership_counts = memberships_df["Memberships"].value_counts()

upset_input = from_memberships(
    membership_counts.index,
    data=membership_counts.values
)

print("Memberships with Features:")
for membership, group in memberships_df.groupby("Memberships"):
    print(f"{membership}: {list(group['Feature'])}")

plot = UpSet(upset_input, show_counts='%d', show_percentages=False)
plot.plot()
plt.show()


# In[ ]:


#MCCM
from types import MethodType
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.metrics import mean_squared_error as root_mean_squared_error
from sklearn.metrics import r2_score
from copy import deepcopy
import numpy as np
import pandas as pd

def montecarlo_cv(self, x, y, groups=None, classification=True, return_train=False, test_size=0.2, repeats=10, random_state=42):
    """
    Evaluate MB-PLS model using Monte Carlo Cross-Validation (MCCV).
    """

    # Check if PLS model is fitted
    check_is_fitted(self, 'beta_')

    # Validate inputs
    _x = x.copy()
    if isinstance(_x, list) and not isinstance(_x[0], list):
        pass
    else:
        _x = [x]
    _y = y.copy()
    _y = check_array(_y, ensure_2d=False)

    # Generate random sequence of seeds for MCCV
    rng = np.random.RandomState(random_state)

    # Generate n random numbers
    random_numbers = rng.randint(1, np.iinfo(np.int32).max, size=repeats)

    # Placeholder for MCCV scores
    scores = pd.DataFrame()

    # if groups are provided, group test-train split is performed otherwise sk-learn test-train split is used
    for i in range(repeats):
        if groups is None:
            train, test, y_train, y_test = train_test_split(_x[0], _y, test_size=test_size, random_state=random_numbers[i])
            x_train = [df.loc[train.index] for df in _x]
            x_test = [df.loc[test.index] for df in _x]
        else:
            x_train, x_test, y_train, y_test = self.group_train_test_split(_x, _y, groups=groups, test_size=test_size, random_state=random_numbers[i])

        # Fit model and predict
        x_train_copy = deepcopy(x_train)
        self.fit_transform(x_train_copy, y_train)
        y_predicted = self.predict(x_test)
        predictions = [y_predicted]
        truths = [y_test]

        # Add training scores
        if return_train:
            y_predicted_train = self.predict(x_train)
            predictions.append(y_predicted_train)
            truths.append(y_train)

        # Classification model evaluation
        if classification:
            predictions_cl = [np.where(y_predicted > 0.5, 1, 0)]
            if return_train:
                predictions_cl.append(np.where(y_predicted_train > 0.5, 1, 0))

            # Calculate evaluation metrics for testing and training sets
            for prediction_cl, prediction, truth, j in zip(predictions_cl, predictions, truths, [0, 1]):
                # Evaluation metrics
                try:
                    accuracy = accuracy_score(truth, prediction_cl)
                except ValueError:
                    accuracy = np.nan
                try:
                    precision = precision_score(truth, prediction_cl)
                except ValueError:
                    precision = np.nan
                try:
                    recall = recall_score(truth, prediction_cl, zero_division=np.nan)
                except ValueError:
                    recall = np.nan
                try:
                    f1 = f1_score(truth, prediction_cl)
                except ValueError:
                    f1 = np.nan
                try:
                    tn, fp, _, _ = confusion_matrix(truth, prediction_cl, labels=[0, 1]).ravel()
                    specificity_score = tn/(tn+fp) if (tn + fp) != 0 else np.nan
                except ValueError:
                    specificity_score = np.nan
                try:
                    roc_auc = roc_auc_score(truth, prediction)
                except ValueError:
                    roc_auc = np.nan

                row = pd.DataFrame({
                        'precision': [precision],
                        'recall': [recall],
                        'specificity': [specificity_score],
                        'f1': [f1],
                        'roc_auc': [roc_auc],
                        'accuracy': [accuracy]
                    })

                if j == 0:
                    # save MCCV scores
                    test_score_row = row.copy()
                else:
                    train_score_row = row.copy()

        # Regression model evaluation    
        else:
            for prediction, truth, j in zip(predictions, truths, [0,1]):
                # Evaluation metrics
                rmse = root_mean_squared_error(truth, prediction)
                q2 = r2_score(truth, prediction)

                row = pd.DataFrame({
                        'rmse': [rmse],
                        'q2': [q2]
                    })

                if j == 0:
                    # save MCCV scores
                    test_score_row = row.copy()
                else:
                    train_score_row = row.copy()

        if len(scores) == 0:
            scores = test_score_row
            if return_train:
                train_scores = train_score_row
        else:
            scores = pd.concat([scores, test_score_row], ignore_index=True)
            if return_train:
                train_scores = pd.concat([train_scores, train_score_row], ignore_index=True)

    if return_train:
        return scores, train_scores
    else:    
        return scores


# In[ ]:


#PM
PM_model=MamsiPls(n_components=2)
PM_model.fit([PM_train],Y_train)
PM_model.montecarlo_cv =MethodType(montecarlo_cv, PM_model)
results_PM=PM_model.montecarlo_cv( PM, Y, 
                    classification=True, 
                    return_train=False, 
                    test_size=0.2, 
                    repeats=100, 
                    random_state=42)
results_PM.describe()
results_PM.to_csv(r"C:\\Users\\hxr\\Desktop\\3models\\result_1.csv", index=False)

#HDF_PM
model_2=MamsiPls(n_components=2)
model_2.fit([HDF_train,PM_train],Y_train)
model_2.montecarlo_cv =MethodType(montecarlo_cv, model_2)
result_2=model_2.montecarlo_cv([HDF,PM],Y,
                    classification=True, 
                    return_train=False, 
                    test_size=0.2, 
                    repeats=100, 
                    random_state=42)

result_2.describe()

result_2.to_csv(r"C:\\Users\\hxr\\Desktop\\3models\\result_2.csv", index=False)

#HDF_BM_PM
model_3=MamsiPls(n_components=2)
model_3.fit([HDF_train,BM_train,PM_train],Y_train)
model_3.montecarlo_cv =MethodType(montecarlo_cv, model_3)
result_3=mamsipls.montecarlo_cv([HDF,BM,PM],Y,
                    classification=True, 
                    return_train=False, 
                    test_size=0.2, 
                    repeats=100, 
                    random_state=42)
result_3.to_csv(r"C:\\Users\\hxr\\Desktop\\3models\\result_3.csv", index=False)

#HDF_BM_LD_PM
model_4= MamsiPls(n_components=3)
model_4.fit([HDF_train,BM_train,LD_train, PM_train], Y_train)
model_4.montecarlo_cv =MethodType(montecarlo_cv, model_4)
result_4=mamsipls.montecarlo_cv([HDF,BM,LD,PM],Y,
                    classification=True, 
                    return_train=False, 
                    test_size=0.2, 
                    repeats=100, 
                    random_state=33)

result_4.to_csv(r"C:\\Users\\hxr\\Desktop\\3models\\result_4.csv", index=False)


# In[ ]:




