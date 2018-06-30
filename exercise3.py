
# coding: utf-8

# In[1]:


print("Hello world")


# In[2]:


import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT="https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH="datasets/housing"
HOUSING_URL=DOWNLOAD_ROOT+HOUSING_PATH+"/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path=os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz=tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[3]:


import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path=os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)


# In[4]:


housing=load_housing_data()
housing.head()


# In[5]:


housing.info()


# In[6]:


housing.describe()


# In[7]:


import numpy as np

def split_train_test(data,test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[8]:


train_set, test_set=split_train_test(housing,0.2)
print(len(train_set),"train +", len(test_set),"test")


# In[9]:


from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids=data[id_column]
    in_test_set=ids.apply(lambda id_:test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# In[10]:


housing_with_id=housing.reset_index()
train_set, test_set=split_train_test_by_id(housing_with_id, 0.2, "index")


# In[11]:


housing_with_id["id"]=housing["longitude"]*1000+housing["latitude"]
train_set, test_set=split_train_test_by_id(housing_with_id,0.2,"id")


# In[12]:


from sklearn.model_selection import train_test_split

train_set, test_set=train_test_split(housing,test_size=0.2, random_state=42)


# In[13]:


housing["income_cat"]=np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"]<5,5.0,inplace=True)


# In[14]:


from sklearn.model_selection import StratifiedShuffleSplit

split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]


# In[15]:


housing["income_cat"].value_counts() / len(housing)


# In[16]:


for set_ in (strat_train_set, strat_test_set):
    set.drop("income_cat",axis=1, inplace=True)


# In[17]:


housing=strat_train_set.copy()


# In[18]:


housing.plot(kind="scatter", x="longitude", y="latitude")


# In[19]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# In[20]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
            s=housing["population"]/100, label="population", figsize=(10, 7),
            c="median_house_value", cmap=plt.get_cmap("jet"),colorbar=True, sharex=False
)
plt.legend()


# In[21]:


corr_matrix=housing.corr()


# In[22]:


from pandas.plotting import scatter_matrix
attributes=["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))


# In[23]:


housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)


# In[24]:


housing=strat_train_set.drop("median_house_value",axis=1)
housing_labels=strat_train_set["median_house_value"].copy()


# In[25]:


housing.dropna(subset=["total_bedrooms"])
housing.drop("total_bedrooms", axis=1)
median=housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)


# In[26]:


from sklearn.preprocessing import Imputer
imputer=Imputer(strategy="median")


# In[27]:


housing_num=housing.drop("ocean_proximity", axis=1)


# In[28]:


imputer.fit(housing_num)


# In[30]:


imputer.statistics_


# In[31]:


housing_num.median().values


# In[32]:


X=imputer.transform(housing_nuum)


# In[33]:


housing_tr=pd.DataFrame(X, columns=housing_num.columns, index=list(housing.index.values))


# In[34]:


housing_cat=housing["ocean_proximity"]
housing_cat.head(10)


# In[35]:


housing_cat_encoded,housing_categories = housing_cat.factorize()
housing_cat_encoded[:10]


# In[36]:


housing_categories


# In[37]:


from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder()
housing_cat_1hot=encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot


# In[38]:


housing_cat_1hot.toarray()


# In[39]:


from sklearn.preprocessing import CategoricalEncoder
cat_encoder=CategoricalEncoder()
housing_cat_reshaped = housing_cat.values.reshape(-1,1)
housing_cat_1hot=cat_encoder.fit_transform(housing_cat_reshaped)
housing_cat_1hot


# In[40]:


cat_encoder=CategoricalEncoder(encoding="onehot-dense")
housing_cat_1hot=cat_encoder.fit_transform(housing_cat_reshaped)
housing_cat_1hot


# In[41]:


cat_encoder.categories_


# In[42]:


from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room=add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self,X,y=None):
        rooms_per_household=X[:, rooms_ix]/X[:, household_ix]
        population_per_household=X[:, population_ix]/X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room=X[:, bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X,rooms_per_household,population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder=CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs=attr_adder.transform(housing.values)


# In[43]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline=Pipeline([
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

housing_num_tr=num_pipeline.fit_transform(housing_num)


# In[44]:


from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names=attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names].values


# In[45]:


num_attribs=list(housing_num)
cat_attribs=["ocean_proximity"]

num_pipeline=Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline=Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
])


# In[46]:


from sklearn.pipeline import FeatureUnion
full_pipeline=FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])


# In[47]:


housing_prepared=full_pipeline.fit_transform(housing)
housing_prepared


# In[48]:


housing_prepared.shape


# In[49]:


from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)


# In[50]:


some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
some_data_prepared=full_pipeline.transform(some_data)
print("prediction: ", lin_reg.predict(some_data_prepared))
print("label: ", list(some_labels))


# In[51]:


from sklearn.metrics import mean_squared_error
housing_predictions=lin_reg.predict(housing_prepared)
lin_mse=mean_squared_error(housing_labels, housing_predictions)
lin_rmse=np.sqrt(lin_mse)
lin_rmse


# In[52]:


from sklearn.tree import DEcisionTreeRegressor

tree_reg=DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# In[53]:


housing_predictions=tree_reg.predict(housing_prepared)
tree_mse=mean_squared_error(housing_labels, housing_predictions)
tree_rmse=np.sqrt(tree_mse)
tree_rmse


# In[54]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(tree_reg,housing_prepared, housing_labels, scoring="neg_mean_squared_error",cv=10)
tree_rmse_scores=np.sqrt(-scores)


# In[55]:


from sklearn.model_selection import GridSearchCV

param_grid=[
    {'n_estimators': [3,10,30], 'max_features': [2,4,6,8]},
    {'bootstrap': [False], 'n_estimators': [3,10], 'max_features':[2,3,4]},
]

forest_reg=RandomForestRegressor()

grid_search=GridSearchCV(forest_reg, param_grid, cv=5,
                        scoring='neg_mean_squared_error',
                        return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)


# In[56]:


grid_search.best_params_


# In[57]:


grid_search.best_estimator_


# In[58]:


final_model=grid_search.best_estimator_

X_test=strat_test_set.drop("median_house_value", axis=1)
y_test=strat_test_set["median_house_value"].copy()

X_test_prepared=full_pipeline.transform(X_test)

final_predictions=final_model.predict(X_test_prepared)

final_mse=mean_squared_error(y_test, final_predictions)
final_rmse=np.sqrt(final_mse)

