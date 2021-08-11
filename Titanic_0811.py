#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Setup
import pandas as pd
from sklearn.model_selection import train_test_split


# read the data
train_file = "../titanic/titanic_data/train.csv"
test_file = "../titanic/titanic_data/test.csv"

X_full = pd.read_csv(train_file, index_col="PassengerId")
X_test_full = pd.read_csv(test_file, index_col="PassengerId")


# In[3]:


X_full.head()


# In[4]:


X_full.dropna(axis=0, subset=["Survived"], inplace=True)
y = X_full.Survived
X_full.drop(["Survived"], axis=1, inplace=True)


X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 
                                                      train_size=0.8, test_size=0.2,
                                                      random_state=0)


# In[5]:


categorical_cols = []
for col in ["Sex", "Embarked", "Pclass", "Age", "SibSp", "Parch", "Fare"]:
    if X_train_full[col].dtypes == "object":
        categorical_cols.append(col)

numerical_cols = []
for col in ["Sex", "Embarked", "Pclass", "Age", "SibSp", "Parch", "Fare"]:
    if X_train_full[col].dtypes in ["int64","float64"]:
        numerical_cols.append(col)


my_cols = categorical_cols + numerical_cols
#my_cols = ["Sex", "Embarked", "Pclass", "Age", "SibSp", "Parch", "Fare"]
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()


# In[6]:


X_test.head()


# ## Preprocessing

# In[12]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score


numerical_transformer = Pipeline(steps=[
    ("num_imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])


categorical_transformer = Pipeline(steps=[
    ("cat_imputer", SimpleImputer(strategy="constant")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])


# ## Define model

# In[15]:


model = XGBClassifier(random_state=0, 
                      max_depth=5, 
                      n_estimators=1000, 
                      learning_rate=0.01,
                      use_label_encoder=False,
                      eval_metric="logloss")


# In[16]:


my_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

my_pipeline.fit(X_train, y_train)


# In[17]:


preds = my_pipeline.predict(X_valid)

score = mean_absolute_error(y_valid, preds)


# In[18]:


print(score)


# ## Generate test predictions

# In[19]:


preds_test = my_pipeline.predict(X_test)


# In[21]:


output = pd.DataFrame({"PassengerId":X_test.index,
                       "Survived":preds_test})

output.to_csv("submission_titanic_0811.csv", index=False)


# In[ ]:




