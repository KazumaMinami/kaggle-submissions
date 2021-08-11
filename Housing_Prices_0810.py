#!/usr/bin/env python
# coding: utf-8

# ## regularizer を使用
# Dropoutの代わりに正則化項を使用して過学習を抑えるモデルを作成する。
# 
# LearningRateの調整も行ってみる。

# In[1]:


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer


# In[2]:


filepath = "../kaggle/house_data/train.csv"

train_data = pd.read_csv(filepath)

train_data.head()


# In[4]:


X = train_data.copy()
y = X.pop("SalePrice")

num_features = []
for col in X.columns:
    if X[col].dtypes in ["int64", "float64"]:
        num_features.append(col)
print(num_features)

cat_features = []
for col in X.columns:
    if X[col].dtypes=="object":
        cat_features.append(col)
        
print()
print(cat_features)


# ## Pipeline

# In[5]:


num_transformer = make_pipeline(SimpleImputer(strategy="constant"),
                                StandardScaler())

cat_transformer = make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"),
                                OneHotEncoder(handle_unknown="ignore", sparse=False))

preprocessor = make_column_transformer((num_transformer, num_features),
                                       (cat_transformer, cat_features))


# In[6]:


my_columns = num_features + cat_features


# In[9]:


X_select = X[my_columns]

X_select.shape


# In[10]:


X_train, X_valid, y_train, y_valid = train_test_split(X_select, y, train_size=0.7)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)


# In[11]:


input_shape = X_train.shape[1]

print(input_shape)


# ## Define model

# In[90]:


import tensorflow as tf


model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1024, activation="relu", 
                          kernel_regularizer=tf.keras.regularizers.L2(0.01), 
                          input_shape=(input_shape,)),
    tf.keras.layers.Dense(units=512, activation="relu",
                          kernel_regularizer=tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.Dense(units=256, activation="relu",
                          kernel_regularizer=tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.Dense(units=256, activation="relu",
                          kernel_regularizer=tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.Dense(units=128, activation="relu",
                          kernel_regularizer=tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.Dense(units=1)
])


# In[91]:


#adam = tf.keras.optimizers.Adam(learning_rate=0.1)

model.compile(optimizer="adam", 
              loss="MAE")


# In[92]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[93]:


#early_stopping = tf.keras.callbacks.EarlyStopping(patience=10,
 #                                                 min_delta=0.001,
  #                                                restore_best_weights=True)

history = model.fit(X_train, y_train,
                    validation_data=(X_valid, y_valid),
                    batch_size=256,
                    epochs=200,
                    #callbacks=[early_stopping]
                   )


# In[89]:


print(history.history.keys())


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(["Train", "Valid"])
plt.show()


# ## Test Data

# In[94]:


test_file = "../kaggle/house_data/test.csv"

test = pd.read_csv(test_file)


# In[95]:


test_data = test[my_columns]

test_data.shape


# In[96]:


X_test = preprocessor.transform(test_data)

X_test.shape


# In[97]:


preds = model.predict(X_test)


# In[98]:


import itertools

preds_1d = list(itertools.chain.from_iterable(preds))

preds_1d


# In[99]:


output = pd.DataFrame({"Id":test.Id,
                       "SalePrice":preds_1d})

output.to_csv("submission_house_price_v5.csv", index=False)


# In[ ]:




