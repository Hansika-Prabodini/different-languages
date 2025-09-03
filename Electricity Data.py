#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


# ## Importing Data set

# In[2]:


# Use more efficient data loading with dtypes specified
data = pd.read_csv("Z:\\Sasindu\\Data set\\electricity.csv", 
                  index_col=0, 
                  parse_dates=[0],
                  dtype={'HolidayFlag': 'int8', 'DayOfWeek': 'int8', 'WeekOfYear': 'int8', 
                         'Day': 'int8', 'Month': 'int8', 'Year': 'int16', 'PeriodOfDay': 'int8'})


# In[3]:


# No need to create another DataFrame, use data directly
df = data


# In[4]:


df.head()


# In[5]:


df.shape


# ## Data Cleaning

# In[6]:


df.info()


# ##### Here, ForecastWindProduction, SystemLoadEA, SMPEA, ORKTemperature, ORKWindspeed, CO2Intensity, ActualWindProduction, SystemLoadEP2, SMPEP2 should be numeric values. Hance, Converting them into numeric type is needed.

# In[7]:


# Convert to numeric with smaller datatypes to reduce memory
cols_to_numeric = ['ForecastWindProduction', 'SystemLoadEA', 'SMPEA', 'ORKTemperature', 
                   'ORKWindspeed', 'CO2Intensity', 'ActualWindProduction', 'SystemLoadEP2', 'SMPEP2']
for col in cols_to_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
df.info()


# In[8]:


missing_values =df.isnull().sum()
print(missing_values)


# ##### Before fill missing values, Exploratory data Analysis is essential.

# ## Exploratory data Analysis

# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


numeric_vals = ['HolidayFlag', 'DayOfWeek', 'WeekOfYear', 'Day', 'Month', 'Year', 'PeriodOfDay',
               'ForecastWindProduction', 'SystemLoadEA', 'SMPEA', 'ORKTemperature', 'ORKWindspeed',
                'CO2Intensity', 'ActualWindProduction', 'SystemLoadEP2', 'SMPEP2']
for col in numeric_vals:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col])
    plt.tight_layout()
    plt.show()


# In[11]:


df.describe()


# ### Handling missing values

# In[12]:


# Combine with next step to process in one go


# In[13]:


df_cleaned = df.dropna(subset=['ORKTemperature','ORKWindspeed'])


# In[14]:


df_cleaned.shape


# ##### Because of having more missing values in ORKTemperature, ORKWindspeed columns, If they filled with mean or median, it can be error for model. So Null values were removed.

# ### Removing Outliers

# In[15]:


SMPEP2_out = (df_cleaned['SMPEP2'] > 0) | (df_cleaned['SMPEP2'] <= 550)


# In[16]:


df_cleaned = df_cleaned[SMPEP2_out]


# In[17]:


df_cleaned.shape


# In[18]:


# More efficient filling of missing values
fill_with_median = ['ForecastWindProduction','SystemLoadEA','SMPEA','ActualWindProduction', 'SystemLoadEP2', 'SMPEP2']
medians = {col: df_cleaned[col].median() for col in fill_with_median}
df_cleaned[fill_with_median] = df_cleaned[fill_with_median].fillna(medians)


# ##### Because of having skewed distribution on 'ForecastWindProduction','SystemLoadEA','SMPEA','ActualWindProduction', 'SystemLoadEP2', 'SMPEP2' columns. Missing values were filled with median.

# In[19]:


# Simpler operation for CO2Intensity
df_cleaned['CO2Intensity'].fillna(df_cleaned['CO2Intensity'].mean(), inplace=True)


# ##### CO2Intensity has normal dustribution. So null values were filled with mean value of the CO2Intensity.

# In[20]:


missing_values =df_cleaned.isna().sum()
print(missing_values)


# In[21]:


df_cleaned.head()


# ### EDA (Correlation Analysis)

# In[22]:


plt.figure(figsize=(20, 20))
sns.pairplot(df_cleaned,palette='set1')
plt.show()


# In[23]:


numeric_df = df_cleaned.select_dtypes(include = ['number'])
corr_matrix =numeric_df.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix,annot=True, cmap='coolwarm',fmt=".2f")
plt.title('Correlation for elecricity data')
plt.show()


# #### By studing Heatmap, Following observations can be gathered.
# ###### Highly positive correlation between month and weekofyear
# ###### Highly positive correlation between ForecastWindProduction and ActualWindProduction
# ###### Highly positive correlation between SystemLoadEA and SystemLoadEA2
# ###### Highly positive correlation between ORKWindspeed and ForecastWindProduction
# ###### Highly positive correlation between ORKWindspeed and ActualWindProduction

# ####  If some features in dataset have high correlation, it is often a good idea to consider ignoring or removing one of the highly correlated features.
# #### It can be cause to multicolinearity, increase model complexity, reduse model performence. Hance, one feature can be ignored.
# ###### ** Month might be more straightforward and useful in many contexts compared to WeekOfYear. So ,WeekOfYear was removed.
# ###### ** To get more accurate model, I think removing ForecastWindProduction column is suitable.
# ###### ** SystemLoadEA is forecasted national load and SystemLoadEA2 is actual national system load. hance, Removing SystemLoadEA is more accurate.
# ###### ** Because of model requament, ORKWindspeed or ActualWindProduction weren't removed. ForecastWindProduction has already removed.
# ###### ** Holiday column is also populated. Hence, Holiday column can be removed.

# In[24]:


df_new = df_cleaned.drop(columns=['Holiday','WeekOfYear','ForecastWindProduction','SystemLoadEA'])


# In[25]:


df_new.head()


# In[26]:


df_new.shape


# #### Split Date and timestamp from index

# In[27]:


df_scaled = df_new.reset_index()


# In[28]:


df_scaled.columns


# In[29]:


f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,3))
sns.lineplot(x=df_scaled.DateTime, y = df_scaled.Month)
plt.show()


# In[30]:


f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,3))
sns.lineplot(x=df_scaled.DateTime, y = df_scaled.DayOfWeek)
plt.show()


# In[31]:


f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,3))
sns.lineplot(x=df_scaled.DateTime, y = df_scaled.Day)
plt.show()


# In[32]:


from datetime import date

f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,3))
sns.lineplot(x=df_scaled.DateTime, y = df_scaled.PeriodOfDay)
ax.set_xlim([date(2013,12,1),date(2014,1,1)])
plt.show()


# ##### By looking above graphs, We can see cyclic nature.Convert cyclic features into two continuous features using sine and cosine transformations to preserve the cyclic nature.Sine and Cosine functions help in representing the cyclic nature by mapping the feature to a circle. 

# ## Feature Engineering

# In[33]:


def periodic_transform(df,variable):
    # More efficient - calculate angle once
    max_val = df_scaled[variable].max()
    angle = 2 * np.pi * df_scaled[variable] / max_val
    df_scaled[f"{variable}_SIN"] = np.sin(angle)
    df_scaled[f"{variable}_COS"] = np.cos(angle)
    return df_scaled


# In[34]:


# Apply all transformations at once
cyclic_features = ['DayOfWeek', 'Day', 'Month', 'PeriodOfDay']
for feature in cyclic_features:
    df_scaled = periodic_transform(df_scaled, feature)
df_scaled.head()


# ##### Hence, DayOfWeek, Day, Month, and PeriodOfDay  columns can be removed.

# In[35]:


df_scaled = df_scaled.drop(columns=['DateTime','DayOfWeek','Day','Month','PeriodOfDay'])


# In[36]:


df_scaled.columns


# In[37]:


df_scaled.head()


# ### Splitting Data

# In[38]:


x = df_scaled.drop(columns = 'SMPEP2', axis = 1)
y = df_scaled['SMPEP2']


# In[39]:


from sklearn.model_selection import train_test_split


# In[40]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42)


# ### Scaling data

# #### HolidayFlag column 0 and 1 values. Hence, no need to encord this column.
# #### As most of columns having skewed distribution, minmaxscaler canbe applied for those columns such as Year,SMPEA,ORKTemperature,ORKWindspeed,ActualWindProduction,CO2Intensity,SystemLoadEP2,and SMPEP2
# 

# In[41]:


from sklearn.preprocessing import MinMaxScaler


# In[42]:


mm = MinMaxScaler()
x_train_scaled = mm.fit_transform(x_train)
x_test_scaled = mm.transform(x_test)


# ### Define Model

# #### Defining modes for scaled Data

# In[43]:


def model_acc(model):
    model.fit(x_train_scaled,y_train)
    acc = model.score(x_test_scaled,y_test)
    print(str(model)+'-->'+str(acc))


# In[44]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model_acc(lr)

from sklearn.linear_model import Lasso
lasso = Lasso()
model_acc(lasso)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
model_acc(dt)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
model_acc(rf)

from sklearn.svm import SVR
svm = SVR()
model_acc(svm)


# In[45]:


best_model = rf


# In[46]:


rnd_reg = RandomForestRegressor(n_estimators=200, oob_score=True, criterion='squared_error', n_jobs=-1, random_state=42)
bc_rf = rnd_reg.fit(x_train_scaled, y_train)


# In[47]:


y_test_pred_2 = bc_rf.predict(x_test_scaled)


# In[48]:


y_test_pred = best_model.predict(x_test_scaled)


# In[ ]:





# ### Evaluation for Scaled Data for Best Model

# In[49]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_pacf


# In[50]:


y_test_pred = y_test_pred.flatten()


# In[51]:


y_test_pred_2 = y_test_pred.flatten()


# In[52]:


final_df = pd.DataFrame(np.hstack((y_test_pred[:, np.newaxis], y_test[:, np.newaxis])), columns=['Prediction', 'Real'])


# In[53]:


final_df = pd.DataFrame(np.hstack((y_test_pred_2[:, np.newaxis], y_test[:, np.newaxis])), columns=['Prediction', 'Real'])


# In[54]:


print(f'MAE: {mean_absolute_error(final_df["Prediction"],final_df["Real"])}')
print(f'MSE: {mean_squared_error(final_df["Prediction"],final_df["Real"])}')


# In[55]:


print(f'MAE: {mean_absolute_error(final_df["Prediction"],final_df["Real"])}')
print(f'MSE: {mean_squared_error(final_df["Prediction"],final_df["Real"])}')


# In[56]:


feature_importance = bc_rf.feature_importances_

# If x_train_scaled is a NumPy array, generate feature names
feature_names = [f'feature_{i}' for i in range(x_train_scaled.shape[1])]

# Create a DataFrame for easy plotting
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.title('Feature Importance in Random Forest Model')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top
plt.show()


# In[57]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

# Optimized dataset class
class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Neural network model
class ANNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Convert pandas DataFrame/Series to numpy arrays if necessary
x_train_scaled = x_train_scaled.to_numpy() if isinstance(x_train_scaled, pd.DataFrame) else x_train_scaled
y_train = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train

# Check that the features and targets have the same length
assert len(x_train_scaled) == len(y_train), "Mismatch between features and targets length!"

# Create dataset and dataloader
train_dataset = RegressionDataset(x_train_scaled, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)

# Model, loss function, optimizer
input_size = x_train_scaled.shape[1]
hidden_size = 64
output_size = 1
model = ANNModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# More efficient training loop with early stopping
num_epochs = 50
best_loss = float('inf')
patience = 5
counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad(set_to_none=True)  # More efficient than just zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss/len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    # Early stopping to save computation time
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print("Training complete!")


# In[58]:


print(x_train_scaled.shape)
print(y_train.shape)


# In[59]:


print(x_test_scaled.shape)
print(y_test.shape)


# In[60]:


import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Assuming x_test_scaled and y_test are your test data

# Convert pandas DataFrame/Series to numpy arrays if necessary
x_test_scaled = x_test_scaled.to_numpy() if isinstance(x_test_scaled, pd.DataFrame) else x_test_scaled
y_test = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test

# Ensure the model is in evaluation mode
model.eval()

with torch.no_grad():  # No need to calculate gradients during testing
    # Convert the test features to tensor
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
    
    # Make predictions
    y_pred_tensor = model(x_test_tensor)
    
    # Convert predictions back to numpy array
    y_pred = y_pred_tensor.squeeze().numpy()

# Compute MSE and MAE
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Visualize predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Perfect Fit')  # Line for perfect prediction
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()


# In[61]:


import matplotlib.pyplot as plt

# Plot real vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Perfect Fit')  # Line for perfect prediction
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()


# In[62]:


fig, ax = plt.subplots(figsize=(20, 5))
plt.plot(y_test, label="Actual Values", color="blue")
plt.plot(y_pred, label="Predicted Values", color="red")
ax.set_xlim([100,200])
plt.xlabel("Data Points")
plt.ylabel("Values")
plt.title("Predicted vs. Actual Values Over Time")
plt.legend()
plt.show()


# In[ ]:



