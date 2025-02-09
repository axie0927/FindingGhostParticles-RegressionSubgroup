#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error


# In[2]:


# I don't know if we can use the GPUs on DSMLP to utilize the CUDA function of Pytorch
# So do not set epoch too high in order to have a faster training process.


# In[3]:


# create varaibles that holds a dataframe
cwd = os.getcwd()
data_dir = os.path.join(cwd, "Data/")
data_files = [f for f in os.listdir(str(data_dir)) if f.endswith('csv')]

data_train_name = [f for f in data_files if 'TRAIN' in f]
data_test_name = [f for f in data_files if 'TEST' in f]

train_df = pd.read_csv(os.path.join(data_dir,data_train_name[0]))
test_df = pd.read_csv(os.path.join(data_dir,data_test_name[0]))
train_df = train_df.dropna()
test_df = test_df.dropna()


# feature selection and renaming

def manipulate_cols(df):
    boolean_col = ['highavse','lowavse','truedcr','lq']
    useless_col = ['tdrift50','tdrift10']
    new_df = df.drop(columns=boolean_col+['id']+useless_col)
    new_df.columns = [col.strip().replace(' ','_') for col in new_df.columns]
    return new_df

train_df = manipulate_cols(train_df)
test_df = manipulate_cols(test_df)




# Reshaping for consistency
X_train = train_df.drop(columns=['energylabel']).values
X_test = test_df.drop(columns=['energylabel']).values
y_train = train_df['energylabel'].values.reshape(-1,1)
y_test = test_df['energylabel'].values.reshape(-1,1)


# In[4]:


#####################################################################################
# Extremely Important: REMOVE "pass" and UNCOMMENT the codes below to RUN it !!!!!  #
#####################################################################################

#---------------------------------------------------------------------------
# NaN Will Ruin the NN! So make sure it desn't contain any NaN in any corner!
#---------------------------------------------------------------------------

"""
features = [
    'tdrift', 'rea', 'dcr', 'peakindex', 'peakvalue', 'tailslope',
    'currentamp', 'lfpr', 'lq80', 'areagrowthrate', 'inflection_point',
    'risingedgeslope'
]
print(train_df[features].std())
print("Any NaNs in X_train?", np.isnan(X_train).any())
print("Any NaNs in X_test?", np.isnan(X_test).any())
print('\n')
print(train_df.isnull().sum())
print(test_df.isnull().sum())
"""
pass


# In[5]:


# Standardization

scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler() # VERY IMPORTANT! We also need to transform it back to original after prediction!
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)


# In[6]:


# Convert to Pytorch Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# In[7]:


# Define dataloader

class NPDL(Dataset): # Neutrino Physics Deel Learning
    def __init__(self,X,y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]
    
train_NPDL = NPDL(X_train_tensor,y_train_tensor)
test_NPDL = NPDL(X_test_tensor, y_test_tensor)

batch_size = 32
train_loader = DataLoader(train_NPDL, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_NPDL, batch_size=batch_size, shuffle=False)


# In[8]:


# Public Static int main!

class SuperPredictor(nn.Module):
    def __init__(self, input_size):
        super(SuperPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1) 
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
input_size = 12 # the number of our features
model = SuperPredictor(input_size)

accuracy = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    


# In[9]:


# Set epoch to 100 is good, but my computer is trash, you can do it on DSMLP.
num_epochs = 20 # Change this later according to our Computational Power!!  
for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()           
        outputs = model(X_batch)          
        loss = accuracy(outputs, y_batch) 
        loss.backward()                 
        optimizer.step()                

        running_loss += loss.item() * X_batch.size(0)
    epoch_loss = running_loss / len(train_NPDL)
    #if (epoch+1) % 10 == 0:
    print(f"Epoch [{epoch+1}/{num_epochs}], Scaled Loss(standardization): {epoch_loss:.4f}")
        

model.eval()  # set the model to evaluation mode
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = accuracy(predictions, y_test_tensor)
    print(f"Test Loss(Scaled): {test_loss.item():.4f}")

      


# In[10]:


predictions_original = scaler_y.inverse_transform(predictions.numpy())
y_test_original = scaler_y.inverse_transform(y_test_tensor.numpy())


MSE = mean_squared_error(y_test_original, predictions_original)
MAE = mean_absolute_error(y_test_original, predictions_original)
r2 = r2_score(y_test_original, predictions_original)

print(f"MSE: {MSE}, Average Residuals: {MAE}, Variance Explained: {r2}")


# Here we can see that MSE is the lowest out of all our models, this may be our best performing model.
