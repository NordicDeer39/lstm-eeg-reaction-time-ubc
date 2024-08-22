

#%%


import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, BatchNormalization, Activation, Flatten, TimeDistributed, AveragePooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from scikeras.wrappers import KerasRegressor




#%%


task_gvssham = sio.loadmat('C:/Users/Microsoft/EEG_Hackathon/task_gvssham.mat') 
sham_data = sio.loadmat('C:/Users/Microsoft/EEG_Hackathon/shamdata_tlgo.mat')

#%%

Dataset = np.zeros((27,2000,1))

for subj in range(0,22):
    Dataset = np.concatenate((Dataset, sham_data['shamhceeg'][0,subj]), axis=2)

for subj in range(0,20):
    Dataset = np.concatenate((Dataset, sham_data['shampd1eeg'][0,subj]), axis=2)

for subj in range(0,20):
    Dataset = np.concatenate((Dataset, sham_data['shampd2eeg'][0,subj]), axis=2)

Dataset = Dataset[:,:,1:621]

react_time = 12
target = np.zeros(1)

for subj in range(0,22):
    target = np.concatenate((target, task_gvssham['hcoffmed'][0,subj][:,react_time]), axis=0)

for subj in range(0,20):
    target = np.concatenate((target, task_gvssham['pdoffmed'][0,subj][:,react_time]), axis=0)

for subj in range(0,20):
    target = np.concatenate((target, task_gvssham['pdonmed'][0,subj][:,react_time]), axis=0)

target = target[1:621]

#%%



# from scipy.interpolate import griddata
# # Handle NaN values by spline interpolation
# def spline_interpolation(arr):
#     for i in range(arr.shape[0]):
#         for j in range(arr.shape[2]):
#             if np.isnan(arr[i, :, j]).any():
#                 # Indices of non-NaN values
#                 not_nan = ~np.isnan(arr[i, :, j])
#                 x = np.where(not_nan)[0]
#                 y = arr[i, not_nan, j]
#                 if len(x) > 1:  # Ensure there are at least two points to interpolate
#                     # Indices of NaN values
#                     nan_indices = np.isnan(arr[i, :, j])
#                     arr[i, nan_indices, j] = griddata(x, y, np.where(nan_indices)[0], method='cubic')
#                 else:
#                     # If there are less than two points, fill NaNs with 0 or any constant value
#                     arr[i, :, j] = np.nan_to_num(arr[i, :, j], nan=0.0)
#     return arr

# Dataset = spline_interpolation(Dataset)

indices1 = np.array(np.where(~np.isnan(target))).squeeze()
indices2 = np.array(np.where(np.isnan(Dataset[0,0,:]))).squeeze()
#%%
indices = np.setdiff1d(indices1,indices2)
#%%
Dataset = Dataset[:,:,indices]
target = target[indices]


# for i in range(Dataset.shape[2]):
#     for j in range(Dataset.shape[0]):
#         m = np.mean(Dataset[j, :, i])
#         s = np.std(Dataset[j, :, i])
#         Dataset[j, :, i] = (Dataset[j, :, i] - m) / s
    
    
    
#%%

Dataset = np.transpose(Dataset, (2, 1, 0))

# Normalize the data


#%%
x_train, x_test, y_train, y_test = train_test_split(Dataset, target, test_size=0.2, random_state=42)

print(f'X_train shape: {x_train.shape}')  
print(f'y_train shape: {y_train.shape}')  
print(f'X_test shape: {x_test.shape}')    
print(f'y_test shape: {y_test.shape}')   

#%%


input_x = np.zeros((11853,2000,1))


for i in range(439):
   
    slice_matrix = x_train[i, :, :].T
    
    input_x[i*27:(i+1)*27,:,0] = slice_matrix
#%%

input_y = np.repeat(y_train, 27)
st = np.std(y_train)*0.05
noise = np.random.normal(0, st, input_y.shape)
input_y_noisy = input_y + noise


#%%
target_x = np.zeros((2970,2000,1))

for i in range(110):
    slice = x_test[i, :, :].T
    target_x[i*27:(i+1)*27,:,0] = slice
    
print(target_x.shape)
    
#%%

target_y = np.repeat(y_test, 27)
st = np.std(y_train)*0.02
noise = np.random.normal(0, st, target_y.shape)
target_y_noisy = target_y + noise
print(target_y.shape)

#%%

target_y_noisy2 = np.zeros([2970,1])
input_y_noisy2 = np.zeros([11853,1])
target_y_noisy2[:,0]=target_y_noisy
input_y_noisy2[:,0]=input_y_noisy

#%%


#MODEL



model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(2000, 1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(LSTM(50, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dense(1)) 


model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

history = model.fit(input_x, input_y_noisy, epochs=1, batch_size=64, validation_split=0.2)
#%% 
loss = model.evaluate(target_x, target_y_noisy)
print(f"Test Loss: {loss}")
#%%
y_pred = model.predict(target_x)
#%%
print(f"Predictions: {y_pred[:10]}") 


#%%

#here1
model3 = Sequential()
model3.add(LSTM(units=32, return_sequences=True, input_shape=(2000, 1)))
model3.add(LSTM(units=16))
model3.add(Dense(3))
model3.summary()

#%%
model3.compile(optimizer='adam', loss='mean_squared_error', metrics = ['mean_squared_error'])
history3=model3.fit(input_x, input_y_noisy, epochs=1, batch_size=1, validation_split=0.2)
#%%
# Run Completed, Train loss = 79, Test loss = 115 --> Possibly overfitted
# Loss Curve plotted, 10 epochs even half of it was enough, 
model4 = Sequential()
model4.add(LSTM(units=24, return_sequences=True, input_shape=(2000, 27)))
# model4.add(LSTM(units=16, return_sequences=True))
# model4.add(TimeDistributed(Dense(1)))  # Add TimeDistributed Dense layer with 10 units (example)
model4.add(AveragePooling1D(pool_size=10))
model4.add(Flatten())
model4.add(Dense(1))
model4.summary()
model4.compile(optimizer='adam', loss='mean_squared_error', metrics = ['mean_squared_error'])
history4=model4.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.2)
#%%

y_pred = model4.predict(x_test)
loss , mae= model4.evaluate(x_test, y_test)
print(f'Test MAE: {mae}')
print(f'Test loss: {loss}')
trloss = history4.history['loss']
valloss = history4.history['val_loss']
plt.figure()
plt.plot(trloss,label='Training loss')
plt.plot(valloss, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#%%

# Run Completed, Train loss = 106, Test loss = 118 --> Possibly overfitted
# Loss Curve plotted, 10 epochs was enough, 
model5 = Sequential()
model5.add(LSTM(units=16, return_sequences=True, input_shape=(2000, 27)))
# model4.add(LSTM(units=16, return_sequences=True))
# model4.add(TimeDistributed(Dense(1)))  # Add TimeDistributed Dense layer with 10 units (example)
model5.add(AveragePooling1D(pool_size=10))
model5.add(Flatten())
model5.add(Dense(1))
model5.summary()
model5.compile(optimizer='adam', loss='mean_squared_error', metrics = ['mean_squared_error'])
history5=model5.fit(x_train, y_train, epochs=10, batch_size=16, validation_split=0.2)
#%%

y_pred = model5.predict(x_test)
loss , mae= model5.evaluate(x_test, y_test)
print(f'Test MAE: {mae}')
print(f'Test loss: {loss}')
trloss = history5.history['loss']
valloss = history5.history['val_loss']
plt.figure()
plt.plot(trloss,label='Training loss')
plt.plot(valloss, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%%

# 
model6 = Sequential()
model6.add(LSTM(units=16, return_sequences=True, input_shape=(2000, 27)))
model6.add(Dropout(0.2))
model6.add(AveragePooling1D(pool_size=15))
model6.add(Flatten())
model6.add(Dense(1))
model6.summary()
#%%
model6.compile(optimizer='adam', loss='mean_squared_error', metrics = ['mean_squared_error'])
history6=model6.fit(x_train, y_train, epochs=10, batch_size=16, validation_split=0.2)
#%%

y_pred = model6.predict(x_test)
loss , mae= model6.evaluate(x_test, y_test)
print(f'Test MAE: {mae}')
print(f'Test loss: {loss}')
trloss = history6.history['loss']
valloss = history6.history['val_loss']
plt.figure()
plt.plot(trloss,label='Training loss')
plt.plot(valloss, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%% 
#here2
model = Sequential()
model.add(LSTM(units=32, return_sequences=True,activation='relu', input_shape=(2000, 1)))
model.add(LSTM(units=16))
model.add(Dense(1,activation='sigmoid'))
model.summary()
#%%
#here2
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='mean_absolute_error', 
              metrics=['mae'])

model.fit(input_x, input_y_noisy, epochs=5, batch_size=16, validation_split=0.2)

loss, mae = model.evaluate(target_x, target_y_noisy)
print(f'Test MAE: {mae}')



#%%
model2 = Sequential()
model2.add(LSTM(units=32, return_sequences=True, input_shape=(2000, 1)))
model2.add(LSTM(units=16))
model2.add(Dense(1)) 
model2.summary()
#%%

model2.compile(optimizer='adam', loss='mean_squared_error')

model2.fit(input_x, input_y_noisy, epochs=20, batch_size=64, validation_split=0.2)
#%% 
loss = model.evaluate(x_test, y_test)
#%%
print(f"Test Loss: {loss}")
#%%
y_pred = model.predict(x_test)
#%%


def create_model(optimizer='adam', lstm_units=64):
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(2000, 1), return_sequences=True))
    model.add(LSTM(lstm_units))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mean_absolute_error'])
    return model

model = KerasRegressor(build_fn=create_model, verbose=0)

#GridSearchCV
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'lstm_units': [32, 64, 100],
    'batch_size': [16, 32, 64],
    'epochs': [15, 30]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)
grid_result = grid.fit(input_x, input_y_noisy)


print(f"Best Parameters: {grid_result.best_params_}")
print(f"Best Score: {grid_result.best_score_}")


best_model = grid_result.best_estimator_.model
loss, mae = best_model.evaluate(target_x, target_y_noisy)
print(f'Test MAE: {mae}')








