#IMPORTS
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython import display
display.set_matplotlib_formats('svg')
import pickle

#Reading CSV File
df = pd.read_csv('model_python\cancer patient data sets.csv')

#Low --> 0 ; Medium --> 1 ; High --> 2
Range_Mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df['Level'] = df['Level'].map(Range_Mapping)

#Dividing X and Y
x = torch.tensor(df.iloc[: , 2:25].values).float()
y = torch.tensor(df.iloc[: , 25].values).float()

#Checking If # Low, Medium, High Is Around The Same
count_0 = torch.sum(torch.eq(y, 0)).item()
count_1 = torch.sum(torch.eq(y, 1)).item()
count_2 = torch.sum(torch.eq(y, 2)).item()

#Dividing Training & Testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Changing Shape of Y
y_train = y_train.view(-1).long()
y_test = y_test.view(-1).long()

#--------------------------------------------------------------

#Checking If #Low, Medium, High Is Around The Same -- Training
count_0 = torch.sum(torch.eq(y_train, 0)).item()
count_1 = torch.sum(torch.eq(y_train, 1)).item()
count_2 = torch.sum(torch.eq(y_train, 2)).item()

#Checking If #Low, Medium, High Is Around The Same -- Testing
count_0 = torch.sum(torch.eq(y_test, 0)).item()
count_1 = torch.sum(torch.eq(y_test, 1)).item()
count_2 = torch.sum(torch.eq(y_test, 2)).item()

#--------------------------------------------------------------

#The Model
ANNmodel = nn.Sequential(
    nn.Linear(23 , 64),
    nn.ReLU(),
    nn.Linear(64 , 64),
    nn.ReLU(),
    nn.Linear(64 , 3),
)


#loss-function
lossfunction = nn.CrossEntropyLoss()

#optimizer
optimizer = torch.optim.SGD(ANNmodel.parameters() , lr = 0.01)

#-----------------------------------------------------------------

Epochs = 10000
Incoming_Losses = []

for i in range(Epochs):

  #front-prop
  Results = ANNmodel(x_train)


  #compute losses
  Losses = lossfunction(Results , y_train)
  Incoming_Losses.append(Losses)

  #back prop
  optimizer.zero_grad()
  Losses.backward()
  optimizer.step()

#----------------------------------------------------------
  
#TESTING

Test_Loss = []

Testing_Results = ANNmodel(x_test)
Losses = lossfunction(Testing_Results, y_test)
Test_Loss.append(Losses)

#Soft-Max Function
Probabilities = nn.functional.softmax(Testing_Results, dim=1)
Prediction = torch.argmax(Probabilities, dim=1)
#print(Prediction)

#------------------------------------------------------------------

#Losses
#print(Incoming_Losses)
#print(Test_Loss)

#-------------------------------------------------------------------

#Checking How Many Wrong
Incorrect = 0
for i in Prediction:
  if Prediction[i] != y_test[i]:
    Incorrect+=1

#print(Incorrect)
    
#--------------------------------------------------------------------

#ACTUAL INPUTS
Age = int(input("Age: "))
Gender = int(input("Gender: "))
Air_Pollution = int(input("Air Pollution: "))
Alcohol_Use = int(input("Alcohol Usage: "))
Dust_Allergy = int(input("Dust Allergy: "))
Occupational_Hazards = int(input("Occupational Hazards: "))
Genetic_Risk = int(input("Genetic Risk: "))
Chronic_Lung_Disease = int(input("Chronic Lung Disease: "))
Balanced_Diet = int(input("Balanced Diet: "))
Obesity = int(input("Obesity: "))
Smoking = int(input("Smoking: "))
Passive_Smoking = int(input("Passive Smoking: "))
Chest_Pain = int(input("Chest Pain: "))
Coughing_Blood = int(input("Coughing of Blood: "))
Fatigue = int(input("Fatigue: "))
Weight_Loss = int(input("Weight Loss:"))
Shortness_Breath = int(input("Shortness Breath: "))
Wheezing = int(input("Wheezing: "))
Swallowing_Difficulty = int(input("Difficulty Swallowing: "))
Clubbing_Finger = int(input("Clubbing of Finger: "))
Frequent_Cold = int(input("Frequent Cold: "))
Dry_Cough = int(input("Dry Cough: "))
Snoring = int(input("Snoring: "))

Inputs = torch.tensor([Age, Gender, Air_Pollution, Alcohol_Use, Dust_Allergy, Occupational_Hazards, Genetic_Risk, Chronic_Lung_Disease, Balanced_Diet, Obesity, Smoking, Passive_Smoking, Chest_Pain, Coughing_Blood, Fatigue, Weight_Loss, Shortness_Breath, Wheezing, Swallowing_Difficulty, Clubbing_Finger, Frequent_Cold, Dry_Cough, Snoring ])
print(Inputs.shape)

Model_Prediction = ANNmodel(Inputs.view(1, -1).float())
#Plain Prediction
print(Model_Prediction)

#Using Soft-Max Function
Probabilities = nn.functional.softmax(Model_Prediction, dim=1)
Prediction = torch.argmax(Probabilities, dim=1)
print(Prediction)


#Saving
pickle.dump(ANNmodel, open('model.pkl' , "wb"))