import torch
import os
import pandas as pd

os.makedirs(os.path.join('..','data'), exist_ok=True)
dataset  =  os.path.join('..','data','house_tiny.csv')

with open(dataset, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
    
data = pd.read_csv(dataset)
# print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs.iloc[:, 0] = inputs.iloc[:, 0].fillna(inputs.iloc[:, 0].mean())
# print(inputs)
inputs_nodummy = inputs.copy()
inputs_ndm = pd.get_dummies(inputs_nodummy, dummy_na=False)
# print(inputs)
inputs = pd.get_dummies(inputs, dummy_na=True)
# print(inputs)

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
# print(X, y)

dataset_ex = os.path.join('..', 'data', '2_2_test.csv')

with open(dataset_ex, 'w') as f:
    f.write('NumRooms,Alley,Price,Garage,Pool\n')
    f.write('3,Pave,150000,NA,NA\n')
    f.write('NA,Pave,120000,1,Yes\n')
    f.write('5,NA,200000,2,NA\n')
    f.write('2,Pave,130000,NA,No\n')
    f.write('NA,NA,110000,1,NA\n')
    f.write('4,Pave,180000,2,No\n')
    f.write('NA,Pave,125000,NA,NA\n')
    f.write('6,NA,210000,3,Yes\n')
    
data = pd.read_csv(dataset_ex)
# print(data)

input = pd.concat([data.iloc[:, 0:2], data.iloc[:, 3:5]], axis=1)
print(input)

output = data.iloc[:, 2]
# print(output)

# Remove the column with most NAs
na_counts = input.isna().sum()
# print(na_counts)
col_to_drop = na_counts.idxmax()
# input = input.drop(columns=[col_to_drop])

# Fill numeric columns' NAs with average
for col in input.select_dtypes(include=['float64', 'int64']).columns:
    input[col] = input[col].fillna(input[col].mean())

# For columns with Yes/No, fill NAs with 'No'
for col in input.columns:
    if set(input[col].dropna().unique()) == {'Yes', 'No'}:
        input[col] = input[col].fillna('No')
        
for col in input.columns:
    if len(set(input[col].dropna().unique())) == 1:
        input[col] = pd.get_dummies(input[col], dummy_na=False)
        
print(input)