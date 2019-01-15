
# Life Insurance Risk Model with XGBoost on AWS SageMaker
In this tutorial we will revisit the Kaggle competition [Prudential LIfe Insurance Assessment](https://www.kaggle.com/c/prudential-life-insurance-assessment) and re-create  the solution published by [Anton Laptiev](https://github.com/AntonUBC/Prudential-Life-Insurance-Assessment) using AWS SageMaker. 

### Launch an AWS SageMaker Jupyter Notebook Instance and create an IAM Role
1. Click on “Create Notebook Instance”.
2. Name the instance “life-insurance-risk” and keep the instance type as “ml.t2.medium”
3. Create a new IAM role.
4. Select “None” for the S3 buckets.
5. Select “Create role”.
6. Click on the new IAM Role you just created. It will take you to the Role Summary Page in the IAM console. Attach the following policies to the new role you just created:
 - AmazonS3FullAccess
 - AmazonEC2ContainerRegistryFullAccess
7. Keep all other settings as default and select “Create notebook instance” .
8. When it is done launching, open the Jupyter instance and create a ‘conda-python3’ notebook.

### Configure the Notebook
1. Name the notebook “xgboost-demo”
2. Define the Execution Role and S3 Bucket Locations where we will store our data and model artifacts.
```python
import os
import boto3
import time
import re
from sagemaker import get_execution_role

role = get_execution_role()
# s3 bucket where you will copy data and model artificats
bucket = 'prudential-xgboost' 
# place to upload training files within the bucket
prefix = 'demo'  
```
### Import Additional Libraries
```python
import numpy as np
import pandas as pd
import pickle                               # For serializing and saving the model
from IPython.display import display         # For displaying outputs in the notebook
from time import gmtime, strftime           # For labeling SageMaker models, endpoints, etc.
import sys                                  # For writing outputs to notebook
import json                                 # For parsing hosting output
import io                                   # For working with stream data
import sagemaker.amazon.common as smac      # For protobuf data format

# libraries for the model
from sklearn import preprocessing           
from sklearn import metrics 
from sklearn.model_selection import StratifiedKFold    
```
__Set Display Options__
```python
pd.set_option('display.max_columns', 128)     # Make sure we can see all of the columns
pd.set_option('display.max_rows', 6)         # Keep the output on
```

### Load Data
The data provided from the Kaggle Competition has already been split into a train.csv and test.csv files. The train.csv file has labeled attributes wich include a response column. The test.csv file does not have the response column. For the purposes of training and scoring we will only be working with the train.csv file. You can download the file locally [here](https://www.kaggle.com/c/prudential-life-insurance-assessment/data) and upload it to your Jupyter instance. Alternativelly, you can use the [Kaggle API](https://www.kaggle.com/docs/api) if you have your credentials. 

__Local Download__
1. Go to Jupyter Home.
2. Select "Upload" and find the train.csv file on your computer. Back in your Jupyter Home confirm the upload and return back to the notebook. 

__API Download__
1. Go to Jupyter Home
2. Select 'new' --> 'Terminal'
3. In the terminal:
  - update pip: `pip install --upgrade pip`
  - install kaggle: `pip install kaggle`
  - create a kaggle json file: `touch /home/ec2-user/.kaggle/kaggle.json`
  - insert your credentials in json format: `echo '{"username":"<your-username>","key":"<our-api-key>"}' >> /home/ec2-user/.kaggle/kaggle.json`
 4. Return to the SageMaker home folder: `cd /home/ec2-user/SageMaker`
 5. Run the Kaggle API Download: `kaggle competitions download -c prudential-life-insurance-assessment` 
 6. Unzip the train.csv.zip file: `unzip train.csv.zip`
 7. Close the terminal and return to the notebook. 
 
 Once you are back in the notebook, upload the train.csv file into a pandas DataFrame and inspect the first few rows.
 ```python
 df = pd.read_csv('train.csv')
 df.head()
 ```
 
 ### Transform and Format Data
 The data needs to go through a cleaning and transformation process before we can run a model on it. In addition, AWS requires that the response variables are located in the first column and that the CSV has no headers or index. 
 1. Format and move the 'Response' column:
 ```python
 df['Response'] = df['Response'].astype(int)
target = df.pop('Response').values
le = preprocessing.LabelEncoder()
y = le.fit_transform(target)
df.insert(0,'Response', y)
```
2. Create a data-cleaning function and run the function on the training data. 
```python
def format_df(df):
    # format data
    df['Product_Info_2_char'] = df.Product_Info_2.str[0]
    df['Product_Info_2_num'] = df.Product_Info_2.str[1]

    # factorize categorical variables
    df['Product_Info_2'] = pd.factorize(df['Product_Info_2'])[0]
    df['Product_Info_2_char'] = pd.factorize(df['Product_Info_2_char'])[0]
    df['Product_Info_2_num'] = pd.factorize(df['Product_Info_2_num'])[0]

    # feature engineering
    df['BMI_Age'] = df['BMI'] * df['Ins_Age']

    med_keyword_columns = df.columns[df.columns.str.startswith('Medical_Keyword_')]
    df['Med_Keywords_Count'] = df[med_keyword_columns].sum(axis=1)

    # encode missing values
    df.fillna(-1, inplace=True)

    # drop irrevelent attributes
    df.drop(['Id', 'Medical_History_10','Medical_History_24'], axis=1, inplace=True)
```
 ### Split the Data: 80/20 Train-Validation Split
```python
train_list = np.random.rand(len(df)) < 0.8    # 80% train / 20% test
train_data = df[train_list]
val_data = df[~train_list]
```
### Save & Upload to S3
```python
train_file = 'formatted_train.csv'
val_file = 'formatted_val.csv'
train_data.to_csv(train_file, sep=',', header=False, index=False) # save training data 
val_data.to_csv(val_file, sep=',', header=False, index=False) # save validation data

boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train/', train_file)).upload_file(train_file)
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'val/', val_file)).upload_file(val_file)
```


 
    




