
# Life Insurance Risk Model with XGBoost on AWS SageMaker
In this tutorial we will revisit the Kaggle competition [Prudential LIfe Insurance Assessment](https://www.kaggle.com/c/prudential-life-insurance-assessment) and re-create  the solution published by [Anton Laptiev](https://github.com/AntonUBC/Prudential-Life-Insurance-Assessment) using AWS SageMaker. 

## Create the Model
The first step is to develop and train a model and host it in a SageMaker endpoint which we will use to call predictions. 

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

# Display options
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
### Train the Model
1. Specify the Training AMI for SageMaker's XGBoost:
```python
from sagemaker.amazon.amazon_estimator import get_image_uri
container = get_image_uri(boto3.Session().region_name, 'xgboost')
```
2. Create the Training Job:
This is where you specify the model output location, resource configuration, and hyperparameters. 
```python
job_name = 'prudential-xgboost-demo-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print("Training job", job_name)

create_training_params = \
{
    "AlgorithmSpecification": {
        "TrainingImage": container,
        "TrainingInputMode": "File"
    },
    "RoleArn": role,
    "OutputDataConfig": {
        "S3OutputPath": "s3://{}/{}/model/".format(bucket, prefix),
    },
    "ResourceConfig": {
        "InstanceCount": 1,
        "InstanceType": "ml.m4.4xlarge",
        "VolumeSizeInGB": 20
    },
    "TrainingJobName": job_name,
    "HyperParameters": {
        "num_class":"8",
        "eta":"0.003",
        "gamma":"1.2",
        "max_depth":"6",
        "min_child_weight":"2",
        "max_delta_step":"0",
        "subsample":"0.6",
        "colsample_bytree":"0.35",
        "scale_pos_weight":"1.5",
        "silent":"1",
        "seed":"1301",
        "lambda":"1",
        "alpha":"0.2",
        "objective": "multi:softmax",
        "eval_metric": "merror",
        "num_round": "4269"
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 60 * 60
    },
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri":  "s3://{}/{}/train/".format(bucket, prefix),
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "csv",
            "CompressionType": "None"
        },
        {
            "ChannelName": "validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://{}/{}/val/".format(bucket, prefix),
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "csv",
            "CompressionType": "None"
        }
    ]
}
```
3. Run the Training Job:
```python
%%time

region = boto3.Session().region_name
sm = boto3.client('sagemaker')

sm.create_training_job(**create_training_params)

status = sm.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
print(status)
sm.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=job_name)
if status == 'Failed':
    message = sm.describe_training_job(TrainingJobName=job_name)['FailureReason']
    print('Training failed with the following error: {}'.format(message))
    raise Exception('Training job failed')  
```

### Host the Model
```python
model_name=job_name + '-mdl'
hosting_container = {
    'Image': container,
    'ModelDataUrl': sm.describe_training_job(TrainingJobName=job_name)['ModelArtifacts']['S3ModelArtifacts'],
    'Environment': {'this': 'is'}
}

create_model_response = sm.create_model(
    ModelName=model_name,
    ExecutionRoleArn=role,
    PrimaryContainer=hosting_container)

print(create_model_response['ModelArn'])
print(sm.describe_training_job(TrainingJobName=job_name)['ModelArtifacts']['S3ModelArtifacts'])

```
### Configue & Launch Endpoint
```python
endpoint_config_name = 'prudential-demo-EndpointConfig-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print(endpoint_config_name)
create_endpoint_config_response = sm.create_endpoint_config(
    EndpointConfigName = endpoint_config_name,
    ProductionVariants=[{
        'InstanceType':'ml.m4.xlarge',
        'InitialInstanceCount':1,
        'InitialVariantWeight':1,
        'ModelName':model_name,
        'VariantName':'AllTraffic'}])

print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])

%%time
import time

endpoint_name = 'prudential-demo-endpoint-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print(endpoint_name)
create_endpoint_response = sm.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name)
print(create_endpoint_response['EndpointArn'])

resp = sm.describe_endpoint(EndpointName=endpoint_name)
status = resp['EndpointStatus']
print("Status: " + status)

while status=='Creating':
    time.sleep(60)
    resp = sm.describe_endpoint(EndpointName=endpoint_name)
    status = resp['EndpointStatus']
    print("Status: " + status)

print("Arn: " + resp['EndpointArn'])
print("Status: " + status)
```
### Evaluate Model
1. Create Helper Functions:
```python
# Simple function to create a csv from our numpy array

def np2csv(arr):
    print(arr)
    print(type(arr))
    csv = io.BytesIO()
    np.savetxt(csv, arr, delimiter=',', fmt='%g')
    return csv.getvalue().decode().rstrip()

# Function to generate prediction through sample data

def do_predict(data, endpoint_name, content_type):
    
    payload = np2csv(data)
    print(payload)
    print(type(payload))
    response = runtime.invoke_endpoint(EndpointName=endpoint_name, 
                                   ContentType=content_type, 
                                   Body=payload)
    result = response['Body'].read()
    result = result.decode("utf-8")
    result = result.split(',')
    preds = [float((num)) for num in result]
    return preds

# Function to iterate through a larger data set and generate batch predictions

def batch_predict(data, batch_size, endpoint_name, content_type):
    items = len(data)
    arrs = []
    
    for offset in range(0, items, batch_size):
        if offset+batch_size < items:
            datav = data.iloc[offset:(offset+batch_size),:].as_matrix()
            results = do_predict(datav, endpoint_name, content_type)
            arrs.extend(results)
        else:
            datav = data.iloc[offset:items,:].as_matrix()
            arrs.extend(do_predict(datav, endpoint_name, content_type))
        sys.stdout.write('.')
    return(arrs)
```
2. Read in Data & Generate Predictions
```python
data_train = pd.read_csv("formatted_train.csv", sep=',', header=None) 
data_val = pd.read_csv("formatted_val.csv", sep=',', header=None) 

runtime= boto3.client('runtime.sagemaker')

preds_train_xgb = batch_predict(data_train.iloc[:, 1:], 1000, endpoint_name, 'text/csv')
preds_val_xgb = batch_predict(data_val.iloc[:, 1:], 1000, endpoint_name, 'text/csv')
```
3. Score the Model:
```python
train_labels = data_train.iloc[:,0];
val_labels = data_val.iloc[:,0];

Training_f1 = metrics.f1_score(train_labels, preds_train_xgb, average=None)
Validation_f1= metrics.f1_score(val_labels, preds_val_xgb, average=None)

print("Average Training F1 Score", np.average(Training_f1))
print("Average Validation F1 Score", np.average(Validation_f1))
    
for risk, tscore, vscore in zip(range(1,9),Training_f1, Validation_f1):
    ts = np.round(tscore,2)
    vs = np.round(vscore,2)
    print(risk, ts , '\t|' , vs)
```
### Save Predictions to S3
```python
final = pd.concat([val_labels, pd.DataFrame(preds_val_xgb)], axis=1)
preds_file = "prudential-demo-predictions.csv"
final.to_csv(preds_file, sep=',', header=False, index=False)
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'preds/', preds_file)).upload_file(preds_file)
```

# How to Interact with the Model
There are several ways to interact with a model, and what method you ultimately decide to use depends largely on the industry and application. The model we created in this tutorial is meant to predict a Life Insurance applicant's risk score and would be used to assist in the underwriting process. Consider the following scenarios:
1. __Web Form Application:__ ([how to make a Chalice Webapp that calls a model Endpoint]()) 
   * Scenario #1) We can call the model directly after the applicant submits their web-based application. In this scenario we would use an AWS Lambda function to transform the webform into a format that the model can accept. Then we would call the model and add the model's prediction to the applicant's data before it reaches the applicant database. When application is queried by the underwriter, they can see the score the model assigned to that applicant.
   * Scenario #2) Similar to above, we can also set up a rule where applicants recieving a specific score are routed to the appropriate next step. For example, a high risk applicant may be sent to an underwriter before the applicant is rejected to verify that there was not an error. Low risk applicants can be auto-approved and enrolled immediately, skipping the underwriting process. 
   * Scenario #3) Another approach would be to immediately alert the applicant of the model's assessment, letting the applicant know if their application has a probability of being rejected. This gives the applicant the opportunity to provide additional information or change the product that they are applying for. 
1. __Chat Bot:__
   *  Turn the webform into a chatbot! This gives the applicant the ability to answer questions interactively. A chat bot can easily be integrated into any of the above scenarios. \
3. __Manual CSV Upload:__
   * The most basic way to call the model is to use a simple CSV upload. A single applicant, or a batch of applications can receive scores via a CSV output. 


