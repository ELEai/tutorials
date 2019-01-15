
# Life Insurance Risk Model with XGBoost on AWS SageMaker
In this tutorial we will revisit the Kaggle competition [Prudential LIfe Insurance Assessment](https://www.kaggle.com/c/prudential-life-insurance-assessment) and re-create  the solution published by [Anton Laptiev](https://github.com/AntonUBC/Prudential-Life-Insurance-Assessment) using AWS SageMaker. 

### Launch an AWS SageMaker Jupyter Notebook Instance and create an IAM Role
1. Click on “Create Notebook Instance”.
2. Name the instance “life-insurance-risk” and keep the instance type as “ml.t2.medium”
3. Create a new IAM role.
4. Select “None” for the S3 buckets.
5. Select “Create role”.
6. Click on the new IAM Role you just created. It will take you to the Role Summary Page in the IAM console. Attach the following policies to this new role. 
 - AmazonS3FullAccess
 - AmazonEC2ContainerRegistryFullAccess
7. Keep all other settings as default and select “Create notebook instance” .
8. When it is done launching, open the Jupyter instance and create a ‘conda-python3’ notebook.

### Configure the Notebook
1. Name the notebook “xgboost-demo”
2. Define the Execution Role and S3 Bucket Locations where we will store our data and model artifacts.
```
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
```
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

### Load Data
The data provided from the Kaggle Competition has already been split into a train.csv and test.csv files. The train.csv file has labeled attributes wich include a response column. The test.csv file does not have the response column. For the purposes of training and scoring we will only be working with the train.csv file. You can download the file locally [here](https://www.kaggle.com/c/prudential-life-insurance-assessment/data) and upload it to your Jupyter instance. Alternativelly, you can use the [Kaggle API](https://www.kaggle.com/docs/api) if you have your credentials. 

* _Local Download_
1. Go to Jupyter Home.
2. Select "Upload" and find the train.csv file on your computer. Back in your Jupyter Home confirm the upload and return back to the notebook. 

* _API Download_
1. Go to Jupyter Home
2. Select 'new' --> 'Terminal'
3. In the terminal:
  - update pip: `pip install --upgrade pip`
  - install kaggle: `pip install kaggle`
  - create a kaggle json file: `touch /home/ec2-user/.kaggle/kaggle.json`
  - insert your credentials in json format: `echo '{"username":"schmidtbit","key":"d676f7b9bd3f2611834bf8353cb4e5f4"}' >> /home/ec2-user/.kaggle/kaggle.json`
 4. Return to the SageMaker home folder: `cd /home/ec2-user/SageMaker`
 5. Run the Kaggle API Download: `kaggle competitions download -c prudential-life-insurance-assessment` 
 6. Unzip the train.csv.zip file: `unzip train.csv.zip`
 7. Close the terminal and return to the notebook. 
 




