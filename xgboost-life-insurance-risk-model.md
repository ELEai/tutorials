
# Life Insurance Risk Model with XGBoost on AWS SageMaker
In this tutorial we will revisit the Kaggle competition Prudential LIfe Insurance Assessment and re-create  the solution published by Anton Laptiev using AWS SageMaker. 

__Launch an AWS SageMaker Jupyter Notebook Instance and create an IAM Role.__
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

__Configure the Notebook.__ 
1. Name the notebook “xgboost-demo”
```
import os
import boto3
import time
import re
from sagemaker import get_execution_role

role = get_execution_role()

# Now let's define the S3 bucket we'll used for the remainder of this example.

bucket = 'sage-bucket-as' #  enter your s3 bucket where you will copy data and model artifacts
prefix = 'prudential/xgboost-single'  # place to upload training files within the bucket
```
2. Define the Execution Role and S3 Bucket Locations where we will store our data and model artifacts
3. Import Libraries.



