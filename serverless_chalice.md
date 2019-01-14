## Create a Serverless Chalice App for a SageMaker Model Endpoint
Following this tutorial: https://aws.amazon.com/blogs/machine-learning/build-a-serverless-frontend-for-an-amazon-sagemaker-endpoint/

1. Connect to EC2 Amazon-Linux-2
2. Install updates and packages
  - python3
  - pip
  - chalice
  - boto3
3. Set default AWS Region
4. Start a Chalice project
5. Edit the `app.py` and `requirements.txt` files
6. Deploy the app
7. Note the REST API address
8. create and index.html file that points to the REST API of the Chalice app
9. Create and configure an AWS S3 bucket
10. Upload your index.html file to you new bucket

Notes: ensure your EC2 instance, default AWS Region, S3 bucket, and model are all in the same region. 

__Install and Update Packages__
 - `sudo yum update -y`
 - `sudo amazon-linux-extras install python3`
 - `curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py`
 - `python3 get-pip.py --user`
 - `pip install chalice boto3 --user`

__Set Default Region__
 - `export AWS_DEFAULT_REGION=<region>`

__Create Your Chalice Project__
 - `chalice new-project <project-name>`
 - `cd <project-name>`

__Edit `requirements.txt` and `app.py`__
 - Sample app.py file:
```
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from io import BytesIO
import csv
import sys, os, base64, datetime, hashlib, hmac
from chalice import Chalice, NotFoundError, BadRequestError
import boto3
from urllib.parse import parse_qs

app = Chalice(app_name='my-app')
app.debug = True

sagemaker = boto3.client('sagemaker-runtime')

@app.route('/', methods=['POST'], content_types=['application/x-www-form-urlencoded'])
def handle_data():
    input_text = app.current_request.raw_body
    d = parse_qs(input_text)
    lst = d[b'user_input'][0].decode()
    res = sagemaker.invoke_endpoint(
                    EndpointName='<name-of-SageMaker-Endpoint>',
                    Body=lst,
                    ContentType='text/csv',
                    Accept='Accept'
                )
    return res['Body'].read().decode()[0]

```
 - Update `requirements.txt`: `echo 'boto3' >> requirements.txt`

__Deploy the App__
 - `chalice deploy`
_Copy the Chalice REST API from the output and save for next step_

__Create and Edit an `index.html` file__
 - `touch index.html`
 - `sed -i s@CHALICE_ENDPOINT@<your-chalice-api>@g index.html`

 - vim into the file and make something like this:
```
<html>
<head></head>
<body>
<form method="post" action="<chalice-rest-api">

<input type="text" name="user_input"><br>

<input type="submit" value="Submit">
</form>
</body>
</html>
```

__Create and Configure an S3 Bucket__
 - `aws s3api create-bucket --bucket <bucket-name> --region <region> --create-bucket-configuration LocationConstraint=<region>`
 - `aws s3 website s3://<bucket-name>/ --index-document index.html --error-document error.html`
 - `aws s3 cp index.html s3://<bucket-name>/index.html --grants read=uri=http://acs.amazonaws.com/groups/global/AllUsers`
