## Creating a Chalice Serverless Front-end
Following this tutpoorial: https://aws.amazon.com/blogs/machine-learning/build-a-serverless-frontend-for-an-amazon-sagemaker-endpoint/
1. Connect to EC2 Amazon-Linux-2
2. `sudo yum update -y`
3. `sudo amazon-linux-extras install python3`
4. `curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py`
5. `python3 get-pip.py --user`
6. `pip install chalice boto3 --user`
7. `export AWS_DEFAULT_REGION=us-east-2`
8. `chalice new-project prudential-app2`
9. `cd prudential-app2`
10. edit requirements.txt and app.py
11. `chalice deploy`
12. create a index.html file
13. `sed -i s@CHALICE_ENDPOINT@https://1adhvanhe3.execute-api.us-east-2.amazonaws.com/api/@g index.html`
14. `aws s3api create-bucket --bucket prudential-app2 --region us-east-2 --create-bucket-configuration LocationConstraint=us-east-2`
15. `aws s3 website s3://prudential-app2/ --index-document index.html --error-document error.html`
16. `aws s3 cp index.html s3://prudential-app2/index.html --grants read=uri=http://acs.amazonaws.com/groups/global/AllUsers`
