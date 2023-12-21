# pip install boto3
import boto3

# Create a low-level client representing Amazon SageMaker Runtime
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name="us-east-2")

# The endpoint name must be unique within
# an AWS Region in your AWS account.
endpoint_name = "sagemaker-xgboost"

# Gets inference from the model hosted at the specified endpoint:
response = sagemaker_runtime.invoke_endpoint(
    EndpointName=endpoint_name, Body=bytes("[6,148,72,35,0,33.6,0.627,50]", "utf-8")
)

# Decodes and prints the response body:
print(response["Body"].read().decode("utf-8"))
