import os

from dotenv import load_dotenv
from flytekit import ImageSpec, kwtypes, workflow
from flytekitplugins.awssagemaker import (
    SagemakerDeleteEndpointConfigTask,
    SagemakerDeleteEndpointTask,
    SagemakerDeleteModelTask,
    SagemakerEndpointConfigTask,
    SagemakerEndpointTask,
    SagemakerInvokeEndpointTask,
    SagemakerModelTask,
)

load_dotenv()

REGION = os.getenv("REGION")


custom_image = ImageSpec(
    name="sagemaker-xgboost",
    registry=os.getenv("REGISTRY"),
    requirements="requirements.txt",
    apt_packages=["git"],
    base_image="samhitaalla/sagemaker-agent:0.0.1",  # Dockerfile
).with_commands(["chmod +x /root/serve"])


create_sagemaker_model = SagemakerModelTask(
    name="sagemaker_model",
    config={
        "ModelName": "{inputs.model_name}",
        "PrimaryContainer": {
            "Image": "{container.image}",
            "ModelDataUrl": "{inputs.model_data_url}",
        },
        "ExecutionRoleArn": "{inputs.execution_role_arn}",
    },
    region=REGION,
    container_image=custom_image,
    inputs=kwtypes(model_name=str, model_data_url=str, execution_role_arn=str),
)

create_endpoint_config = SagemakerEndpointConfigTask(
    name="sagemaker_endpoint_config",
    config={
        "EndpointConfigName": "{inputs.endpoint_config_name}",
        "ProductionVariants": [
            {
                "VariantName": "variant-name-1",
                "ModelName": "{inputs.model_name}",
                "InitialInstanceCount": 1,
                "InstanceType": "ml.m4.xlarge",
            },
        ],
        "AsyncInferenceConfig": {
            "OutputConfig": {"S3OutputPath": "{inputs.s3_output_path}"}
        },
    },
    region=REGION,
    inputs=kwtypes(endpoint_config_name=str, model_name=str, s3_output_path=str),
)

create_endpoint = SagemakerEndpointTask(
    name="sagemaker_endpoint",
    config={
        "EndpointName": "{inputs.endpoint_name}",
        "EndpointConfigName": "{inputs.endpoint_config_name}",
    },
    region=REGION,
    inputs=kwtypes(endpoint_name=str, endpoint_config_name=str),
)


delete_endpoint = SagemakerDeleteEndpointTask(
    name="sagemaker_delete_endpoint",
    config={"EndpointName": "{inputs.endpoint_name}"},
    region=REGION,
    inputs=kwtypes(endpoint_name=str),
)

delete_endpoint_config = SagemakerDeleteEndpointConfigTask(
    name="sagemaker_delete_endpoint_config",
    config={"EndpointConfigName": "{inputs.endpoint_config_name}"},
    region=REGION,
    inputs=kwtypes(endpoint_config_name=str),
)

delete_model = SagemakerDeleteModelTask(
    name="sagemaker_delete_model",
    config={"ModelName": "{inputs.model_name}"},
    region=REGION,
    inputs=kwtypes(model_name=str),
)


@workflow
def sagemaker_xgboost_fastapi_deployment(
    model_name: str = "sagemaker-xgboost",
    model_data_url: str = os.getenv("MODEL_DATA_URL"),
    execution_role_arn: str = os.getenv("EXECUTION_ROLE_ARN"),
    s3_output_path: str = os.getenv("S3_OUTPUT_PATH"),
    endpoint_config_name: str = "sagemaker-xgboost-endpoint-config",
    endpoint_name: str = "sagemaker-xgboost-endpoint",
):
    create_sagemaker_model(
        model_name=model_name,
        model_data_url=model_data_url,
        execution_role_arn=execution_role_arn,
    )
    create_endpoint_config(
        endpoint_config_name=endpoint_config_name,
        model_name=model_name,
        s3_output_path=s3_output_path,
    )
    create_endpoint(
        endpoint_name=endpoint_name, endpoint_config_name=endpoint_config_name
    )
    delete_endpoint(endpoint_name=endpoint_name)
    delete_endpoint_config(endpoint_config_name=endpoint_config_name)
    delete_model(model_name=model_name)


invoke_endpoint = SagemakerInvokeEndpointTask(
    name="sagemaker_invoke_endpoint",
    config={
        "EndpointName": "sagemaker-xgboost-endpoint",
        "InputLocation": os.getenv("INPUT_LOCATION"),
    },
    region=REGION,
)
