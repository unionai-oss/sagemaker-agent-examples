import os

from dotenv import load_dotenv
from flytekit import ImageSpec, kwtypes, workflow
from flytekitplugins.awssagemaker_inference import (
    SageMakerInvokeEndpointTask,
    create_sagemaker_deployment,
)

load_dotenv()

REGION = "us-east-2"
MODEL_NAME = "sam-model"
ENDPOINT_CONFIG_NAME = "sam-endpoint-config"
ENDPOINT_NAME = "sam-endpoint"

sam_deployment_image = ImageSpec(
    name="sam-deployment",
    registry=os.getenv("REGISTRY"),
    packages=["transformers", "torch", "monai", "matplotlib", "fastapi", "uvicorn"],
    base_image="samhitaalla/sam-base-image:0.0.2",  # required to copy code into the container when executing the workflow locally
).with_commands(["chmod +x /root/serve"])


sam_deployment = create_sagemaker_deployment(
    name="sam",
    model_input_types=kwtypes(model_path=str, execution_role_arn=str),
    model_config={
        "ModelName": MODEL_NAME,
        "PrimaryContainer": {
            "Image": "{container.image}",
            "ModelDataUrl": "{inputs.model_path}",
        },
        "ExecutionRoleArn": "{inputs.execution_role_arn}",
    },
    endpoint_config_input_types=kwtypes(instance_type=str),
    endpoint_config_config={
        "EndpointConfigName": ENDPOINT_CONFIG_NAME,
        "ProductionVariants": [
            {
                "VariantName": "variant-name-1",
                "ModelName": MODEL_NAME,
                "InitialInstanceCount": 1,
                "InstanceType": "{inputs.instance_type}",
            },
        ],
        "AsyncInferenceConfig": {
            "OutputConfig": {
                "S3OutputPath": "s3://sagemaker-sam/inference-output/output"
            }
        },
    },
    endpoint_config={
        "EndpointName": ENDPOINT_NAME,
        "EndpointConfigName": ENDPOINT_CONFIG_NAME,
    },
    container_image=sam_deployment_image,
    region=REGION,
)


@workflow
def sam_deployment_with_default_inputs(
    model_path: str = "s3://sagemaker-sam/model.tar.gz",
    execution_role_arn: str = os.getenv("EXECUTION_ROLE_ARN"),
) -> str:
    return sam_deployment(
        model_path=model_path,
        execution_role_arn=execution_role_arn,
        instance_type="ml.m4.xlarge",
    )


invoke_endpoint = SageMakerInvokeEndpointTask(
    name="sam-invoke-endpoint",
    config={
        "EndpointName": ENDPOINT_NAME,
        "InputLocation": "s3://sagemaker-sam/inference_input",
        "ContentType": "application/octet-stream",
    },
    region=REGION,
)
