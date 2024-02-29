import os
import tarfile
from typing import Annotated

import flytekit
from flytekit import ImageSpec, Secret, task
from flytekit.types.file import FileExt, FlyteFile

SECRET_GROUP = "aws-credentials"

upload_model_image = ImageSpec(
    name="sam-tar-and-upload", registry="samhitaalla", packages=["boto3"]
)

if upload_model_image.is_container():
    import boto3
    from botocore.exceptions import ClientError


@task(
    cache=True,
    cache_version="0.1",
    container_image=upload_model_image,
    secret_requests=[
        Secret(group=SECRET_GROUP, key="aws-access-key"),
        Secret(group=SECRET_GROUP, key="aws-secret-access-key"),
        Secret(group=SECRET_GROUP, key="aws-session-token"),
    ],
)
def upload_model(model: FlyteFile[Annotated[str, FileExt("PyTorchModule")]]) -> str:
    file_name = "model.tar.gz"
    tf = tarfile.open(file_name, "w:gz")
    tf.add(model.download(), arcname="sam_finetuned")
    tf.close()

    bucket = "sagemaker-sam"
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=flytekit.current_context().secrets.get(
            SECRET_GROUP, "aws-access-key"
        ),
        aws_secret_access_key=flytekit.current_context().secrets.get(
            SECRET_GROUP, "aws-secret-access-key"
        ),
        aws_session_token=flytekit.current_context().secrets.get(
            SECRET_GROUP, "aws-session-token"
        ),
    )
    try:
        s3_client.upload_file(file_name, bucket, os.path.basename(file_name))
    except ClientError as e:
        print(e)

    return f"s3://{bucket}/model.tar.gz"
