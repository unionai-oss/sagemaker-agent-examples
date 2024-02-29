from datetime import timedelta

from flytekit import approve, workflow

from .batch_predict import batch_predict
from .fine_tune import fine_tune_sam
from .upload_model import upload_model


@workflow
def sam_sagemaker_deployment() -> str:
    model = fine_tune_sam()
    predictions = batch_predict(model=model)

    approve_filter = approve(
        predictions, "batch_predictions_approval", timeout=timedelta(hours=2)
    )

    s3_uri = upload_model(model=model)
    approve_filter >> s3_uri

    return s3_uri
