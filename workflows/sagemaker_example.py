import os
import tarfile
from contextlib import asynccontextmanager
from datetime import datetime

import fastapi
import numpy as np
from fastapi import Request, Response, status
from numpy import loadtxt
from sklearn.model_selection import train_test_split

import flytekit
from flytekit import ImageSpec, task, workflow
from flytekit.types.file import FlyteFile

custom_image = ImageSpec(name="sagemaker-xgboost", requirements="requirements.txt", commands=["chmod a+x serve"])

if custom_image.is_container():
    from xgboost import XGBClassifier, Booster, DMatrix


class Predictor:
    def __init__(self, model_path: str):
        self._model = Booster()
        self._model.load_model(model_path)

    def predict(self, inputs: DMatrix) -> np.ndarray:
        return self._model.predict(inputs)

ml_model: Predictor = None

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    # Load the ML model
    global ml_model
    path = os.getenv("MODEL_PATH", "xgboost_model")
    ml_model = Predictor(model_path=path)
    yield


app = fastapi.FastAPI(lifespan=lifespan)


@app.get("/ping")
async def ping():
    return Response(content="OK", status_code=200)


@app.post("/invocations")
async def invocations(request: Request):
    print(f"Received request at {datetime.now()}")

    json_payload = await request.json()

    X_test = DMatrix(np.array(json_payload).reshape((1, -1)))
    y_test = ml_model.predict(X_test)

    response = Response(
        content=repr(round(y_test[0])).encode("utf-8"),
        status_code=status.HTTP_200_OK,
        media_type="text/plain",
    )
    return response


@task
def predict(model: FlyteFile, input: DMatrix) -> np.ndarray:
    Predictor(model_path=model.download()).predict(inputs=input)


@task(container_image=custom_image)
def train_model(dataset: FlyteFile) -> FlyteFile:
    dataset = loadtxt(dataset.download(), delimiter=",")
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    X_train, _, y_train, _ = train_test_split(X, Y, test_size=0.33, random_state=7)

    print(X_train)

    model = XGBClassifier()
    model.fit(X_train, y_train)

    serialized_model = os.path.join(
        flytekit.current_context().working_directory, "xgboost_model.json"
    )
    booster = model.get_booster()
    booster.save_model(serialized_model)

    return FlyteFile(path=serialized_model)


@task
def convert_to_tar(model: FlyteFile) -> FlyteFile:
    tf = tarfile.open("model.tar.gz", "w:gz")
    tf.add(model.download(), arcname="xgboost_model")
    tf.close()

    return FlyteFile("model.tar.gz")


DEMO_DATASET = "https://gist.githubusercontent.com/ktisha/c21e73a1bd1700294ef790c56c8aec1f/raw/819b69b5736821ccee93d05b51de0510bea00294/pima-indians-diabetes.csv"


@workflow
def sagemaker_xgboost_wf(dataset: FlyteFile = DEMO_DATASET) -> FlyteFile:
    serialized_model = train_model(dataset=dataset)
    return convert_to_tar(model=serialized_model)


# Write a serve decorator that can be added to any python function for serving
# def serve(fn, *, auto_model_load: bool = False):
#     def wrapper(*args, **kwargs):
#         fn(*args, **kwargs)
#
#     return wrapper
#
#
# import uvicorn
#
#
# @serve.entrypoint
# def run(port: int = 5000, log_level: str = "info", host: str = "127.0.0.1"):
#     uvicorn.run(app, host=host, port=port, log_level=log_level)
#
# @serve.app
# class App:
#   def __init__(self):
#     pass
#   def __call__(self, request: Request):
#     return Response(content="OK", status_code=200)

"""
Automatically add `serve` command when you install flytekitplugins-serve. thus if you run
`docker run <img> serve` it should invoke the serve function
"""
