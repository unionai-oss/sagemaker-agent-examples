import os
from contextlib import asynccontextmanager
from datetime import datetime

import fastapi
import numpy as np
import xgboost as xgb
from fastapi import Request, Response, status


class ModelManager:
    _model = {}
    _name = ""

    def __init__(self, name: str):
        self._name = name

    def load(self, location: str):
        booster = xgb.Booster()
        booster.load_model(os.path.join(location, self._name))
        self._model["model"] = booster

        print(f"Loaded model {self._model}")

    def predict(self, request: Request) -> float:
        return self._model["model"].predict(request)

    def reset(self):
        self._model = {}


ml_model = ModelManager(name="xgboost_model")


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    # Load the ML model
    path = os.getenv("MODEL_PATH", "/opt/ml/model")
    ml_model.load(location=path)
    yield

    # Clean up the ML model and release the resources
    ml_model.reset()


app = fastapi.FastAPI(lifespan=lifespan)


@app.get("/ping")
async def ping():
    return Response(content="OK", status_code=200)


@app.post("/invocations")
async def invocations(request: Request):
    print(f"Received request at {datetime.now()}")

    json_payload = await request.json()

    X_test = xgb.DMatrix(np.array(json_payload).reshape((1, -1)))
    y_test = ml_model.predict(X_test)

    response = Response(
        content=repr(round(y_test[0])).encode("utf-8"),
        status_code=status.HTTP_200_OK,
        media_type="text/plain",
    )
    return response
