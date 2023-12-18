import os
import tarfile

import flytekit
from flytekit import ImageSpec, task, workflow
from flytekit.types.file import FlyteFile
from numpy import loadtxt
from sklearn.model_selection import train_test_split

custom_image = ImageSpec(name="sagemaker-xgboost", packages=["xgboost"])

if custom_image.is_container():
    from xgboost import XGBClassifier


@task(container_image=custom_image)
def train_model() -> FlyteFile:
    dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
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


@workflow
def sagemaker_xgboost_wf() -> FlyteFile:
    serialized_model = train_model()
    return convert_to_tar(model=serialized_model)
