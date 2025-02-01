import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

# Load the latest saved model
iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

# Define BentoML service
svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input_series: np.ndarray) -> np.ndarray:
    result = iris_clf_runner.predict.run(input_series)
    return result