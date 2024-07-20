import pytest
from modelguard.validator import ModelGuard
import pandas as pd

@pytest.fixture
def model_guard():
    return ModelGuard(input_features=["feature1", "feature2", "feature3"],
                      output_labels=[{"text": "The mystery animal is {text}", "mapping": {0: "Octopus", 1: "Starfish"}},
                                     {"text": "My favorite pet is {text}", "mapping": {0: "Dog", 1: "Cat"}}],
                      impute_value=0)


def test_transform_input(model_guard):
    # Test case 1: Test with a single dictionary input
    input_data = {"feature1": 1, "feature2": 2}
    transformed_data = model_guard.transform_input(input_data)
    assert isinstance(transformed_data, pd.DataFrame)
    assert transformed_data.shape == (1, 3)
    assert list(transformed_data.columns) == ["feature1", "feature2", "feature3"]

    # Test case 2: Test with a list of dictionaries input
    input_data = [{"feature1": 1, "feature2": 2}, {"feature1": 3, "feature2": 4}]
    transformed_data = model_guard.transform_input(input_data)
    assert isinstance(transformed_data, pd.DataFrame)
    assert transformed_data.shape == (2, 3)
    assert list(transformed_data.columns) == ["feature1", "feature2", "feature3"]

    # Test case 3: Test with a pandas DataFrame input (out of order and missing column)
    input_data = pd.DataFrame({"feature2": [2, 4], "feature1": [1, 3]})
    transformed_data = model_guard.transform_input(input_data)
    assert isinstance(transformed_data, pd.DataFrame)
    assert transformed_data.shape == (2, 3)
    assert list(transformed_data.columns) == ["feature1", "feature2", "feature3"]

    # Test case 4: Test with a pandas DataFrame input (includes extra feature)
    input_data = pd.DataFrame({"feature1": [1, 3], "feature4": [2, 4]})
    transformed_data = model_guard.transform_input(input_data)
    assert isinstance(transformed_data, pd.DataFrame)
    assert transformed_data.shape == (2, 3)
    assert list(transformed_data.columns) == ["feature1", "feature2", "feature3"]
    assert transformed_data['feature2'].sum() == 0


    # Test case 5: Test with invalid input data type
    input_data = "invalid_input"
    with pytest.raises(TypeError):
        transformed_data = model_guard.transform_input(input_data)



def test_transform_output(model_guard):
    # Test case 1: Test with a single prediction value
    output_data = [[0, 1]]
    transformed_data = model_guard.transform_output(output_data)
    assert isinstance(transformed_data, list)
    assert len(transformed_data) == 1
    assert isinstance(transformed_data[0], list)
    assert len(transformed_data[0]) == 2
    assert transformed_data[0][0] == "The mystery animal is Octopus"
    assert transformed_data[0][1] == "My favorite pet is Cat"


    # Test case 2: Test with a single prediction value
    output_data = [[0, 1], [1, 1]]
    transformed_data = model_guard.transform_output(output_data)
    assert isinstance(transformed_data, list)
    assert len(transformed_data) == 2
    assert isinstance(transformed_data[0], list)
    assert len(transformed_data[0]) == 2
    assert transformed_data[0][0] == "The mystery animal is Octopus"
    assert transformed_data[0][1] == "My favorite pet is Cat"

    assert transformed_data[1][0] == "The mystery animal is Starfish"
    assert transformed_data[1][1] == "My favorite pet is Cat"

    # Test case 3: Test with invalid output data type
    output_data = [0, 1]
    with pytest.raises(TypeError):
        transformed_data = model_guard.transform_output(output_data)