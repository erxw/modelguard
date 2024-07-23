import pytest
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict, Any, Union
from pydantic import BaseModel, create_model, Field
import pandas as pd
from typing import List, Dict, Any, Union, Type
from datetime import datetime, date, time, timedelta
from modelguard.guard import InputGuard, OutputGuard

class Validator(BaseModel):
    column1: int
    column2: int

@pytest.fixture
def input_guard():
    # Create an instance of InputGuard for testing
    fields = {
        'column1': [1, 2, 3],
        'column2': [4, 5, 6]
    }
    return InputGuard.from_dataframe(pd.DataFrame(fields))

def test_transform_with_list_of_dicts(input_guard):
    data = [
        {'column1': 1, 'column2': 4},
        {'column1': 2, 'column2': 5},
        {'column1': 3, 'column2': 6}
    ]
    transformed_data = input_guard.transform(data)
    assert transformed_data == [[1, 4], [2, 5], [3, 6]]

def test_transform_with_dict_of_dicts(input_guard):
    data = {
        'row1': {'column1': 1, 'column2': 4},
        'row2': {'column1': 2, 'column2': 5},
        'row3': {'column1': 3, 'column2': 6}
    }
    transformed_data = input_guard.transform(data)
    assert transformed_data == [[1, 4], [2, 5], [3, 6]]

def test_transform_with_dict_of_lists(input_guard):
    data = {
        'column1': [1, 2, 3],
        'column2': [4, 5, 6]
    }
    transformed_data = input_guard.transform(data)
    assert transformed_data == [[1, 4], [2, 5], [3, 6]]

def test_transform_with_single_dict(input_guard):
    data = {'column1': 1, 'column2': 4}
    transformed_data = input_guard.transform(data)
    assert transformed_data == [[1, 4]]

def test_transform_with_dataframe(input_guard):
    data = pd.DataFrame({
        'column1': [1, 2, 3],
        'column2': [4, 5, 6]
    })
    transformed_data = input_guard.transform(data)
    assert transformed_data == [[1, 4], [2, 5], [3, 6]]

def test_transform_with_invalid_data(input_guard):
    data = 'invalid_data'
    transformed_data = input_guard.transform(data)
    assert transformed_data == 'invalid_data'

def test_output_guard_transform_with_valid_predictions():
    output = OutputGuard(
        labels=[
            {
                "text": "The predicted Height is {value}"
            },
            {
                "text": "The predicted value is {value}",
                "mapping": {
                    0: "Low",
                    1: "High"
                }
            }
        ]
    )

    predictions = [[1, 0]]
    transformed_data = output.transform(predictions)
    assert transformed_data == [["The predicted Height is 1", "The predicted value is Low"]]
    predictions = [[1, 1]]
    transformed_data = output.transform(predictions)
    assert transformed_data == [["The predicted Height is 1", "The predicted value is High"]]
    predictions = [[0, 3]]
    transformed_data = output.transform(predictions)
    assert transformed_data == [["The predicted Height is 0", "The predicted value is Unknown"]]

def test_output_guard_transform_with_empty_predictions():
    output = OutputGuard(
        labels=[
            {
                "text": "The predicted Height is {value}"
            },
            {
                "text": "The predicted value is {value}",
                "mapping": {
                    0: "Low",
                    1: "High"
                }
            }
        ]
    )


    predictions = [[]]
    with pytest.raises(AssertionError):
        transformed_data = output.transform(predictions)


def test_output_guard_transform_with_invalid_predictions():
    output = OutputGuard(
        labels=[
            {
                "text": "The predicted Height is {value}"
            },
            {
                "text": "The predicted value is {value}",
                "mapping": {
                    0: "Low",
                    1: "High"
                }
            }
        ]
    )
    predictions = [[1, 2, 3]]
    with pytest.raises(AssertionError):
        transformed_data = output.transform(predictions)



def test_output_guard_transform_with_no_labels():
    output = OutputGuard(labels=None)

    predictions = [[1, 0]]
    transformed_data = output.transform(predictions)
    assert transformed_data == [[1, 0]]

    predictions = [[1, 1]]
    transformed_data = output.transform(predictions)
    assert transformed_data == [[1, 1]]

    predictions = [[0, 3]]
    transformed_data = output.transform(predictions)
    assert transformed_data == [[0, 3]]

    predictions = [[1, 0], [1, 1]]
    transformed_data = output.transform(predictions)
    assert transformed_data == [[1, 0], [1, 1]]
