import pytest
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict, Any, Union
from pydantic import BaseModel, create_model, Field
import pandas as pd
from typing import List, Dict, Any, Union, Type
from datetime import datetime, date, time, timedelta

dtype_mapping = {
    'int64': int,
    'float64': float,
    'object': str,
    'bool': bool,
    'datetime64[ns]': datetime,
    'timedelta[ns]': timedelta,
    'datetime64[ns, tz]': datetime,
    'category': str,
    'int32': int,
    'float32': float,
    'uint8': int,
    'uint16': int,
    'uint32': int,
    'uint64': int,
    'int8': int,
    'int16': int,
    'complex64': complex,
    'complex128': complex,
    'date': date,
    'time': time,
}

class InputGuard:
    validator: Type[BaseModel]

    def __init__(self, fields: Dict[str, List[Any]]):
        self.validator = create_model('Validator', **fields)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        fields = {
            column: (dtype_mapping.get(str(df[column].dtype), str), 0) # todo: add default impute_values
            for column in df.columns
        }
        return cls(fields = fields)
    
    @classmethod
    def from_dict(cls, data: Dict[str, List[Any]]):
        # dict of tuples (type, impute_value)
        fields = {
            key: v 
            for key, v in data.items()
        }
        return cls(fields = fields)

    def transform(self, data: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]], pd.DataFrame, Dict[str, Any], Dict[str, List[Any]]]):

        format = lambda data: [list(self.validator(**d).model_dump().values()) for d in data]
        if isinstance(data, list): # assume list of dict
            pass
        elif isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()): # dict of dict
            data = list(data.values())
        elif isinstance(data, dict) and all(isinstance(v, list) for v in data.values()): # dict of list
            data = [dict(zip(data.keys(), v)) for v in zip(*data.values())]
        elif isinstance(data, dict): # just a dict
            data = [data]
        elif isinstance(data, pd.DataFrame): # pandas dataframe
            data = list(data.to_dict(orient = 'index').values())
        else:
            return data
        return format(data)

class TextWrapper(BaseModel):
    text: str = Field(description = "Text description of the model output")
    mapping: Union[Dict[int, str], None] = Field(default = None, description = "Mapping of prediction values to output labels")


class OutputGuard(BaseModel):
    labels: Union[List[TextWrapper], None] = Field(default = None, description = "List of output labels")

    def transform(self, data) -> List[List[str]]:
        if not self.labels:
            return data
        assert len(data) > 0, "No data to transform"
        assert len(data[0]) == len(self.labels), "Output label length mismatch. You should have a label for each prediction"
        results = []
        for sample in data:
            temp = []
            for i, label in enumerate(self.labels):
                if label.mapping: 
                    value = label.mapping.get(sample[i],  "Unknown")
                else:
                    value = sample[i]
                text = label.text.format(value = value)
                temp.append(text)
            results.append(temp)
        return results