from pydantic import BaseModel, Field
from typing import Dict, List, Union, Optional
from typing import Union, List, Dict, Any
import numpy as np
import pandas as pd

# ensures data has the correct ordering of features (if too many features, truncated)

class TextWrapper(BaseModel):
    text: str = Field(description = "Text description of the model output")
    mapping: Dict[int, str] = Field(description = "Mapping of prediction values to output labels")

class ModelGuard(BaseModel):
    input_features: Union[List[Any], None] = Field(default = None, description = "List of input features")
    output_labels: Union[List[TextWrapper], None] = Field(default = None, description = "List of output labels")
    impute_value: Optional[float] = Field(default = 0, description = "Value to impute for missing feature values")

    @staticmethod
    def _validate_input_format(data: Union[dict, List[dict], pd.DataFrame]) -> pd.DataFrame:
        """
        Validates the input data format and returns a pandas DataFrame.

        Parameters:
            data (list, dict, pd.DataFrame): The input data to be validated.

        Returns:
            pd.DataFrame: The validated input data as a pandas DataFrame.

        Raises:
            ValueError: If the data format is not supported.

        """
        if isinstance(data, list) and all(isinstance(d, dict) for d in data):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            if all(isinstance(d, dict) for d in data.values()):
                return pd.DataFrame.from_dict(data, orient='index')
            elif all(isinstance(d, list) for d in data.values()):
                return pd.DataFrame(data)
            else:
                return pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            return data
        else:
            raise TypeError("Unsupported data format")

    def transform_input(self, data: Union[dict, List[dict], pd.DataFrame]) -> pd.DataFrame:
        """
        Transforms the input data by validating its format and filling missing values.

        Args:
            data (pandas.DataFrame): The input data to be transformed.

        Returns:
            pandas.DataFrame: The transformed data with validated format and filled missing values.
        """
        if not self.input_features:
            return data
        data = self._validate_input_format(data)
        empty = np.full((data.shape[0], len(self.input_features)), self.impute_value)
        empty = pd.DataFrame(empty, index=data.index, columns=self.input_features)
        empty.update(data)
        return empty.copy()

    def transform_output(self, data) -> List[List[str]]:
            """
            Transforms the output data into a list of strings based on the output labels.

            Args:
                data (List[List[int]]): The output data to be transformed.

            Returns:
                List[List[str]]: The transformed output data as a list of strings.
            """
            if not self.output_labels:
                return data
            assert len(data) > 0, "No data to transform"
            assert len(data[0]) == len(self.output_labels), "Output label length mismatch. You should have a label for each prediction"

            results = []
            for sample in data:
                temp = []
                for i, label in enumerate(self.output_labels):
                    temp.append(label.text.format(text = label.mapping[sample[i]]))
                results.append(temp)
            return results
