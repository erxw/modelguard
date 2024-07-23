# ModelGuard

ModelGuard is a simple library for validating model inputs and formatting outputs. It provides a set of tools and utilities to ensure that the inputs provided to your machine learning models are valid and conform to the expected format. Additionally, it helps you format the outputs of your models in a consistent and standardized way as text instead of numbers.

## Features

- Input formatting: It provides utilities to automatically format and reorder the inputs before they are passed to your models.
- Output formatting: ModelGuard helps you format the outputs of your models in a consistent and standardized way, making it easier to consume the results.

## Installation

To install ModelGuard, simply run the following command:

pip install modelguard

## Examples

### Input Transform
 Create a ModelGuard instance for input validation to ensure that features (feature1, feature2, feature3) appear in the inference dataset. If not, impute missing columns and remove unnecessary columns. 

from modelguard import InputGuard

guard = InputGuard.from_dataframe(df)

guard = InputGuard.from_dict({'name': str, 'age': int})

validated_xtrain = guard.transform(xtrain)

### Output Transform

 Creates a wrapper around model predictions to map the individual model predictions (numerical) into a string

from modelguard import OutputGuard

output_labels=[{"text": "The mystery animal is {value}", "mapping": {0: "Octopus", 1: "Starfish"}},
                {"text": "My favorite pet is {value}", "mapping": {0: "Dog", 1: "Cat"}}]


guard = OutputGuard(labels = output_labels)

predictions = ... # any way of generating a matrix of predictions

predicted_texts = guard.transform(predictions) 

joined_texts = ['.'.join(p) for p in predicted_texts] 

### No Output Mapping of values (Regression-type tasks)
from modelguard import OutputGuard

guard = OutputGuard(labels=[{"text": "The mystery number is {value} on Monday"},
                                  {"text": "My favorite digit is {value}"}])

predictions = ... # any way of generating a matrix of predictions

predicted_texts = guard.transform(predictions) 

### Example use with sklearn

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
label_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))

output_labels = [{"text": 'The patient is {value}', "mapping" = label_mapping}]
guard = OutputGuard(labels = output_labels)