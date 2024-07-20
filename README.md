# ModelGuard

ModelGuard is a powerful library for validating model inputs and formatting outputs. It provides a set of tools and utilities to ensure that the inputs provided to your machine learning models are valid and conform to the expected format. Additionally, it helps you format the outputs of your models in a consistent and standardized way.

## Features

- Input formatting: It provides utilities to automatically format and reorder the inputs before they are passed to your models.
- Output formatting: ModelGuard helps you format the outputs of your models in a consistent and standardized way, making it easier to consume the results.

## Installation

To install ModelGuard, simply run the following command:


## Examples

#### Input validation
### Create a ModelGuard instance for input validation to ensure that features (feature1, feature2, feature3) appear in the inference dataset. If not, impute missing columns and remove unnecessary columns. 

from modelguard import ModelGuard

guard = ModelGuard(input_features=["feature1", "feature2", "feature3"], impute_value = 0)
validated_xtrain = guard.transform_input(xtrain)

### Output transformation
### Creates a wrapper around model predictions to map the individual model predictions (numerical) into a string
from modelguard import ModelGuard

output_labels=[{"text": "The mystery animal is {text}", "mapping": {0: "Octopus", 1: "Starfish"}},
                {"text": "My favorite pet is {text}", "mapping": {0: "Dog", 1: "Cat"}}]


guard = ModelGuard(output_labels = output_labels)

predictions = ... # any way of generating a matrix of predictions
predicted_texts = guard.transform_output(predictions) # predicted_texts is a list of a list of strings
joined_texts = ['.'.join(p) for p in predicted_texts] # each sample gets one paragraph of strings instead of a single sentence


#### Both Input and Output validation
from modelguard import ModelGuard

output_labels=[{"text": "The mystery animal is {text} on Mondays", "mapping": {0: "Octopus", 1: "Starfish"}},
                {"text": "My favorite pet is {text}", "mapping": {0: "Dog", 1: "Cat"}}]

guard = ModelGuard(input_features=["feature1", "feature2", "feature3"], impute_value = 0, output_labels = output_labels)

validated_xtrain = guard.transform_input(xtrain)
predictions = ... # any way of generating a matrix of predictions
predicted_texts = guard.transform_output(predictions) # predicted_texts is a list of a list of strings


#### No Output Mapping of values
from modelguard import ModelGuard
guard = ModelGuard(output_labels=[{"text": "The mystery number is {text}"},
                                  {"text": "My favorite digit is {text}"}])

predictions = ... # any way of generating a matrix of predictions
predicted_texts = guard.transform_output(predictions) # predicted_texts is a list of a list of strings