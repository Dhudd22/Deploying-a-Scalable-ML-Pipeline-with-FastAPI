# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is used to predict if individuals make more than $50k per year. The model used is a Random Forest Classifier.

## Intended Use
This model is used to predict the level of individuals income using other individual attributes to aid in the prediction. It is intended primarily for academic purposes.
## Training Data
The data comes from the US Census Bureau and was obtained from the CI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income) as a csv file. The data has 32562 records which track 14 different attributes (columns). The primary focus of this project is the "salary" attribute which is split into "<=50k" and ">50K". A small amount of simple general cleaning was used on the data set as detailed in the data.py file.
## Evaluation Data
For this particular model 20% was used for testing the model while 80% was initially used for the training of the model.
## Metrics
The model's precision score was 0.7342 with a recall score of 0.6249 and an F1: score of 0.6752
## Ethical Considerations
The dataset should not be considered universally representative, and therefore should not be used to assume the salaries of groups based on identifying characteristics.

## Caveats and Recommendations
While the data is valuable for the purposes of training ML models, since the data is older, many of the findings may be outdated as much of the demographic data has changed significantly. I would recommend using more up to date versions of the data for any application that seeks to make accurate predictions for the present day.
