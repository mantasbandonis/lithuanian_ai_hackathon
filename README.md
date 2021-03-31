# Hackathon organised by Artificial Intelligence Association of Lithuania (March 16-22) 

Code from Team JKU Linz (Mantas Bandonis, Christoph Ostertag, Lukas Nöbauer, Sebastian Eibl)

Result: 
* 1st Place Track 1 
* 4th Place Track 2

# Track 1:
## Financial texts and messages sentiment classification task.

You will be provided with file: 
all-data.csv
The dataset contains two columns, "Sentiment" and "News Headline". The sentiment can be negative, neutral or positive.

The original dataset was from Kaggle dataset:
https://www.kaggle.com/ankurzing/sentiment-analysis-for-financial-news

The model will measure the accuracy of sentiment classification.

Since data is Kaggle dataset, the private testing will be done on a private dataset.
The illustrative example of public-test set is added as additional file: 
public-test-set.csv.
The private testing will be done by Stockgeist.ai an financial sentiment analysis monitoring platform.



# Track 2:
## Predict sales of retail store chain

You will be provided with a store sales prediction problem. Over 1000 store sales records covering over 2.5 years of daily sales revenue for each shop - overall over 1 million sales records are available. This is real life data from a retail store chain in Germany which has a strong position in the market.

Given this data you are asked to analyze customer shopping patterns, seasonality and other factors. With given data you will need to predict daily sales figures for next 7 weeks for each shop. Having accurate predictions would help stores to organize and optimize their inventory and thus save money on inventory and staff management.

The data covers a period from 2013-01-01 to 2015-07-31, and you are asked to predict sales for the period of 2015-08-01 to 2015-09-17.

The data is organized into 3 files:
* train_data.csv - all data available from a period 2013-01-01 to 2015-07-31. This file contains the “Sales” field.
* test_data.csv - all data available from a period 2015-08-01 to 2015-09-17. This file does not contain a “Sales” field - this is what you are asked to predict.
* test_predictions_format.csv - this is an example file you are asked to generate with your models. “Id” field matches test_data.csv file’s “Id” field.

Data column explanations:

* Store - store identification number
* Date - date of the store sales record
* DayOfWeek - weekday of the Date
* Sales - revenue from goods sold during that Date (only available in train_data.csv)
* ShopOpen - boolean flag if shop was open during that Date (if not open, Sales should be 0)
* Promotion - boolean flag if any promotions were done during that Date
* StateHoliday - factor variable if the Date is state holiday or not
* SchoolHoliday - factor variable if the Date is school holiday or not
* StoreType - factor variable describing the type of a store
* AssortmentType - factor variable describing the assortment type of a store

Models will be evaluated by measuring the Root Mean Square Percentage Error (RMSPE) between the actual Sales and the predicted Sales during 2015-08-01 to 2015-09-17 period. 
Days when the store had 0 actual sales are excluded from metric calculations.

