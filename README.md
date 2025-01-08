# PCE-Prediction

## Part 1: Forecasting US Personal Consumption Expenditures

The goal is to compare three forecasting models to identify the best-performing one for predicting US seasonally-adjusted personal consumption expenditures (PCE):

Simple Forecasting Method (Average, Naïve, Seasonal Naïve, or Drift) - (Naive method was used)

Exponential Smoothing Model - Holt's method was used

ARIMA Model

### Steps Involved

#### Data Preprocessing:

Handle missing data and split the dataset into training and test sets.

#### Model Development and Evaluation:

Train and evaluate all three models using performance metrics like MAE, RMSE, or MAPE.

Visualize predictions vs. actual values for all models in one graph.

#### Future Estimation:

Predict PCE for October 2024 using the best-performing model.

#### Rolling Forecast Comparison:

Perform one-step-ahead rolling forecasts without re-estimating parameters and compare models.

## Part 2: Topic Modeling on Hotel Reviews

Analyze 10,000 hotel reviews to extract topics discussed in positive and negative reviews, based on customer ratings (Likert scale 1–5).

### Steps Involved

#### Sampling:

Used set.seed() to select random reviews for reproducibility.

#### Classify Reviews:

Positive Reviews: Ratings of 4 and 5.

Negative Reviews: Ratings of 1 and 3.

#### Text Preprocessing:

Tokenization, stop word removal, stemming/lemmatization, and other cleaning steps.

#### Sentiment Analysis:

A sentiment analysis was performed using the 'bing' lexicon. The sentiment analysis assigned positive and negative scores to words in each review, allowing to calculate an overall sentiment score for each review.

#### Topic Modeling:

Use Latent Dirichlet Allocation (LDA) to identify key topics.

Select the number of topics based on coherence scores or similar criteria.

#### Interpret Results:

Label topics and discuss the top three factors affecting customer satisfaction and dissatisfaction.
