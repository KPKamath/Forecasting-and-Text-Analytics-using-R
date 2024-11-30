# Part 1
The given data contains information about the monthly personal consumption expenditures (PCE) in the US from January 1959 to November 2023. This section of the report seeks to identify the best performing forecasting model for the given data as it helps in providing insights on the economic trend and anticipated fluctuations. The report will tackle following objectives:
•	Compare the forecasting ability of simple forecasting model, exponential smoothing model and ARIMA model and suggest the best model based on the results.
•	Using the best model from the first task to predict the PCE for October 2024.
•	Using one step ahead rolling forecasting without re-estimation to compare the three models.
Before proceeding to test the forecasting abilities of the models it is essential to perform the data preparation steps. This is started by installing and loading necessary libraries, and the dataset into R. To test the forecasting models the input needs to be a structured sequence where each data point is associated with particular time. Hence, data is converted into time series with frequency equal to 12 since the data has monthly records. Once the data is converted to time series, a check for missing data is conducted as it is important to handle the missing data to have a reliable forecasting model. It is found that there are 779 data points, 43 out of which are missing values in the dataset. Imputation is carried out using the interpolation function to address the missing data. Interpolation function performs a linear interpolation by replacing the missing value with the mean of the data values before and after it. 


   <img width="640" alt="image" src="https://github.com/user-attachments/assets/a0d584e7-b517-4ac1-a92f-38b7137de4f4">

Figure 1.1: Plot of data time series


Although it is already known that the data is seasonally adjusted and from Figure1.1 it can be observed that the data has a clear trend, a check for seasonality is conducted to ensure the same. From the Figure 1.2 shown below it is confirmed that there is no seasonality in the dataset.

<img width="717" alt="image" src="https://github.com/user-attachments/assets/4d713c35-22a4-4d77-91b2-60fa82e91df0">
 
 Figure 1.2: Seasonal plot of the data

 <img width="716" alt="image" src="https://github.com/user-attachments/assets/8c3e3be6-c312-4e79-9c32-a0a91272c7f1">

 Figure 1.3: Residuals plot

From the residuals plot above we can see that the data exhibits a clear trend as the ACF plot decreases exponentially. Additive decomposition is performed to analyse if there is any anomaly or irregularity in the data to ensure better forecasting. From the figure below it is observed that there are no irregularities in the data and the seasonal component is constant.

 <img width="722" alt="image" src="https://github.com/user-attachments/assets/c134cfc7-9c26-4461-b425-ac7d1f8f3e2a">

Figure 1.4: Additive decomposition plot

Now that it is clearly established that there is no seasonality, irregularity or missing values, the data can be split into train set and test set. This split is essential because it assists in evaluating the predictive ability of the model. The model is trained on the data using the train set and the test set is used to assess the performance of the model on the unseen data. This assures that the model is analysing the patterns rather than just memorizing and it can now be used to forecast the new data. For the purpose of this analysis the time series of the data is split as 80% (i.e 623 data points) for the train set and 20% (i.e 156 data points) for the test set.
Once the test and the train split is established the next step is to test the predictive ability of the models. Naive model, is the simplest among the four simple forecasting models. This model predicts the future values entirely on the basis of most recently observed value of the time series completely ignoring the other data points. Naive forecast method basically assumes that the projected value will be same as the recently observed value of the time series. After training the naive model on the train set, summary as shown in Figure 1.4 suggests that mean error (ME) is a positive value indicating that the forecasted values are slightly greater than the actual value. Root mean square error (RMSE) is observed to be 29.69 and mean absolute error (MAE) which corresponds to the deviation of the predicted values from the actual value, is observed to have a value of 19.89. Figure 1.6 shows the plot for the forecast using the train set in naive method.

 <img width="892" alt="image" src="https://github.com/user-attachments/assets/ee545b24-0aaa-4f1f-8713-13a01a5058ef">

Figure 1.5: Summary of naive model

 <img width="739" alt="image" src="https://github.com/user-attachments/assets/82a67a9d-926e-4cee-8cd1-d112b89ddb4f">

Figure 1.6: Naive forecast plot

In exponential smoothing, the model allocates exponentially reducing weights to past values in which the most recent value gets the highest weightage implying that the recent observation will have a greater impact on the forecast. Exponential smoothing does not account for trend and seasonality in the data and since PCE has the trend component but no seasonality, Holt’s method is most appropriate type of exponential smoothing that can be used. Figure 1.7 shows the plot for the forecast using the train set in Holt’s method.

 <img width="739" alt="image" src="https://github.com/user-attachments/assets/12f6654c-6a0d-4a2b-9ea4-5b058ec6987d">

Figure 1.7: Holt’s method forecast plot

Autoregressive integrated moving average (ARIMA) model is a forecasting model that predicts future values of a time series using the previously observed values by combining autoregression, difference and moving average components. The autoregressive component analyses the link between the past values and the current value. The moving average component checks for the impact of previous forecast errors and the differencing component removes trends or seasonality and helps in achieving stationarity. 

 <img width="700" alt="image" src="https://github.com/user-attachments/assets/cffa2f7f-71d3-4d33-b940-a980c5c09e6a">

Figure 1.8: ACF and PACF plots

From the figure above it can be noticed that in the ACF plot, values beyond lag 35 are above the significance boundaries which implies that the present value can be predicted from the previous values until lag 35 and it also suggest non-stationarity around the mean. The PACF plot has a strong spike at lag 1 which implies a strong partial autocorrelation between the current observation and just preceding observation suggesting that taking the first difference can assist in removing the trend component.

 <img width="617" alt="image" src="https://github.com/user-attachments/assets/dd472861-2060-4056-b7fa-65a2234b388e">


Figure 1.9: Result of ARIMA model residuals check

<img width="617" alt="image" src="https://github.com/user-attachments/assets/9b7c10eb-a981-49ae-addb-e8c6e71faac3">

Figure 1.10: ARIMA model forecast plot

The ARIMA model chosen by the auto arima function has 3 autoregressive, 2 differencing and 2 moving average components. From Figure 1.8 it can be said that there is not enough evidence to reject the null hypothesis as the p-value is greater that 0.05. This implies that the residuals of ARIMA(3,2,2) do not have significant autocorrelation. Figure 1.10 shows the plot for the forecast using auto ARIMA function that employs the train set in the model. 
Checking the accuracy of all three models using the test and train set, the predictive ability of each model can be evaluated. For the model to be a good fit, error measures have to be considerably less.


<img width="902" alt="image" src="https://github.com/user-attachments/assets/50b998ea-209a-4b73-903c-b65f4d8e25c1">

Figure 1.11: Accuracy of all three models

From the Figure 1.11 it can be observed that the naive model has the highest value of ME, RMSE and MAE on the test set suggesting that performance of the naive model is not the best. Holt’s method and ARIMA model have lower values of MAE, this implies that these models have better accuracy than naive model. Although there is a possibility that all the models are overfitted since the predictive ability is not the best for test set as demonstrated by higher values of error measures. This is because the count of values given for training the model is insufficient and the predictive ability can be improved by providing a larger train set. However, with the current split Holt’s model has a lower value of RMSE and ME on the test set in contrast with the ARIMA model thereby suggesting better performance. From accuracy checks and from the plot as shown in Figure 1.12 it can be concluded that Holt’s model is the best out of the three models to forecast the personal consumption expenditures. 

 <img width="788" alt="image" src="https://github.com/user-attachments/assets/15891eed-f16e-4cf2-af96-08db87e5df5b">

Figure 1.12: Plot of all three models and the main dataset

<img width="921" alt="image" src="https://github.com/user-attachments/assets/eca33298-8597-42c4-9608-cf8cdad2e50a">


Figure 1.13: Forecasted value for October 2024

Using the Holt’s model to predict the value of personal consumption expenditure for October 2024, the predicted value is 16052.81.
The next task in this analysis is to use one step ahead rolling forecast without re-estimation to compare the naive, Holt’s and ARIMA models and determine which has a better performance. Rolling forecasting is a technique in which the model is first trained on a sample data set and then it is used to predict one step ahead data points for the test set which means the forecasting horizon is one (h=1). After the prediction is made for that time point, the available value is now updated into the model to predict the next value. This iterative process keeps repeating until end of the test set. For the purpose of this analysis the window size for the train set is 623 data points which occurs in the year 2010 in the data set and the test set is from 2011 to 2023.
Checking the accuracy of all three models after training the model it is observed that, ARIMA model has the lowest ME value but Holt’s and naive model have the lowest RMSE and ACF1 values closer to zero. Although all three models performed well, Holt’s model has lesser value of RMSE, ME, MAE and ACF1 thereby making it the best model in the rolling forecast as well. 
       
<img width="921" alt="image" src="https://github.com/user-attachments/assets/10d5fa6c-0098-4446-ae89-b31dbfd324d0">

Figure 1.14: Rolling forecast accuracy for three models

To conclude, in this analysis the forecasting abilities of the three models namely naive, Holt’s and ARIMA model were compared to find the best model to forecast the personal consumption expenditure for October 2024. The results showed that Holt’s model has the best accuracy and the forecast of PCE for October 2024 is 16052.81. Further, one step ahead rolling forecast confirmed that the Holt’s model has the best performance.

# Part 2
The data for this section of the report contains 10,000 customer reviews given online for hotels and their corresponding ratings on likert scale ranging from 1(low satisfaction) to 5(high satisfaction). The task outlined is to identify the top 3 key factors that lead to satisfactory or unsatisfactory reviews from the customer. Text analysis with topic modelling will be conducted to analyse the objectives.
After installing and loading all the necessary packages and libraries in R, the data is loaded. Since the data contains reviews in multiple languages it is important to filter the reviews that are only given in English for convenience of this analysis. The filtering is done using textcat function of textcat package. Once the filtering is successful, a test sample of 2000 reviews is taken and split into positive and negative reviews. This split is on the basis of ratings given by the customers and is categorized into 3 or less as negative reviews and 4 or above as positive reviews. To ensure that all the text, special characters and emojis present in the reviews is processed appropriately UTF-8 encoding is performed. Now the text that is converted into UTF is converted into a corpus. Corpus is a collection of documents with text and creating a corpus is usually the first step in topic modelling. Converting the document into corpus helps in cleaning the data and prepares it for the text mining tasks. The next step is text processing which involves operations like tokenization, removing punctuation, converting to lowercase, removing stop words and lemmatization. Tokenization converts text into words or meaningful units called tokens facilitating easier understanding and manipulation of data. Converting text to lowercase prevents considering the words with same spelling but have a dissimilar combination of upper and lowercase letters as different tokens. Removal of stop words and punctuation marks is also essential as they do not carry a lot of meaning. The next step is lemmatization in which the tokens are converted into their root form and similar tokens are grouped together. Consequently, due to the cleaning step it is very likely that all the tokens from certain documents might have been cleared which is why it is important to remove the empty documents from the corpus. 
Wordcloud can be used to derive insights on frequently encountered words in the corpus. Figure 2.1 shows the wordcloud for positive reviews and it can be observed that words like hotel, room, London, breakfast are prominent which implies that these words appear more frequently in the corpus.

 <img width="405" alt="image" src="https://github.com/user-attachments/assets/55d6fcb1-5739-499a-b78b-9351d907af4e">

Figure 2.1: Wordcloud for positive reviews

Figure 2.2 shows the wordcloud for negative reviews and it can be noticed that hotel, room, staff, stay are prominent.
 
 <img width="405" alt="image" src="https://github.com/user-attachments/assets/b64b5774-a9c5-47d4-bb51-3661f072bba5">

Figure 2.2: Wordcloud for negative reviews

To find the topics that are discussed in the reviews and to have an understanding of the factors that lead to positive or negative reviews, topic modelling using Latent Dirichlet Allocation is performed. Topic modelling is a technique used to generate a set of topics in the collection of documents. Initially is important to determine the number of topics (k). Different metrics like Griffiths2004, CaoJuan2009 and Arun2010 are used to determine the optimal number of topics that have to be extracted from the corpus. These metrics calculate the probability, frequency and divergence of each word within a topic facilitating easy interpretation of the topics.
From Figure 2.3 and Figure 2.4 it can be said that the optimal number of topics for both positive and negative reviews is between 15-20 topics as the curves seem to gain some stability in that duration. For positive reviews, 18 topics and for negative reviews 19 topics can be a good choice as it can be observed that there is a good compromise between the three metrics at these points.

            
<img width="780" alt="image" src="https://github.com/user-attachments/assets/a923a6ce-d8ab-418e-be96-648c356ed3e4">

Figure 2.3: Plot of the metrics to find the number of topics for positive reviews

               
<img width="780" alt="image" src="https://github.com/user-attachments/assets/44f65800-90f5-4022-a4e0-aa79deb93218">

Figure 2.4: Plot of the metrics to find the number of topics for negative reviews

<img width="877" alt="image" src="https://github.com/user-attachments/assets/d158d0bf-b975-40de-952d-1d5cc87c6430">

Figure 2.5: Topics for positive reviews

Figure 2.5 shows the 18 topics and 10 most frequently occurring words under each topic for positive reviews. The themes for these topics are identified and the topics are labeled accordingly. The labels for 18 topics from the positive reviews are given in the table below.

Topic Number	Label
1 - Description of the room facilities
2,11 - Overall experience in the hotel
3,6,13 -	Staff and service
4 -	Room, bathroom amenities
5 -	Accommodation experience
7 -	Quality of additional facilities offered
8,18 -	Overall satisfaction
9 -	Breakfast options
10,17 -	Hotel location and recommendation
12,15 -	Convenience of transportation 
14 -	Check in experience
16 -	Dining Experience

<img width="899" alt="image" src="https://github.com/user-attachments/assets/aac1c2bf-b71b-409c-a412-a7ca5876d0b2">

Figure 2.6: Topics for negative reviews

Figure 2.6 shows the 19 topics and 10 most frequently occurring words under each topic for negative reviews. The labels for 19 topics from the negative reviews are given in the table below.

Topic Number	Label
1 -	Hotel amenities
2 -	Room reservation issues
3 -	Bathroom cleanliness
4 -	Interaction with the staff
5 -	Complete stay experience
6 -	Breakfast Experience
7 -	Bar area and atmosphere
8 -	Surroundings of the hotel
9 -	Reception desk service
10 -	Quality of additional facilities offered
11 -	Disturbance during the night
12,15 -	Overall Experience
13 -	Room size
14 -	Arrival experience
16 -	Room amenities
17 -	Bathroom amenities
18 -	Accessibility of the hotel
19 -	Value for money paid

On observing the topics, some of the positive terms may be found in the topics related to negative reviews. This could be because the customer started the review by pointing out positive aspects and then proceed to highlight the negative aspects. The reviews with rating 3 might have a combination of positive and negative feedbacks throughout the corpus. 

 <img width="801" alt="image" src="https://github.com/user-attachments/assets/e0d39b22-cc60-458f-86ef-a7d28895fb63">

Figure 2.7: Visualisation of topic modelling for positive reviews

 <img width="852" alt="image" src="https://github.com/user-attachments/assets/6721a03d-cc6f-450a-816b-c5a29cf21493">

Figure 2.8: Visualisation of topic modelling for negative reviews

The intertopic distance map shows the relationship between the topics. The topics that are clustered demonstrate similarity in the topics and the topics that are far apart are mostly dissimilar to each other. From Figure 2.7 it can be said that hotel, room and staff are three most frequently occurring terms in all the topics for positive reviews. From Figure 2.8 it can be said that room, hotel and breakfast are three most frequently occurring terms in all the topics for negative reviews.
After establishing a clear understanding of topics, to identify the top three factors that affect the satisfaction and dissatisfaction of the customer, probability of occurrence or topic prevalence of each topic is checked. It can be measured by computing the number of documents assigned to each topic with respect to the total number of documents and sorting it in decreasing order. This provides insights on the distribution of topics in the data.

 <img width="977" alt="image" src="https://github.com/user-attachments/assets/39d5eb94-e14b-40cf-be6e-ae9e29202ed4">

Figure 2.9: Probability of occurrence of topics for positive reviews

From the figure above it can be observed that the top three topics that lead to customer satisfaction are topic 14, 3 and 13 as they have a higher prevalence score. From Table 1 it can be inferred that topics 14, 3 and 13 highlight check-in experience and staff service are the factors that lead to satisfactory reviews.

 
<img width="977" alt="image" src="https://github.com/user-attachments/assets/7423fb1b-5030-48a0-a251-c5025607dfb3">

Figure 2.10: Probability of occurrence of topics for negative reviews

From the figure above it can be observed that the top three topics that lead to customer dissatisfaction are topic 18, 14 and 12. From Table 2 it can be inferred that dissatisfaction stems from accessibility of the hotel, customer’s arrival experience at the hotel and overall experience with the hotel offers respectively.
By the means of text analysis and using topic modelling on the data containing online reviews provided by the customers, insights can be drawn on the factors influencing the satisfaction or dissatisfaction. By working on factors like customer arrival experience and improving other services offered, the dissatisfaction of the customers can be reduced. Maintaining good staff service can further enhance the overall experience of the customer.

# R File

## PART 1

install.packages("imputeTS")

install.packages("forecast")

library(imputeTS)

library(forecast)

data<- read.csv('PCE.csv', header=TRUE)

sum(is.na(data)) #check for missing data

pce_data<- ts(data$PCE,start=c(1959, 1), end=c(2023, 11), frequency=12)


pce_dataset<-na_interpolation(pce_data)

plot(pce_dataset)

seasonplot(pce_dataset) 

pce_decompose<- decompose(pce_dataset, type="additive")

plot(pce_decompose) #there is a trend

lines(pce_decompose$trend, col=2)

lines(pce_decompose$seasonal, col=4)

#splitting the data set into 80-20

trainset_pce <- subset(pce_dataset, end = length(pce_dataset) - 156)

options(max.print = 1500)

options(max.print = getOption("max.print"))

#NAIVE forecast

predict_naive <- naive(trainset_pce, h = 156) #naive forecast

predict_naive

plot(predict_naive)


#exponential forecast

fce_holt <- holt(trainset_pce, h=156)

fce_holt

plot(fce_holt)

#ARIMA

tsdisplay(pce_dataset)

arfit_pce <- auto.arima(trainset_pce)

fc_arima<- forecast(arfit_pce, h=156)

fc_arima

plot(forecast(arfit_pce, h=156))

checkresiduals(arfit_pce)

#Check accuracy of all models

accuracy(predict_naive,pce_dataset)

accuracy(fce_holt,pce_dataset)

accuracy(fc_arima,pce_dataset)

#Plot all three

autoplot(pce_dataset) + autolayer(fce_holt$mean) + autolayer(predict_naive$mean) + autolayer(fc_arima$mean)

holtmodel_pce <- holt(trainset_pce,h=167) 

holtmodel_pce

plot(holtmodel_pce)


#Rolling forecast

rf_holt <- tsCV(pce_dataset, holt, h = 1, window=623)

rf_holt[1:779]

rf_naive <- tsCV(pce_dataset, naive, h = 1, window=623)

rf_naive[1:779]

rf <- function(x, h){forecast(auto.arima(x), h=h)}

rf_arima <- tsCV(pce_dataset, rf, h=1, window=623)

rf_arima[1:779]


train_ar <- window(pce_dataset,end=2010.99)

fit_ar <- auto.arima(train_ar)

refit_ar <- Arima(pce_dataset, model=fit_ar)

fc_ar <- window(fitted(refit_ar), start=2011)

accuracy(fc_ar,pce_dataset)

train_holt <- window(pce_dataset,end=2010.99)

fit_holt <- holt(train_holt)

refit_holt <- holt(pce_dataset, model=fit_holt)

fc_holt <- window(fitted(refit_holt), start=2011)

accuracy(fc_holt,pce_dataset)

train_naive <- window(pce_dataset,end=2010.99)

fit_naive <- naive(train_naive)

refit_naive <- naive(pce_dataset, model=fit_naive)

fc_naive <- window(fitted(refit_naive), start=2011)

accuracy(fc_naive,pce_dataset)

autoplot(pce_dataset) + autolayer(fc_holt) + autolayer(fc_naive) + autolayer(fc_ar)





## PART 2

install.packages('stringr')

install.packages('RColorBrewer')

install.packages('topicmodels')

install.packages('ggplot2')

install.packages('LDAvis')

install.packages('servr')

install.packages('textcat')

install.packages('ldatuning')

install.packages('wordcloud')


library(dplyr) # basic data manipulation

library(tm) # package for text mining package

library(stringr) # package for dealing with strings

library(RColorBrewer)# package to get special theme color

library(wordcloud) # package to create wordcloud

library(topicmodels) # package for topic modelling

library(ggplot2) # basic data visualization

library(LDAvis) # LDA specific visualization 

library(servr) # interactive support for LDA visualization

library(textcat)

library(ldatuning)

set.seed(129)

data <- read.csv(file = "HotelsData.csv", header = TRUE)

data$lang <- sapply(data$Text.1, textcat)

write.csv(data, "data_lang.csv")

rev_english <- data %>%

  filter(data$lang == "english")

colnames(rev_english)

test <- sample_n(rev_english,2000)

#Split into positive and negative reviews

positive_review <- test[test$Review.score >= 4, ]

negative_review <- test[test$Review.score <= 3, ]

#FUNCTION BEGINS HERE

main_function <- function(reviews) {
  
  docs_main <- stringr::str_conv(reviews$Text.1, "UTF-8")

  reviews_main <- Corpus(VectorSource(docs_main))
  
  dtmrev_main <- DocumentTermMatrix(reviews_main,

                                      control = list(lemma=TRUE,removePunctuation = TRUE,
                                      
                                                     removeNumbers = TRUE, stopwords = TRUE,
                                                     
                                                     tolower = TRUE))
  
  
  
  row.sum=apply(dtmrev_main,1,FUN=sum)
  
  dtmdocs_main=dtmrev_main[row.sum!=0,]
  
  
  dtmnew_main <- as.matrix(dtmdocs_main)
  
  main_f <- colSums(dtmnew_main)
  
  main_f <- sort(main_f, decreasing=TRUE)
  
  doclen_main <- rowSums(dtmnew_main)
  
  main_f[1:20]
  
  words_main<- names(main_f)
  
  wordcloud(words_main[1:70], main_f[1:70], rot.per=0.15, 
  
            random.order = FALSE, scale=c(4,0.5),
            
            random.color = FALSE, colors=brewer.pal(8,"Dark2"))
  
  
  result_main <- FindTopicsNumber(
  
    dtmnew_main,
    
    topics = seq(from = 5, to = 20, by = 1),
    
    metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010"),
    
    method = "Gibbs",
    
    control = list(seed = 129),
    
    mc.cores = 2L,
    
    verbose = TRUE
  )
  
  
  FindTopicsNumber_plot(result_main)
 
 df_main<- data.frame(dtmdocs_main=dtmdocs_main,doclen_main=doclen_main,main_f=main_f)
 
  return(df_main)

}

#Call the function

result_positive<- main_function(positive_review)

result_negative <- main_function(negative_review)


#Finding topics for POSITIVE REVIEWS

ldaOut_positive <-LDA(result_positive$dtmdocs_main,18, method="Gibbs", 

                 control=list(iter=1000,seed=1000))

phi_postive <- posterior(ldaOut_positive)$terms %>% as.matrix 

theta_positive <- posterior(ldaOut_positive)$topics %>% as.matrix 

ldaOut.positive <- as.matrix(terms(ldaOut_positive, 10))

ldaOut.positive

ldaOut.topics_pos <- data.frame(topics(ldaOut_positive))

ldaOut.topics_pos$index <- as.numeric(row.names(ldaOut.topics_pos))

positive_review$index <- as.numeric(row.names(positive_review))

datawithtopic_pos <- merge(positive_review, ldaOut.topics_pos, by='index',all.x=TRUE)

datawithtopic_pos <- datawithtopic_pos[order(datawithtopic_pos$index), ]

datawithtopic_pos[,]

probability_positive <- as.data.frame(ldaOut_positive@gamma)

probability_positive[0:10,1:18]


#Finding topics for NEGATIVE REVIEWS

ldaOut_negative <-LDA(result_negative$dtmdocs_main,19, method="Gibbs", 

                 control=list(iter=1000,seed=1000))

phi_negative <- posterior(ldaOut_negative)$terms %>% as.matrix 

theta_negative <- posterior(ldaOut_negative)$topics %>% as.matrix 

ldaOut.negative <- as.matrix(terms(ldaOut_negative, 10))

ldaOut.negative

ldaOut.topics_neg <- data.frame(topics(ldaOut_negative))

ldaOut.topics_neg$index <- as.numeric(row.names(ldaOut.topics_neg))

negative_review$index <- as.numeric(row.names(negative_review))

datawithtopic_neg <- merge(negative_review, ldaOut.topics_neg, by='index',all.x=TRUE)

datawithtopic_neg <- datawithtopic_neg[order(datawithtopic_neg$index), ]

datawithtopic_neg[,]

probability_negative <- as.data.frame(ldaOut_negative@gamma)

probability_negative[0:10,1:19]

#Function for topic prevalence

#Topic prevalence 

function_prev<- function(theta){

  num_doc<- nrow(theta)
  
  num_topic<- ncol(theta)
  
  topic_prev<- colSums(theta)/num_doc
  
  sort_prev<- sort(topic_prev, decreasing = TRUE)}

positive_prevalence<- function_prev(theta_positive)

positive_prevalence

negative_prevalence<- function_prev(theta_negative)

negative_prevalence


#Function for LDAvis

create_ldavis <- function(phi, theta, doc_length, frequency, out_dir = 'vis') {

  vocab <- colnames(phi)
  
  json_lda <- createJSON(phi = phi, theta = theta, 
  
                         vocab = vocab, doc.length = doc_length, 
                         
                         term.frequency = frequency)
  
  serVis(json_lda, out.dir = out_dir, open.browser = TRUE)

}


#Visualizing with LDAVis

LDAvis_positive <- create_ldavis(phi_positive,theta_positive,result_positive$doclen_main,result_positive$main_f,out_dir = 'vis')

LDAvis_negative<- create_ldavis(phi_negative,theta_negative,result_negative$doclen_main,result_negative$main_f,out_dir = 'vis')



