#PART 2

install.packages('stringr')
install.packages('RColorBrewer')
install.packages('topicmodels')
install.packages('ggplot2')
install.packages('LDAvis')
install.packages('servr')
install.packages('textcat')
install.packages('ldatuning')
install.packages('wordcloud')
install.packages("syuzhet")
install.packages("tidytext")
install.packages("tidyr")

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
library(tidytext)
library(syuzhet)
library(tidyr)

set.seed(129)

data <- read.csv(file = "HotelsData.csv", header = TRUE)
data$lang <- sapply(data$Text.1, textcat)
write.csv(data, "data_lang.csv")
rev_english <- data %>%
  filter(data$lang == "english")
colnames(rev_english)

test <- test %>% mutate(review_id = row_number())

tokens <- test %>%
  unnest_tokens(word, Text.1) %>%
  select(review_id, word)

#SENTIMENT ANALYSIS 
bing_sentiment <- tokens %>%
  inner_join(get_sentiments("bing"), by = "word") %>%
  count(sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment_score = positive - negative)

print(bing_sentiment)

review_sentiment <- tokens %>%
  inner_join(get_sentiments("bing"), by = "word") %>%
  group_by(review_id) %>%
  summarize(sentiment_score = sum(ifelse(sentiment == "positive", 1, -1)))

test <- test %>%
  left_join(review_sentiment, by = "review_id")

print(head(test))


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

