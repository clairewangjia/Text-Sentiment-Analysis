import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import vader_sentiment_analyzer
import hashtag

data_dir = './data' 

print("Loading data...")

x = pd.read_csv('./data/features_processed.csv')
x = x.values # convert pandas dataframe to ndarray

with open(os.path.join(data_dir, 'labels.txt'), 'r') as f:
    y = np.array([ int(line.strip()) for line in f.readlines()])
print("----------------------------------------------------")
print("Training RF classifer with features: favourites, followers, friends, likes, lists, retweets, statuses")
print("----------------------------------------------------")

n = 3
prob_count= np.empty(shape=[0, n])
kf = KFold(n_splits=10)
avg_p = 0
avg_r = 0
macro_f1 = 0
for train, test in kf.split(x):
    # model = MultinomialNB().fit(x[train], y[train])
#     model = KNeighborsClassifier(n_neighbors=3).fit(x[train], y[train])
    model = RandomForestClassifier(n_estimators = 300, max_features = 4, random_state=0).fit(x[train], y[train])
    # model = LogisticRegression().fit(x[train], y[train])
    # model = Lasso(alpha = 0.1).fit(x[train], y[train])
    # model = svm.SVC(probability=True).fit(x[train], y[train])
    prob = model.predict_proba(x[test])
    predicts = model.predict(x[test])
    prob_count = np.concatenate((prob_count, prob),axis=0)
    # print(classification_report(y[test],predicts))
    
    avg_p	+= precision_score(y[test],predicts, average='macro')
    avg_r	+= recall_score(y[test],predicts, average='macro')
    macro_f1  += f1_score(y[test],predicts, average='macro')
print('Feature importances of the above features in the RandomForestClassifier:')
print(model.feature_importances_)
print('probability: neg, neu, pos')
print(prob_count)
print('Average Precision of features_set_classifier is %f.' %(avg_p/10.0))
print('Average Recall of features_set_classifier is %f.' %(avg_r/10.0))
print('Average Macro-F1 of features_set_classifier is %f.' %(macro_f1/10.0))
print("----------------------------------------------------")
print("Training VADER sentiment classifer for tweets text and emoji")
prob_text = vader_sentiment_analyzer.main()
print("----------------------------------------------------")
print("Training ML TF-IDF classifer for hashtag")
print("----------------------------------------------------")
prob_hashtag = hashtag.main()

# fuse two models
print("----------------------------------------------------")
print("Combining 2 models in the rule-based late fusion model: text + social features")
print("----------------------------------------------------")
weights = []
for w1 in np.arange(0,0.8,0.01):
    w2 = 1-w1
    weights.append([w1,w2])


precisions = []
recalls = []
macrof1 = []

for i in range(len(weights)):
    w_count, w_text = weights[i]
    result_prob = w_count*prob_count + w_text*prob_text
    result = np.argmax(result_prob, axis=1)
    avg_p = precision_score(y, result, average='macro')
    avg_r = recall_score(y, result, average='macro')
    macro_f1  = f1_score(y,result, average='macro')

    precisions.append(avg_p)
    recalls.append(avg_r)
    macrof1.append(macro_f1)


opt_id = np.argmax(macrof1)
print('Weight of social_feature_classifier: ' + str(weights[opt_id][0]) + ', Weight of vader_sentiment_analyzer: '+ str(weights[opt_id][1]))
print('Optimal Precision of late fusion model is %f.' %precisions[opt_id])
print('Optimal Recall of late fusion model is %f.' %recalls[opt_id])
print('Optimal Macro-F1 of late fusion model is %f.' %macrof1[opt_id])


# fuse 3 models
print("----------------------------------------------------")
print("Combining 3 models in the rule-based late fusion model: text + social features + hashtags ")
print("----------------------------------------------------")

weights = []
for w1 in np.arange(0,0.7,0.01):
    for w2 in np.arange(0,0.7,0.01):
        w3 = 1-w1-w2
        weights.append([w1, w2, w3])

precisions = []
recalls = []
macrof1 = []

for i in range(len(weights)):
    w_count, w_text, w_hashtag = weights[i]
    result_prob = w_count*prob_count + w_text*prob_text + w_hashtag*prob_hashtag
    result = np.argmax(result_prob, axis=1)
    avg_p = precision_score(y, result, average='macro')
    avg_r = recall_score(y, result, average='macro')
    macro_f1  = f1_score(y,result, average='macro')

    precisions.append(avg_p)
    recalls.append(avg_r)
    macrof1.append(macro_f1)


opt_id = np.argmax(macrof1)
print('Weight of social_feature_classifier: ' + str(weights[opt_id][0]))
print('Weight of vader_sentiment_analyzer: '+ str(weights[opt_id][1]))
print('Weight of hashtag_classifier: '+ str(weights[opt_id][2]))
print('Optimal Precision of late fusion model is %f.' %precisions[opt_id])
print('Optimal Recall of late fusion model is %f.' %recalls[opt_id])
print('Optimal Macro-F1 of late fusion model is %f.' %macrof1[opt_id])