import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.linear_model import SGDClassifier

def main():
# if __name__ == "__main__":
	data_dir = './data' 

	print("Loading data...")
	with open(os.path.join(data_dir, 'hashtag_processed.txt'), 'r') as f_hashtag:
		x_hashtag = f_hashtag.readlines()

	with open(os.path.join(data_dir, 'labels.txt'), 'r') as f:
		y = np.array(f.readlines())
	    
	print("Extract features...")

	x_hashtag_feats = TfidfVectorizer().fit_transform(x_hashtag)

	print(x_hashtag_feats.shape)


	print("Start training and predict...")
	n = 3

	prob_hashtag= np.empty(shape=[0, n])


	kf = KFold(n_splits=10)
	avg_p = 0
	avg_r = 0
	macro_f1 = 0

	for train, test in kf.split(x_hashtag_feats):
	    model = MultinomialNB().fit(x_hashtag_feats[train], y[train])
	    # model = KNeighborsClassifier(n_neighbors=7).fit(x_hashtag_feats[train], y[train])
	    # model = RandomForestClassifier(n_estimators=500, max_features=7, random_state=0).fit(x_hashtag_feats[train], y[train])
	    # model = LogisticRegression().fit(x_hashtag_feats[train], y[train])    
	    # model = SGDClassifier(loss='log', penalty='l2', alpha=1e-3, random_state=0, max_iter=100).fit(x_hashtag_feats[train], y[train])
	    prob = model.predict_proba(x_hashtag_feats[test])
	    predicts = model.predict(x_hashtag_feats[test])
	    # print(classification_report(y[test],predicts))
	    prob_hashtag = np.concatenate((prob_hashtag, prob))
	    avg_p   += precision_score(y[test],predicts, average='macro')
	    avg_r   += recall_score(y[test],predicts, average='macro')
	    macro_f1  += f1_score(y[test],predicts, average='macro')

	print('Average Precision of hashtag classifer is %f.' %(avg_p/10.0))
	print('Average Recall of hashtag classifer is %f.' %(avg_r/10.0))
	print('Average Macro-F1 of hashtag classifer is %f.' %(macro_f1/10.0))
	return prob_hashtag