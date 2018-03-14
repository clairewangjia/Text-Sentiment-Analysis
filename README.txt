Lab 2: Sentiment Analysis
Deadline: 5 Mar 2018(Mon, 1800 Hrs)
Student: Wang Jia / A0176605B / E0232209

### Environment Setting
1. Windows 10
2. Python 2.7


### Installation 
0. emoji
1. nltk 
2. simplejson
3. pickle
4. numpy
5. scipy
6. scikit-learn
Note: please install emoji package for emoji description conversion via pip. (e.g., pip install emoji)  


### Usage 
1. Run 'Step1_preprocess.py' to prepocess tweet content. (e.g. python Step1_preprocess.py)
2. Run 'Step1_social_hashtag_preprocess.py' to extract social features and hashtags.
3. Run 'Step2_fused_analyzer.py'. 
Noted that 'vader_sentiment_analyzer.py' and 'hashtag.py' will be called automatically. The former one is to process text data by VADER sentiment analyzer, the latter is to train hashtag classifier. You are supposed to see performances (average percision, average recall, f1 score) printing into the screen for 3 individual models and 2 fused models:
	a. Social feature RF classifier
	b. VADER sentiment classifier
	c. hashtag classifier
	d. 2-feature fused model
	e. 3-feature fused model