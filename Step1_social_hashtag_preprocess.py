# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
import re
import nltk
# nltk.download()
from nltk.corpus import stopwords
import simplejson as json
import pickle
import numpy as np
import pandas as pd


def rm_html_tags(str):
    html_prog = re.compile(r'<[^>]+>',re.S)
    return html_prog.sub('', str)

def rm_html_escape_characters(str):
    pattern_str = r'&quot;|&amp;|&lt;|&gt;|&nbsp;|&#34;|&#38;|&#60;|&#62;|&#160;|&#20284;|&#30524;|&#26684|&#43;|&#20540|&#23612;'
    escape_characters_prog = re.compile(pattern_str, re.S)
    return escape_characters_prog.sub('', str)

def rm_at_user(str):
    return re.sub(r'@[a-zA-Z_0-9]*', '', str)

def rm_url(str):
    return re.sub(r'http[s]?:[/+]?[a-zA-Z0-9_\.\/]*', '', str)

def rm_repeat_chars(str):
    return re.sub(r'(.)(\1){2,}', r'\1\1', str)

def rm_hashtag_symbol(str):
    return re.sub(r'#', '', str)

def replace_emoticon(emoticon_dict, str):
    for k, v in emoticon_dict.items():
        str = str.replace(k, v)
    return str

def rm_time(str):
    return re.sub(r'[0-9][0-9]:[0-9][0-9]', '', str)

def rm_punctuation(current_tweet):
    return re.sub(r'[^\w\s]','',current_tweet)


def pre_process(str, porter):
    # do not change the preprocessing order only if you know what you're doing 
    str = str.lower()
    str = rm_url(str)        
    str = rm_at_user(str)        
    str = rm_repeat_chars(str) 
    str = rm_hashtag_symbol(str)       
    str = rm_time(str)        
    str = rm_punctuation(str)
        
    try:
        str = nltk.tokenize.word_tokenize(str)
        try:
            str = [porter.stem(t) for t in str]
        except:
            print(str)
            pass
    except:
        print(str)
        pass
        
    return str


if __name__ == "__main__":
    data_dir = './data'  ##Setting your own file path here.

    x_filename = 'tweets.txt'
    y_filename = 'labels.txt'

    porter = nltk.PorterStemmer()
    stops = set(stopwords.words('english'))
    stops.add('rt')

    ##load and process samples
    print('start extract social features...')
    retweets = []
    likes = []
    friends = []
    followers = []
    lists = []
    favourites = []
    statuses = []
    cnt = 0
    with open(os.path.join(data_dir, x_filename)) as f:
        for i, line in enumerate(f):
            tweet_obj = json.loads(line.strip(), encoding='utf-8')
            retweet = tweet_obj['retweet_count']
            like = tweet_obj['favorite_count']
            friend = tweet_obj['user']['friends_count']
            follower = tweet_obj['user']['followers_count']
            listed = tweet_obj['user']['listed_count']
            favourite = tweet_obj['user']['favourites_count']
            statuse = tweet_obj['user']['statuses_count']
            retweets.append(retweet)
            likes.append(like)
            friends.append(friend)      
            followers.append(follower)
            lists.append(listed)
            favourites.append(favourite)
            statuses.append(statuse)


    tweets_df = pd.DataFrame(
        {'retweets': retweets,
         'likes': likes,
         'friends': friends,
         'followers': followers,
         'lists': lists,
         'favourites': favourites,
         'statuses': statuses
        })

    print('Samples of extracted features...')
    print tweets_df.head(5)

    ###Save df to csv
    tweets_df.to_csv('./data/features_processed.csv', index=False)


    print("Social features preprocessing is completed")
    ##load and process hashtags
    print("Start extract and process hashtags...")
   
    words_stat = {}
    hashtags = []
    cnt = 0
    with open(os.path.join(data_dir, x_filename)) as f:
        for i, line in enumerate(f):
            postprocess_tweet = []
            tweet_obj = json.loads(line.strip(), encoding='utf-8')
            hashtag_dict = tweet_obj['entities']['hashtags']
            if len(hashtag_dict) > 0:
                hashtag_list = []
                for h in hashtag_dict:
                    single_tag = h['text']
                    hashtag_list.append(single_tag)
                hashtag = ' '.join(hashtag_list)
            else:
                hashtag = ""            
            words = pre_process(hashtag, porter)
            for word in words:
                if word not in stops:
                    postprocess_tweet.append(word)
                    if word in words_stat.keys():
                        words_stat[word][0] += 1
                        if i != words_stat[word][2]:
                            words_stat[word][1] += 1
                            words_stat[word][2] = i
                    else:
                        words_stat[word] = [1,1,i]
            hashtags.append(' '.join(postprocess_tweet))
    ##saving the statistics of tf and df for each words into file
    print("The number of unique words in hashtag data set is %i." %len(words_stat.keys()))
    lowTF_words = set()
    # with open(os.path.join(data_dir, 'words_statistics_hashtag.txt'), 'w', encoding='utf-8') as f:
    #     f.write('TF\tDF\tWORD\n')
    for word, stat in sorted(words_stat.items(), key=lambda i: i[1], reverse=True):
        if stat[0]<2:
            lowTF_words.add(word)
    print("The number of low frequency words of hashtags is %d." %len(lowTF_words))
    # print(stops)
            
    ###Re-process samples, filter low frequency words...
    fout = open(os.path.join(data_dir, 'hashtag_processed.txt'), 'w')
    hashtag_new = []
    for hashtag in hashtags:
        words = hashtag.split(' ')
        new = [] 
        for w in words:
            if w not in lowTF_words:
                new.append(w)
        new_hashtag = ' '.join(new)
        hashtag_new.append(new_hashtag)
        fout.write('%s\n' %new_hashtag)
    fout.close()

    print("Hashtag preprocessing is completed")