import collections
import re

from nltk.stem.wordnet import WordNetLemmatizer
from nltk import SnowballStemmer
from nltk.corpus import stopwords
import unicodedata

stop_words = set(stopwords.words('english'))
lmt = WordNetLemmatizer()


def main():
    preprocessSet()

def preprocessSet():
    with open("DataSet/ProcessedTestData/UnlabledProcessed_http_handle.txt", "w+", encoding='utf-8') as processed_set:

        with open("DataSet/TestData/test_tweets_unlabeled.txt", "r", encoding='utf-8') as train_set:
            for line in train_set:
                # processed_tweet = ""
                # list_values = ("".join(line)).split('\t')
            
                # processed_set.write(list_values[0].strip())
                # processed_set.write("\t")

                # for words in processTweet(list_values[1].strip()):
                #     processed_tweet.append("".join(words))

                # processed_tweet = (" ".join(processTweet(list_values[1].strip()))).strip()
                processed_tweet = processTweet(line.strip())
                processed_set.write(processed_tweet)
                processed_set.write("\n")

            # X_pre.append(process_tweet(line_split[1].strip()))
            # y.append(line_split[0])

def processTweet(text):
    all_words = {}
    normalised_tokens = []

    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    normalised_tokens = []
    split_text = [re.sub(r"[^@a-zA-Z0-9']+", ' ', k).lower().strip() for k in text.split(" ")]
    for item in split_text:
        if item not in stop_words:
            # normalised_item = lmt.lemmatize(item)
            normalised_item = item
            if normalised_item in all_words:
                all_words[normalised_item]+=1
            else:
                all_words[normalised_item]=1
            normalised_tokens.append(item)
    normalised_tweet = " ".join([x for x in normalised_tokens])
    return normalised_tweet

    # tweet_text = re.sub(r'^https?:\/\/.*[\r\n]*', '', tweet_text, flags=re.MULTILINE)
    # print("RegEX: ", tweet_text)

    # split_text = [re.sub(r"[^a-zA-Z0-9']+", ' ', k).lower().strip() for k in tweet_text.split(" ")]

    #split_text = [k.lower().strip().replace("rt", "").replace("@handle:", "").replace("@handle", "") for k in tweet_text.split(" ")]
    # split_text = [k.lower().strip().replace("rt", "").replace("handle:", "").replace("handle", "") for k in tweet_text.split(" ")]

    # for word in split_text:
    #     if word not in stop_words and ("".join(word)).find("http:"):
    #         normalised_item = lmt.lemmatize(word)
    #         if normalised_item in all_words:
    #             all_words[normalised_item] += 1
    #         else:
    #             all_words[normalised_item] = 1
    #         normalised_tokens.append(word)
    # return normalised_tokens

main()