import collections

def main():
    developmentSet()

def developmentSet():
    line_count = 0

    with open("DataSet/DevelopmentDataSet/DevelomentSet.txt", "w+", encoding='utf-8') as dev_set:
        with open("DataSet/RawData/train_tweets.txt", "r", encoding='utf-8') as train_set:
            for tweet in train_set:
                line_count += 1
                if (line_count % 4) == 0:
                    tweet_line = ("".join(tweet).split("\t"))

                    dev_set.write(tweet_line[1].strip())
                    dev_set.write("\n")
                    # line_count += 1

                # else:
                #     line_count += 1

main()