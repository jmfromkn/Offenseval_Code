import pandas as pd
import numpy as np
import spacy
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

np.random.seed(500)
threshold_gold = pd.read_csv("Offenseval2020/task_a_distant/ThresholdGold.tsv", sep='\t', header=0)
threshold_id = threshold_gold['id'].tolist()

def retrieve_table(train_path, test_path):
    test_table = pd.read_csv(train_path, sep='\t', header=0)
    test_table = test_table[~test_table['id'].isin(threshold_id)]
    gold_standard = pd.read_csv(test_path, sep='\t', header=0)
    return test_table, gold_standard

def store_data(train_path, test_path):
    test_table = pd.read_csv(train_path, sep='\t', header=0)
    test_table = test_table[~test_table['id'].isin(threshold_id)]
    testlist = test_table['text'].tolist()
    test_table = None
    gold_standard = pd.read_csv(test_path, sep='\t', header=0)
    threshold_test = gold_standard['text'].tolist()
    ###comment line above if using OLID and uncomment line below
    #threshold_test = gold_standard['tweet'].tolist()

    tweet_test_doc = []
    tweet_test_doc2 = []
    tweet_test_doc3 = []
    tweet_test_doc4 = []
    nlp = spacy.load("spacy_twitterglove_vecs/")
    for tweet in nlp.pipe(testlist, batch_size=1500, n_threads=8):
        tweet_test_doc.append(tweet)
    print ("loop1 complete")
    for tweet in threshold_test:
        testdoc = nlp(tweet)
        tweet_test_doc2.append(testdoc)
    nlp = spacy.load("spacy_twitterw2v_vecs/")
    for tweet in nlp.pipe(testlist, batch_size=1500, n_threads=8):
        tweet_test_doc3.append(tweet)
    print ("loop2 complete")
    for tweet in threshold_test:
        testdoc = nlp(tweet)
        tweet_test_doc4.append(testdoc)
    print ("loop3 complete")

    testlist_BOW = pd.DataFrame((tweet.vector for tweet in tweet_test_doc)).round(8)
    testlist_BOW2 = pd.DataFrame((tweet.vector for tweet in tweet_test_doc3)).round(8)
    testlist_BOW = pd.concat([testlist_BOW, testlist_BOW2], axis=1)
    testlist_BOW2 = None
    tweet_test_doc3 = None
    print ("concatenate 1 complete")
    threshold_BOW = pd.DataFrame((tweet.vector for tweet in tweet_test_doc2)).round(8)
    threshold_BOW2 = pd.DataFrame((tweet.vector for tweet in tweet_test_doc4)).round(8)
    threshold_BOW = pd.concat([threshold_BOW, threshold_BOW2], axis=1)
    threshold_BOW2 = None
    tweet_test_doc4 = None
    print ("concatenate 2 complete")


    testlist_tags = []
    testlist_dep = []
    testlist_ner = []
    threshold_tags = []
    threshold_dep = []
    threshold_ner = []
    nlp = spacy.load('en_core_web_sm')
    for tweet in nlp.pipe(testlist, batch_size=1500, n_threads=8):
        tags = []
        dep = []
        ner = []
        for word in tweet:
            tags.append(word.tag_)
            dep.append(word.dep_ + '_' + word.head.pos_)
        for word in tweet.ents:
            ner.append(word.label_)
        testlist_tags.append(tags)
        testlist_dep.append(dep)
        testlist_ner.append(ner)
    print ("loop 4 complete")
    for tweet in nlp.pipe(threshold_test, batch_size=1500, n_threads=8):
        tags = []
        dep = []
        ner = []
        for word in tweet:
            tags.append(word.tag_)
            dep.append(word.dep_ + '_' + word.head.pos_)
        for word in tweet.ents:
            ner.append(word.label_)
        threshold_tags.append(tags)
        threshold_dep.append(dep)
        threshold_ner.append(ner)
    print ("loop 5 complete")
    return testlist_BOW, threshold_BOW, testlist_tags, testlist_dep, testlist_ner, threshold_tags, threshold_dep, threshold_ner, tweet_test_doc, tweet_test_doc2

def feature_extraction(testlist_BOW, threshold_BOW, testlist_tags, testlist_dep, testlist_ner, threshold_tags, threshold_dep, threshold_ner, tweet_test_doc, tweet_test_doc2, test_dep=True, test_lex=True, test_NER=True, test_POS=True):
    ###Token Level Features + Embeddings
    if test_lex == True:
        punctuation = ['!', '.', ',', ':', '?']
        testlist_BOW['# of Tokens'] = [len(tweet) for tweet in tweet_test_doc]
        threshold_BOW['# of Tokens'] = [len(tweet) for tweet in tweet_test_doc2]
        testlist_BOW['Avg Length of Token'] = [sum(len(word) for word in tweets) / len(tweets) for tweets in tweet_test_doc]
        threshold_BOW['Avg Length of Token'] = [sum(len(word) for word in tweets) / len(tweets) for tweets in tweet_test_doc2]
        testlist_BOW['Avg Punctuation'] = [sum(len(word) for word in tweets if word.text in punctuation) / len(tweets) for
                                           tweets in tweet_test_doc]

        threshold_BOW['Avg Punctuation'] = [sum(len(word) for word in tweets if word.text in punctuation) / len(tweets) for tweets in tweet_test_doc2]
    def word_with_nonalpha_count(tweet):
        freq_counter = 0
        for word in tweet:
            if re.match(r"[a-zA-Z]+[\*&\^\$\?\!][a-zA-Z]*$", word.text) == None:
                pass
            else:
                freq_counter += 1
        return freq_counter

    def check_user_position(index, tweet):
        if tweet[index].text == '@USER':
            return 1
        else:
            return 0
    if test_lex == True:
        testlist_BOW['Alpha+Non-Alpha Tokens'] = [word_with_nonalpha_count(tweet) for tweet in tweet_test_doc]
        threshold_BOW['Alpha+Non-Alpha Tokens'] = [word_with_nonalpha_count(tweet) for tweet in tweet_test_doc2]
        testlist_BOW['@User at Front'] = [check_user_position(0, tweet) for tweet in tweet_test_doc]
        threshold_BOW['@User at Front'] = [check_user_position(0, tweet) for tweet in tweet_test_doc2]
        testlist_BOW['@User at Back'] = [check_user_position(-1, tweet) for tweet in tweet_test_doc]

        threshold_BOW['@User at Back'] = [check_user_position(-1, tweet) for tweet in tweet_test_doc2]

    ###POS_Level Features
    def calc_tag_average(target_tag, tweet_tags):
        freq_count = 0
        token_number = len(tweet_tags)
        for tag in tweet_tags:
            if tag == target_tag:
                freq_count += 1
        if freq_count == 0:
            return 0
        else:
            return freq_count / token_number
    if test_POS == True:
        testlist_BOW['Average Interjections'] = [calc_tag_average('UH', tweet) for tweet in testlist_tags]
        threshold_BOW['Average Interjections'] = [calc_tag_average('UH', tweet) for tweet in threshold_tags]
    if test_dep == True:
        testlist_BOW['Average Expletive'] = [calc_tag_average('expl', tweet) for tweet in testlist_dep]

        threshold_BOW['Average Expletive'] = [calc_tag_average('expl', tweet) for tweet in threshold_dep]

    def already_tokenized(text):
        return text

    tfidf_vect = TfidfVectorizer(tokenizer=already_tokenized, lowercase=False, sublinear_tf=True)
    train_dependency = tfidf_vect.fit_transform(testlist_dep)
    train_deps = tfidf_vect.get_feature_names()
    train_dependency_counts = pd.DataFrame(train_dependency.toarray(), columns=train_deps)

    threshold_dependency = tfidf_vect.transform(threshold_dep)
    test_deps = tfidf_vect.get_feature_names()
    threshold_dependency_counts = pd.DataFrame(threshold_dependency.toarray(), columns=test_deps)


    train_pos = tfidf_vect.fit_transform(testlist_tags)
    train_tags = tfidf_vect.get_feature_names()
    train_pos_counts = pd.DataFrame(train_pos.toarray(), columns=train_tags)
    threshold_pos = tfidf_vect.transform(threshold_tags)
    test_tags = tfidf_vect.get_feature_names()
    threshold_pos_counts = pd.DataFrame(threshold_pos.toarray(), columns=test_tags)
    train_ner = tfidf_vect.fit_transform(testlist_ner)
    train_nertags = tfidf_vect.get_feature_names()
    train_ner_counts = pd.DataFrame(train_ner.toarray(), columns=train_nertags)
    threshold_nercounts = tfidf_vect.transform(threshold_ner)
    test_nertags = tfidf_vect.get_feature_names()
    threshold_ner_counts = pd.DataFrame(threshold_nercounts.toarray(), columns=test_nertags)
    if test_POS == False and test_dep == False and test_NER == False:
        pass
    elif test_POS == False and test_dep == False and test_NER == True:
        testlist_BOW = pd.concat([testlist_BOW, train_ner_counts], axis=1)
        threshold_BOW = pd.concat([threshold_BOW, threshold_ner_counts], axis=1)
    elif test_POS == False and test_dep == True and test_NER == False:
        testlist_BOW = pd.concat([testlist_BOW, train_dependency_counts], axis=1)
        threshold_BOW = pd.concat([threshold_BOW, threshold_dependency_counts], axis=1)
    elif test_POS == True and test_dep == False and test_NER == False:
        testlist_BOW = pd.concat([testlist_BOW, train_pos_counts], axis=1)
        threshold_BOW = pd.concat([threshold_BOW, threshold_pos_counts], axis=1)
    elif test_POS == False and test_dep == True and test_NER == True:
        testlist_BOW = pd.concat([testlist_BOW, train_dependency_counts, train_ner_counts], axis=1)
        threshold_BOW = pd.concat([threshold_BOW, threshold_dependency_counts, threshold_ner_counts], axis=1)
    elif test_dep == False and test_POS == True and test_NER == True:
        testlist_BOW = pd.concat([testlist_BOW, train_pos_counts, train_ner_counts], axis=1)
        threshold_BOW = pd.concat([threshold_BOW, threshold_pos_counts, threshold_ner_counts], axis=1)
    elif test_NER == False and test_POS == True and test_dep == True:
        testlist_BOW = pd.concat([testlist_BOW, train_dependency_counts, train_pos_counts], axis=1)
        threshold_BOW = pd.concat([threshold_BOW, threshold_dependency_counts, threshold_pos_counts], axis=1)
    else:
        testlist_BOW = pd.concat([testlist_BOW, train_dependency_counts, train_pos_counts, train_ner_counts], axis=1)
        threshold_BOW = pd.concat([threshold_BOW, threshold_dependency_counts, threshold_pos_counts, threshold_ner_counts], axis=1)



    ###NER Features
    def check_NER(entity, tweet):
        for word in tweet:
            if word == entity:
                return 1
        return 0
    if test_NER == True:
        testlist_BOW['Has Person'] = [check_NER('PERSON', tweet) for tweet in testlist_ner]
        testlist_BOW['Has Organization'] = [check_NER('ORG', tweet) for tweet in testlist_ner]
        testlist_BOW['Has Nationality/Religion'] = [check_NER('NORP', tweet) for tweet in testlist_ner]
        testlist_BOW['Has Place'] = [check_NER('GPE', tweet) for tweet in testlist_ner]
        threshold_BOW['Has Person'] = [check_NER('PERSON', tweet) for tweet in threshold_ner]
        threshold_BOW['Has Organization'] = [check_NER('ORG', tweet) for tweet in threshold_ner]
        threshold_BOW['Has Nationality/Religion'] = [check_NER('NORP', tweet) for tweet in threshold_ner]
        threshold_BOW['Has Place'] = [check_NER('GPE', tweet) for tweet in threshold_ner]


    features = testlist_BOW

    gold_features = threshold_BOW
    print ("3")
    return features, gold_features


def svm_model(train_data, test_data, train_features, test_features, threshold_value, path):
    test_threshold_value = threshold_value

    test_conf_values = train_data['average']
    test_conf_values2 = test_conf_values.tolist()

    test_labels = []

    for value in test_conf_values2:
        if value >= test_threshold_value:
            test_labels.append("OFF")
        elif value <= test_threshold_value:
            test_labels.append("NOT")


    train_data['labels'] = test_labels


    Train_X = train_features

    Test_X = test_features

    Train_Y = train_data['labels']

    Test_Y = test_data['labels']

    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)

    SVM = svm.LinearSVC(dual=False)
    SVM.fit(Train_X, Train_Y)

    predictions_SVM = SVM.predict(Test_X)
    print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y) * 100)
    print("SVM F1 Score: ", f1_score(predictions_SVM, Test_Y, average='macro'))
    return Encoder.inverse_transform(predictions_SVM)

def predict(train_path, test_path, trainfeatures, testfeatures):
    ideal_threshold = 0.40
    base_tables = retrieve_table(train_path, test_path)
    prediction = svm_model(base_tables[0], base_tables[1], trainfeatures, testfeatures, 0.40, train_path)
    return prediction


###Replace csv paths with your own feature.csv paths
stored_train = pd.read_csv("TrainFeatures.csv", header=0)
stored_test = pd.read_csv("ThresholdGoldFeatures.csv", header=0)

###Replace first two arguments of predict with your train and test path
print ("Baseline: Only Embeddings")
full_train_baseline = predict("Offenseval2020/task_a_distant/1MTweetRandomSample.tsv", "Offenseval2020/task_a_distant/ThresholdGold.tsv", stored_train.iloc[:, 0:399], stored_test.iloc[:, 0:399])
print ("Only add NER")
full_train_NER = predict("Offenseval2020/task_a_distant/1MTweetRandomSample.tsv", "Offenseval2020/task_a_distant/ThresholdGold.tsv", stored_train.iloc[:, np.r_[0:399, 1197:1218]], stored_test.iloc[:, np.r_[0:399, 1197:1218]])
print ("Only add POS")
full_train_POS = predict("Offenseval2020/task_a_distant/1MTweetRandomSample.tsv", "Offenseval2020/task_a_distant/ThresholdGold.tsv", stored_train.iloc[:, np.r_[0:399, 1146:1196]], stored_test.iloc[:, np.r_[0:399, 1146:1196]])
print ("Only add LEX")
full_train_LEX = predict("Offenseval2020/task_a_distant/1MTweetRandomSample.tsv", "Offenseval2020/task_a_distant/ThresholdGold.tsv", stored_train.iloc[:, np.r_[0:399, 400:407]], stored_test.iloc[:, np.r_[0:399, 400:407]])
print ("Only add DEP")
full_train_DEP = predict("Offenseval2020/task_a_distant/1MTweetRandomSample.tsv", "Offenseval2020/task_a_distant/ThresholdGold.tsv", stored_train.iloc[:, np.r_[0:399, 408:1145]], stored_test.iloc[:, np.r_[0:399, 408:1145]])
print ("Syntactic Suite + Lex: POS + Dep + Lex, NO NER")
full_train_SYN_LEX = predict("Offenseval2020/task_a_distant/1MTweetRandomSample.tsv", "Offenseval2020/task_a_distant/ThresholdGold.tsv", stored_train.iloc[:, 0:1196], stored_test.iloc[:, 0:1196])
print ("Without POS Features")
full_train_no_POS = predict("Offenseval2020/task_a_distant/1MTweetRandomSample.tsv", "Offenseval2020/task_a_distant/ThresholdGold.tsv", stored_train.iloc[:, np.r_[0:1145, 1197-1218]], stored_test.iloc[:, np.r_[0:1145, 1197-1218]])
print ("Without LEX features")
full_train_no_LEX = predict("Offenseval2020/task_a_distant/1MTweetRandomSample.tsv", "Offenseval2020/task_a_distant/ThresholdGold.tsv", stored_train.iloc[:, np.r_[0:399, 408:1218]], stored_test.iloc[:, np.r_[0:399, 408:1218]])
print ("Without DEP Features")
full_train_no_DEP = predict("Offenseval2020/task_a_distant/1MTweetRandomSample.tsv", "Offenseval2020/task_a_distant/ThresholdGold.tsv", stored_train.iloc[:, np.r_[0:407, 1146:1218]], stored_test.iloc[:, np.r_[0:407, 1146:1218]])
print ("Full Feature")
full_train = predict("Offenseval2020/task_a_distant/1MTweetRandomSample.tsv", "Offenseval2020/task_a_distant/ThresholdGold.tsv", stored_train, stored_test)

testing = pd.read_csv("Offenseval2020/task_a_distant/ThresholdGold.tsv",sep='\t', header=0)
testing['baseline pred'] = full_train_baseline
testing['Only NER pred'] = full_train_NER
testing['Only POS pred'] = full_train_POS
testing['Only LEX pred'] = full_train_LEX
testing['Only DEP pred'] = full_train_DEP
testing['Full Syn + Lex no NER pred'] = full_train_SYN_LEX
testing['Without POS features pred'] = full_train_no_POS
testing['Without LEX Features pred'] = full_train_no_LEX
testing['Without DEP features pred'] = full_train_no_DEP
testing['Full Features'] = full_train
testing.to_csv("dev_predictions.csv", header=False, index=False)

