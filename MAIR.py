import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# functions that deletes all duplicates for labels exept reqmore, restart, deny and repeat
def delete_duplicates_alt(df):
    df_filtered = df[df['dialog_act'].isin(['inform', 'request', 'confirm', 'ack', 'affirm', 'hello', 'reqalts', 'null', 'negate', 'bye', 'thankyou'])]
    df_filtered_no_duplicates = df_filtered.drop_duplicates(subset='utterance_content', keep='first')
    df_remaining = df[~df['dialog_act'].isin(['inform', 'request', 'confirm', 'ack', 'affirm', 'hello', 'reqalts', 'null', 'negate', 'bye', 'thankyou'])]
    df_cleaned = pd.concat([df_filtered_no_duplicates, df_remaining]).reset_index(drop=True)

    return df_cleaned

# Opens the dataframe
df = pd.read_csv('./dialog_acts.dat', names=['dialog_act', 'utterance_content'])
df[['dialog_act', 'utterance_content']] = df['dialog_act'].str.split(' ', n=1, expand=True)

# Deletes duplicates from DF.
df_unique = df.drop_duplicates(subset='utterance_content', keep='first')
df_unique_alt = delete_duplicates_alt(df)

# Split the full dataset in a training part of 85% and a test part of 15%.
train_df, test_df = train_test_split(df_unique_alt, test_size=0.15, random_state=1)

# Obtains average utterance length in the dataset
def average_length(dataframe):
    utterance_lengths = []
    for index, row in dataframe.iterrows():
        utterance_lengths.append(len(row['utterance_content']))
    
    average = sum(utterance_lengths) / len(utterance_lengths)

    return average

# Function that counts amount of instances of all unique labels in the dataframe (just to get insight in the data)
def label_count(dataframe):
    # Creates dictionary of all unique labels
    labels = set(dataframe['dialog_act'].unique())
    labels_dict = dict.fromkeys(labels, 0)
    total_count = 0

    # Counts the amount of instances of labels in dataset
    for index, row in dataframe.iterrows():
        labels_dict[row['dialog_act']] += 1
        total_count = total_count + 1
    
    # Prints the amounts
    print('Instance count per label: ')
    for label, amount in labels_dict.items():
        print(label, ' : ', amount)
    
    print('total: ', total_count)
    
    print()


# deletes duplicates in the dataframe
def delete_duplicates(dataframe):
    print('original length: ', len(dataframe))
    dropped_dataframe = dataframe.drop_duplicates()
    print('new length: ', len(dropped_dataframe))
    return dropped_dataframe


def delete_duplicates_alt(df):
    df_filtered = df[df['dialog_act'].isin(['inform', 'request', 'confirm', 'ack', 'affirm', 'deny', 'hello', 'reqalts', 'null', 'negate', 'bye', 'thankyou'])]
    df_filtered_no_duplicates = df_filtered.drop_duplicates(subset='utterance_content', keep='first')
    df_remaining = df[~df['dialog_act'].isin(['inform', 'request', 'confirm', 'ack', 'affirm', 'deny', 'hello', 'reqalts', 'null', 'negate', 'bye', 'thankyou'])]
    df_cleaned = pd.concat([df_filtered_no_duplicates, df_remaining]).reset_index(drop=True)

    return df_cleaned



# Quick function to reuse for asking prompts from user and converts to lower case
def ask_utterance():
    utterance = input("Insert prompt or type 'quit': ").lower()
    return utterance


# Classifier 0 == majority_baseline
# Classifier 1 == keyword_baseline
# Classifier 2 == Decision trees
# Classifier 4 == K-Nearest neighbors
def perform_classification(dataframe, classifier, train_df=train_df):
    while True:
        utterance = ask_utterance()

         # Checks if user wants to exit
        if utterance == 'quit':
            break
        
        # Performs the chosen way of classificaiton
        elif classifier == 0:
            label = majority_baseline(dataframe)
            print('Utterance is classified as: ', label, '\n')
        elif classifier == 1:
            label = keyword_baseline(utterance)
            print('Utterance is classified as: ', label, '\n')
        elif classifier == 2:
            # First machine learning technique: Decision trees
            tree, vectorizer, label_encoder = CreateTree(train_df)
            utterance_vec = vectorizer.transform([utterance]) # DOES NOT WORK YET
            label_int = tree.predict(utterance_vec)
            label_text = label_encoder.inverse_transform(label_int)

            print('Utterance is classified as: ', label_text, '\n')
        else:
            # Second machine learning technique: K nearest neighbors (K=5)
            kn, vectorizer, label_encoder = CreateKNearest(dataframe)
            utterance_vec = vectorizer.transform([utterance])
            label_int = kn.predict(utterance_vec) # DOES NOT WORK YET
            label_text = label_encoder.inverse_transform(label_int)

            print('Utterance is classified as: ', label_text, '\n')
            


# IMPLEMENT: A baseline system that, regardless of the content of the utterance, always assigns the majority class of in the data. 
# In the current dataset this is the inform label (almost 40% of all utterances).
# In both cases, think about the data you’re working with to develop your systems (i.e. think about making sure you’re not (accidentally) ‘training’ on your test data). 
# Your code should offer a prompt to enter a new utterance and classify this utterance, and repeat the prompt until the user exits.
def majority_baseline(dataframe):
    # Obtains majority label from dataframe
    majority_label = dataframe['dialog_act'].value_counts().idxmax()
    return majority_label


# A baseline system that classifies an uttertance based on keywords
def keyword_baseline(utterance):
    label_keywords = {
        'inform' : ['any', 'looking'],
        'request' : ['what', 'whats'],
        'thankyou' : ['thank', 'thanks'],
        'reqalts' : ['else', 'about'], 
        'null' : ['noise', 'cough'],
        'affirm' : ['yes'],
        'negate' : ['no'],
        'bye' : ['bye', 'goodbye', 'adieu'],
        'confirm' : ['does', 'serve'],
        'repeat' : ['repeat', 'again'],
        'ack' : ['okay', 'um'],
        'hello' : ['hello', 'hi'],
        'deny' : ['wrong'],
        'restart' : ['start', 'again'],
        'reqmore' : ['more']
    }

    # Splits prompt into words
    utterance_split = utterance.split(' ')

    # Loops through keywords and words and classifies the utterance accordingly
    keyword_found_check = False
    for label, keywords in label_keywords.items():
        for word in utterance_split:
            if word in keywords:
                keyword_found_check = True
                return label
    if not keyword_found_check:        
        keyword_found_check = False
        return 'inform'


# Calculates accuracy of baseline models on a given test dataframe
def check_baseline_performance(dataframe, classifier):
    total_classifications = 0
    correct_classifications = 0

    for index, row in dataframe.iterrows():
        utterance = row['utterance_content']
        
        if classifier == 0:
            label = majority_baseline(dataframe)
            if label == row['dialog_act']:
                correct_classifications += 1
            total_classifications += 1
            

        elif classifier == 1:
            label = keyword_baseline(utterance)
            if label == row['dialog_act']:
                correct_classifications += 1
            total_classifications += 1
        
    print('The accuracy is: ', (correct_classifications / total_classifications) * 100, '% \n')




# Train a minimum of two different machine learning classifiers on the dialog act data. 
  # Possible classifiers include Decision Trees, Logistic Regression, or a Feed Forward neural network.
  # 
  # Use a bag of words representation as input for your classifier. 
  # Depending on the classifier that you use and the setup of your machine learning pipeline you may need to keep an integer (for example 0) 
  # for out-of-vocabulary words, i.e., when a test sentence is entered that contains a word which was not in the training data, 
  # and therefore the word is not in the mapping, assign the special integer. After training, testing, and reporting performance, 
  # the program should offer a prompt to enter a new sentence and classify this sentence, and repeat the prompt until the user exits.


# https://scikit-learn.org/stable/modules/tree.html#classification 1.10.1. classification
# https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features preprocessing
def CreateTree(df=train_df, target_column='dialog_act', feature_column='utterance_content'):
    features_train = df[feature_column]
    target_train = df[target_column]
    
    # Retains only unique labels
    labels = set(df[target_column].unique())

    # Count vectorizer lijkt wel te werken
    vectorizer = CountVectorizer()
    features_encoded = vectorizer.fit_transform(features_train)

    # label encoder voor de target column
    label_encoder = preprocessing.LabelEncoder()
    target_encoded = label_encoder.fit_transform(target_train)

    # Fit data to decision tree
    clf = DecisionTreeClassifier()
    clf.fit(features_encoded, target_encoded)

    return clf, vectorizer, label_encoder

def TestTree(tree, vectorizer, label_encoder, df=test_df, target_column='dialog_act', feature_column='utterance_content'):
    features_test = df[feature_column]
    target_test = df[target_column]

    # Count vectorizer
    features_encoded = vectorizer.transform(features_test)

    # Label encoder for the target column
    target_encoded = label_encoder.transform(target_test)

    target_predict = tree.predict(features_encoded)

    # Ensure target_names matches the number of unique classes
    target_names = ['inform', 'request', 'confirm', 'ack', 'affirm', 'hello', 'reqalts', 'null', 'negate', 'bye', 'thankyou', 'deny', 'restart', 'reqmore']
    report = classification_report(target_encoded, target_predict, target_names=target_names)

    print(report)

# https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification 1.6.2. Nearest Neighbors Classification
# --> weights = 'distance'
def CreateKNearest(train_df=train_df, target_column='dialog_act', feature_column='utterance_content'):
    features_train = train_df[feature_column]
    target_train = train_df[target_column]
    
    # Retains only unique labels
    labels = set(train_df[target_column].unique())

    # Count vectorizer
    vectorizer = CountVectorizer()
    features_encoded = vectorizer.fit_transform(features_train)

    # label encoder voor de target column
    label_encoder = preprocessing.LabelEncoder()
    target_encoded = label_encoder.fit_transform(target_train)
    
    #K nearest neighbors classifier maken en fitten
    kn = KNeighborsClassifier(weights='distance')
    kn.fit(features_encoded, target_encoded)

    return kn, vectorizer, label_encoder

def TestKN(kn, vectorizer, label_encoder, df=test_df, target_column='dialog_act', feature_column='utterance_content'):
    features_test = df[feature_column]
    target_test = df[target_column]

    # Count vectorizer
    features_encoded = vectorizer.transform(features_test)

    # Label encoder for the target column
    target_encoded = label_encoder.transform(target_test)

    target_predict = kn.predict(features_encoded)

    # Ensure target_names matches the number of unique classes
    target_names = ['inform', 'request', 'confirm', 'ack', 'affirm', 'hello', 'reqalts', 'null', 'negate', 'bye', 'thankyou', 'deny', 'restart', 'reqmore']
    report = classification_report(target_encoded, target_predict, target_names=target_names)

    print(report)


def main():

    # print(df_unique_alt)

    # print(average_length(df))

    print('df')
    label_count(df)
    # print('unique')
    # label_count(df_unique)
    print('unique_alt')
    label_count(df_unique_alt)
    #majority_baseline(test_df)
    #keyword_baseline()
    #perform_classification(test_df, 1)

    check_baseline_performance(test_df, 0)
    check_baseline_performance(test_df, 1)

    

    kn, vectorizer, label_encoder = CreateKNearest()
    TestKN(kn, vectorizer, label_encoder)

    tree, vectorizer_t, label_encoder_t = CreateTree(train_df)
    TestTree(tree, vectorizer_t, label_encoder_t)


    perform_classification(df,3)

    perform_classification(df, 2)

    delete_duplicates(df)


main()

#if __name__ == '__main__':  