import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_table('dialog_acts.dat')

#df['dialog_act', 'utterance_content'] = df.series.str.Split(" ", expand=False)

df[['dialog_act', 'utterance_content']] = df['inform im looking for a moderately priced restaurant that serves'].str.split(" ", n=1, expand=True) # Add two additional columns;
df = df.drop('inform im looking for a moderately priced restaurant that serves', axis=1) # Remove the first column. This method loses the first row as data.

# Split the full dataset in a training part of 85% and a test part of 15%.
train_df, test_df = train_test_split(df, test_size=0.15, random_state=0)



# Function that counts amount of instances of all unique labels in the dataframe (just to get insight in the data)
def label_count(dataframe):
    # Creates dictionary of all unique labels
    labels = set(dataframe['dialog_act'].unique())
    labels_dict = dict.fromkeys(labels, 0)

    # Counts the amount of instances of labels in dataset
    for index, row in dataframe.iterrows():
        labels_dict[row['dialog_act']] += 1
    
    # Prints the amounts
    print('Instance count per label: ')
    for label, amount in labels_dict.items():
        print(label, ' : ', amount)
    
    print()


# Quick function to reuse for asking prompts from user and converts to lower case
def ask_utterance():
    utterance = input("Insert prompt or type 'quit': ").lower()
    return utterance


# Classifier 0 == majority_baseline
# Classifier 1 == keyword_baseline
def perform_classification(dataframe, classifier):
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
            # First machine learning technique
            pass
        else:
            # Second machine learning technique
            pass


# IMPLEMENT: A baseline system that, regardless of the content of the utterance, always assigns the majority class of in the data. 
# In the current dataset this is the inform label (almost 40% of all utterances).
# In both cases, think about the data you’re working with to develop your systems (i.e. think about making sure you’re not (accidentally) ‘training’ on your test data). 
# Your code should offer a prompt to enter a new utterance and classify this utterance, and repeat the prompt until the user exits.
def majority_baseline(dataframe):
    # Obtains majority label from dataframe
    majority_label = dataframe['dialog_act'].value_counts().idxmax()
    return majority_label


# A baseline system that classifies an uttertance based on keywords
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

    #One Hot encoding the utterances WERKT NIET
    # enc = preprocessing.OneHotEncoder()
    # labels_enc = enc.fit_transform(features_train).toarray()

    # Count vectorizer lijkt wel te werken
    vectorizer = CountVectorizer()
    features_encoded = vectorizer.fit_transform(features_train)

    # label encoder voor de target column
    label_encoder = preprocessing.LabelEncoder()
    target_encoded = label_encoder.fit_transform(target_train)

    clf = DecisionTreeClassifier()
    clf.fit(features_encoded, target_encoded)

    return clf, vectorizer, label_encoder

def TestTree(tree, vectorizer, label_encoder, df=test_df, target_column='dialog_act', feature_column='utterance_content'):

    features_test = df[feature_column]
    target_test = df[target_column]

    # Count vectorizer
    features_encoded = vectorizer.transform(features_test)

    # label encoder voor de target column
    target_encoded = label_encoder.transform(target_test)

    target_predict = tree.predict(features_encoded)
    accuracy = accuracy_score(target_encoded, target_predict)
    report = classification_report(target_encoded, target_predict, target_names=label_encoder.classes_)

    print(report)

def knearest():
# https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification 1.6.2. Nearest Neighbors Classification
    return 


def main():
    label_count(df)
    #majority_baseline(test_df)
    #keyword_baseline()
    #perform_classification(test_df, 1)

    check_baseline_performance(test_df, 0)
    check_baseline_performance(test_df, 1)

    tree, vectorizer, label_encoder = CreateTree()
    TestTree(tree, vectorizer, label_encoder)

main()

#if __name__ == '__main__':
  

  