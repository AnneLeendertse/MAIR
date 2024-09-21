import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score

# Opens the dataframe
df = pd.read_csv('./dialog_acts.dat', names=['dialog_act','utterance_content'])
df[['dialog_act','utterance_content']] = df["dialog_act"].str.split(" ", n=1, expand=True)

# Split the full dataset in a training part of 85% and a test part of 15%.
train_df, test_df = train_test_split(df, test_size=0.15, random_state=0)

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

    # label encoder voor de target column
    target_encoded = label_encoder.transform(target_test)

    target_predict = tree.predict(features_encoded)
    # accuracy = accuracy_score(target_encoded, target_predict) # Overbodig eigenlijk, ook al in de report
    report = classification_report(target_encoded, target_predict, target_names=label_encoder.classes_)

    print(report)

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

    # label encoder voor de target column
    target_encoded = label_encoder.transform(target_test)

    target_predict = kn.predict(features_encoded)
    # accuracy = accuracy_score(target_encoded, target_predict) # Overbodig eigenlijk, ook al in de report
    report = classification_report(target_encoded, target_predict, target_names=label_encoder.classes_)

    print(report)

# Quick function to reuse for asking prompts from user and converts to lower case
def ask_utterance():
    utterance = input("Insert prompt or type 'quit': ").lower()
    return utterance


def perform_classification(classifier=0, dataframe=df, train_df=train_df):
    while True:
        utterance = ask_utterance()

         # Checks if user wants to exit
        if utterance == 'quit':
            break
        
        # Performs the chosen way of classificaiton
        elif classifier == 0:
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

class dialogClass:
  def __init__(self, state, area, type, price):
      self.state = "welcome"
      self.area = None
      self.type = None
      self.price = None

def xyz(utterance, state):
    
    response 
    return response


def main():
    perform_classification()

    
    dialog = dialogClass()
    xyz(utterance(), dialog.state)


if __name__ == '__main__':
    main()
    

    