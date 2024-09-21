import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

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


# Quick function to reuse for asking prompts from user and converts to lower case
def ask_utterance():
    utterance = input("Insert prompt or type 'quit': ").lower()
    return utterance


def perform_classification(utterance, classifier=0, dataframe=df, train_df=train_df):
    while True:

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

            return label_text
        else:
            # Second machine learning technique: K nearest neighbors (K=5)
            kn, vectorizer, label_encoder = CreateKNearest(dataframe)
            utterance_vec = vectorizer.transform([utterance])
            label_int = kn.predict(utterance_vec) # DOES NOT WORK YET
            label_text = label_encoder.inverse_transform(label_int)

            return label_text


# Class that:
# # # # stores dialog state, food type, area, price range
# # # # responds to user based on dialog state and utterance classification
# # # # extracts preferences from user utterance
class dialogClass:
    def __init__(self, state=None, type=None, area=None, price=None):
        self.state = "welcome"
        self.type = None
        self.area = None
        self.price = None

    # Method to respond to the user depending on the dialog state and utterance classification
    def responder(self, utterance):
        dialog_act = perform_classification(utterance)

        print(dialog_act)

        # WELCOME state
        if self.state == 'welcome':
            if dialog_act == 'hello':
                response = "Hi, please respond with your preferences"
                return response
        
            else:
                self.state = "askfoodtype"
                  
        # ASK FOODTYPE state
        if self.state == 'askfoodtype':
            if self.type == None:
                response = 'Hello, what type of food do you want?'
                return response

            else:
                self.state = "askarea"
        
        # ASK AREA state
        if self.state == 'askarea':
            if self.area == None:
                response = 'Hello, in which area do you want to eat (reply with north/east/south/west)?'
                return response
                
            else:
                self.state = "askpricerange"

        # ASK PRICE RANGE state
        if self.state == 'askpricerange':
            if self.price == None:
                response = 'Hello, what price range do you want?'
                return response

            else:
                self.state = "recommend"

        # RECOMMEND state (if all values are filled, recommend restaurant based on restaurant csv file)
        if self.state == 'recommend':
            print('yo')
            pass

    
    # IMPLEMENT: Method that extracts food type, area, price range from user utterance
    def extractor(self, utterance):
        if True:
            #self.type = "Italian"
            self.area = "West"
        
        

def main():

    dialog = dialogClass()
    print('system: Hello , welcome to the MAIR restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?')

    while True:
        utterance = input('user: ')
        dialog.extractor(utterance)
        response = dialog.responder(utterance)
        print('system: ', response)

    


if __name__ == '__main__':
    main()
    

    