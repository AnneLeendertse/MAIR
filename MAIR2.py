import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import re


# Opens the dataframe
# df = pd.read_csv('./dialog_acts.dat', names=['dialog_act','utterance_content'])
# df[['dialog_act','utterance_content']] = df["dialog_act"].str.split(" ", n=1, expand=True)

df = pd.read_csv('./dialog_acts.dat', names=['dialog_act', 'utterance_content'])
df[['dialog_act', 'utterance_content']] = df['dialog_act'].str.split(' ', n=1, expand=True)

# dialog_df = pd.read_csv('./all_dialogs.txt', sep='\n', header=None)
# dialog_df.columns = ['dialog']
# # Optionally, you can parse dialog turns and separate out user and system utterances
# dialog_df[['user', 'system']] = dialog_df['dialog'].str.split('\t', expand=True)


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

# Store trained models outside the loop
tree, vectorizer_tree, label_encoder_tree = CreateTree(train_df)
kn, vectorizer_kn, label_encoder_kn = CreateKNearest(train_df)

def perform_classification(utterance, classifier=0):
    if classifier == 0:
        # Decision tree classification
        utterance_vec = vectorizer_tree.transform([utterance])
        label_int = tree.predict(utterance_vec)
        label_text = label_encoder_tree.inverse_transform(label_int)
        return label_text[0]
    else:
        # K-nearest neighbors classification
        utterance_vec = vectorizer_kn.transform([utterance])
        label_int = kn.predict(utterance_vec)
        label_text = label_encoder_kn.inverse_transform(label_int)
        return label_text[0]
    

#--------------------------------------------------------------
# def perform_classification(utterance, classifier=0, dataframe=df, train_df=train_df):
#     while True:

#          # Checks if user wants to exit
#         if utterance == 'quit':
#             break
        
#         # Performs the chosen way of classificaiton
#         elif classifier == 0:
#             # First machine learning technique: Decision trees
#             tree, vectorizer, label_encoder = CreateTree(train_df)
#             utterance_vec = vectorizer.transform([utterance]) # DOES NOT WORK YET
#             label_int = tree.predict(utterance_vec)
#             label_text = label_encoder.inverse_transform(label_int)

#             return label_text
#         else:
#             # Second machine learning technique: K nearest neighbors (K=5)
#             kn, vectorizer, label_encoder = CreateKNearest(dataframe)
#             utterance_vec = vectorizer.transform([utterance])
#             label_int = kn.predict(utterance_vec) # DOES NOT WORK YET
#             label_text = label_encoder.inverse_transform(label_int)

#             return label_text

#--------------------------------------------------------------

# Function that finds possible restaurants in restaurants_info.csv based on food type, area and price range 
# and returns list of the entire row of restaurant info from the dataframe


#--------------------------------------------------------------
# def find_restaurant(foodtype, area, price):
#     df = pd.read_csv('./restaurants_info.csv', names=["restaurantname","pricerange","area","food","phone","addr","postcode"])
    
#     possible_restaurants = []

#     for index, row in df.iterrows():

#         if row['food'] == foodtype or foodtype == None:
#             if row['area'] == area or area == None:
#                 if row['pricerange'] == price or price == None:
#                     possible_restaurants.append(row)

#     return possible_restaurants
#--------------------------------------------------------------



def find_restaurant(foodtype, area, price):
    df = pd.read_csv('./restaurants_info.csv', names=["restaurantname", "pricerange", "area", "food", "phone", "addr", "postcode"])
    
    # Case insensitive matching and handling None values
    possible_restaurants = df[
        (df['food'].str.contains(foodtype, case=False, na=False) | pd.isna(foodtype)) &
        (df['area'].str.contains(area, case=False, na=False) | pd.isna(area)) &
        (df['pricerange'].str.contains(price, case=False, na=False) | pd.isna(price))
    ]
    return possible_restaurants



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

    #-------------------------------------------------------------------------
    # def responder(self, utterance):
    #     dialog_act = perform_classification(utterance)

    #     print(dialog_act)

    #     # WELCOME state
    #     if self.state == 'welcome':
    #         if dialog_act == 'hello':
    #             response = "Hi, please respond with your preferences"
    #             return response
        
    #         else:
    #             self.state = "askfoodtype"
                  
    #     # ASK FOODTYPE state
    #     if self.state == 'askfoodtype':
    #         if self.type == None:
    #             response = 'Hello, what type of food do you want?'
    #             return response

    #         else:
    #             self.state = "askarea"
        
    #     # ASK AREA state
    #     if self.state == 'askarea':
    #         if self.area == None:
    #             response = 'Hello, in which area do you want to eat (reply with north/east/south/west)?'
    #             return response
                
    #         else:
    #             self.state = "askpricerange"

    #     # ASK PRICE RANGE state
    #     if self.state == 'askpricerange':
    #         if self.price == None:
    #             response = 'Hello, what price range do you want?'
    #             return response

    #         else:
    #             self.state = "recommend"

    #     # RECOMMEND state (if all values are filled, recommend restaurant based on restaurant csv file)
    #     if self.state == 'recommend':
    #         print('yo')
    #         pass
    #-------------------------------------------------------------------------
    
    # def responder(self, utterance):
    #     dialog_act = perform_classification(utterance)

    #     if self.state == 'welcome':
    #         if dialog_act == 'hello':
    #             self.state = "askfoodtype"
    #             return "Hi, please tell me what type of food you're looking for."

    #     if self.state == 'askfoodtype':
    #         if self.type is None:
    #             self.extractor(utterance)
    #             if self.type:
    #                 self.state = "askarea"
    #                 return f"Great! You want {self.type}. In which area would you like to eat (north/east/south/west)?"

    #     if self.state == 'askarea':
    #         if self.area is None:
    #             self.extractor(utterance)
    #             if self.area:
    #                 self.state = "askpricerange"
    #                 return f"Got it! In the {self.area} area. What's your price range (cheap/moderate/expensive)?"

    #     if self.state == 'askpricerange':
    #         if self.price is None:
    #             self.extractor(utterance)
    #             if self.price:
    #                 self.state = "recommend"
    #                 return "Thanks for providing all the information. Let me find the best restaurants for you."

    #     if self.state == 'recommend':
    #         restaurants = find_restaurant(self.type, self.area, self.price)
    #         if restaurants:
    #             return f"I found these restaurants for you: {', '.join([r['restaurantname'] for r in restaurants])}"
    #         else:
    #            return "Sorry, no restaurants found with those preferences."
    #-------------------------------------------------------------------------
    def responder(self, utterance):
        # Extract preferences from user utterance
        food, area, price = self.extractor(utterance)

        if self.state == "welcome":
            self.state = "askfoodtype"
            return "Welcome! What type of food would you like?"


        elif self.state == "askfoodtype":
            if self.food:
                self.state = "askarea"
                return f"Got it! You want {food} food. Which area would you like to dine in (north, south, east, west)?"
            else:
                return "Please tell me what type of food you want."
        
        elif food:
            self.state = 'askarea'
            return f"Got it! You want {food}. In which area would you like to eat (north, south, east, west)?"

        elif self.state == 'askarea' and area is None:
            return "Which area would you like to dine in?"
        elif area:
            self.state = 'askpricerange'
            return f"Great! In the {area} area. What price range are you looking for (cheap, moderate, expensive)?"

        elif self.state == 'askpricerange' and price is None:
            return "What's your price range?"
        elif price:
            self.state = 'recommend'
            return "Thanks for the details. Let me find the best restaurant for you!"




    #--------------------------------------------------------------------------
    # # IMPLEMENT: Method that extracts food type, area, price range from user utterance
    # def extractor(self, utterance):
    #     if True:
    #         #self.type = "Italian"
    #         self.area = "West"
    # --------------------------------------------------------------------------



    def extractor(self, utterance):
        # Lowercase the utterance for case-insensitive matching
        utterance = utterance.lower()

        # Define keywords for food type
        food_keywords = {
            'italian': 'Italian', 'chinese': 'Chinese', 'indian': 'Indian', 'british': 'British',
            'thai': 'Thai', 'french': 'French', 'bistro': 'Bistro', 'mediterranean': 'Mediterranean',
            'seafood': 'Seafood', 'japanese': 'Japanese', 'turkish': 'Turkish', 'romanian': 'Romanian', 
            'steakhouse': 'Steakhouse', 'asian oriental': 'Asian Oriental', 'spanish': 'Spanish', 
            'north american': 'North American', 'fast food': 'Fast Food', 'modern european': 'Modern European',
            'european': 'European', 'portuguese': 'Portuguese', 'dont care': 'Dont Care', 'any': 'Any',
            'jamaican': 'Jamaican', 'lebanese': 'Lebanese', 'gastropub': 'Gastropub', 'cuban': 'Cuban',
            'catalan': 'Catalan', 'maroccan': 'Maroccan', 'persian': 'Persian', 'african': 'African',
            'polynesian': 'Polynesian', 'traditional': 'Traditional', 'international': 'International',
            'tuscan': 'Tuscan', 'australasian': 'Australasian', 'fusion': 'Fusion', 'korean': 'Korean',
            'vietnamese': 'Vietnamese'
        }
        # Define keywords for area
        area_keywords = {
            'north': 'North', 'south': 'South', 'east': 'East', 'west': 'West', 'center': 'Center',
            'downtown': 'Center', 'dont care': 'Dont care', 'any': 'Any'
        }

        # Define keywords for price range
        price_keywords = {
            'cheap': 'Cheap', 'moderate': 'Moderate', 'expensive': 'Expensive', 'dont care': 'Dont care',
            'any': 'Any'
        }

        # Extract food type
        for keyword, food in food_keywords.items():
            if keyword in utterance:
                self.type = food
                break

    # Extract area
        for keyword, area in area_keywords.items():
            if keyword in utterance:
                self.area = area
                break

    # Extract price range
        for keyword, price in price_keywords.items():
            if keyword in utterance:
                self.price = price
                break

    # If no preference found, set to None
        if not self.type:
            self.type = None
        if not self.area:
            self.area = None
        if not self.price:
            self.price = None

        return self.type, self.area, self.price

    def advanced_extractor(self, utterance):
        utterance = utterance.lower()

        # Regular expressions for matching patterns (more needed)
        food_regex = r"(italian|chinese|indian|british|thai|french|bistro|mediterranean|seafood|korean|vietnamese|japanese|turkish|romanian|steakhouse|asian oriental|spanish|north american|fast food|modern european|european|portuguese|dont care|any|jamaican|lebanese|gastropub|cuban|catalan|maroccan|thai|turkish|persian|african|polynesian|traditional|international|tuscan|australasian|fusion)"
        area_regex = r"(north|south|east|west|center|downtown|dont care|any)"
        price_regex = r"(cheap|moderate|expensive|dont care|any)"
    
        # Search for food type
        food_match = re.search(food_regex, utterance)
        if food_match:
            self.food = food_match.group(0).capitalize()

        # Search for area
        area_match = re.search(area_regex, utterance)
        if area_match:
            self.area = area_match.group(0).capitalize()

        # Search for price range
        price_match = re.search(price_regex, utterance)
        if price_match:
            self.price = price_match.group(0).capitalize()

        return self.food, self.area, self.price
        

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
    

    