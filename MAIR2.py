import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from Levenshtein import ratio

# Opens the dataframe
df = pd.read_csv('./dialog_acts.dat', names=['dialog_act', 'utterance_content'])
df[['dialog_act', 'utterance_content']] = df['dialog_act'].str.split(' ', n=1, expand=True)

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


# both functions below work (I think) 

# def perform_classification(utterance, classifier=0):
#     if classifier == 0:
#         # Decision tree classification
#         utterance_vec = vectorizer_tree.transform([utterance])
#         label_int = tree.predict(utterance_vec)
#         label_text = label_encoder_tree.inverse_transform(label_int)
#         return label_text[0]
#     else:
#         # K-nearest neighbors classification
#         utterance_vec = vectorizer_kn.transform([utterance])
#         label_int = kn.predict(utterance_vec)
#         label_text = label_encoder_kn.inverse_transform(label_int)
#         return label_text[0]
#--------------------------------------------------------------
def perform_classification(utterance, classifier=0, dataframe=df, train_df=train_df):
    # Performs the chosen way of classification
    if classifier == 0:
        # First machine learning technique: Decision trees
        tree, vectorizer, label_encoder = CreateTree(train_df)
        utterance_vec = vectorizer.transform([utterance]) 
        label_int = tree.predict(utterance_vec)
        label_text = label_encoder.inverse_transform(label_int)

        return label_text
    else:
        # Second machine learning technique: K nearest neighbors (K=5)
        kn, vectorizer, label_encoder = CreateKNearest(dataframe)
        utterance_vec = vectorizer.transform([utterance])
        label_int = kn.predict(utterance_vec) 
        label_text = label_encoder.inverse_transform(label_int)

        return label_text

#--------------------------------------------------------------

# Function that finds possible restaurants in restaurants_info.csv based on food type, area and price range 
# and returns list of the entire row of restaurant info from the dataframe


#--------------------------------------------------------------
def find_restaurant(food, area, price):
    df = pd.read_csv('./restaurants_info_with_attributes.csv', names=["restaurantname","pricerange","area","food","phone","addr","postcode","food_quality","crowdedness","length_of_stay"])
    
    possible_restaurants = []

    for index, row in df.iterrows():

        if row['food'] == food or food == "any":
            if row['area'] == area or area == "any":
                if row['pricerange'] == price or price == "any":
                    possible_restaurants.append(row)

    return possible_restaurants
#--------------------------------------------------------------



# def find_restaurant(food, area, price):
#     df = pd.read_csv('./restaurants_info.csv', names=["restaurantname", "pricerange", "area", "food", "phone", "addr", "postcode", "food_quality", "crowdedness", "length_of_stay"])
    
#     # Case insensitive matching and handling of missing values
#     possible_restaurants = df[
#         (df['food'].str.contains(food, case=False, na=False) | pd.isna(food)) &
#         (df['area'].str.contains(area, case=False, na=False) | pd.isna(area)) &
#         (df['pricerange'].str.contains(price, case=False, na=False) | pd.isna(price))
#     ]
#     return possible_restaurants



# Class that:
# # # # stores dialog state, food type, area, price range
# # # # responds to user based on dialog state and utterance classification
# # # # extracts preferences from user utterance

# Things to add:
# Recognize keywords based on the keywords FOOD/AREA/PRICE when no exact keywords are found and then proceed to use leventein distance or whatever.
# e.g. I’m looking for ItaliEn food → ItaliEn is the food type based on pattern {variable} food. ItaliEn is one leventein distance away from Italian -> keyword should be Italian.

class dialogClass:
    def __init__(self, state=None, food=None, area=None, price=None, possible_restaurants=None, terminate=None):
        self.state = "welcome"
        self.food = None
        self.area = None
        self.price = None
        self.askfood = None
        self.askarea = None
        self.askprice = None
        self.cutoff = 0.8
        self.possible_restaurants = None
        self.terminate = 0

    # Method to respond to the user depending on the dialog state and utterance classification

    #-------------------------------------------------------------------------
    def responder(self, utterance):
        dialog_act = perform_classification(utterance)

        print(dialog_act)

        # WELCOME state
        if self.state == 'welcome':
            if dialog_act == 'hello':
                response = "Hi, please respond with your preferences."
                return response
        
            else:
                self.state = "askfoodtype"
                  
        # ASK FOODTYPE state
        # WE STILL NEED COPY THIS ARCHITECTURE TO THE REST OF THE STATES !!! 
# WE STILL NEED COPY THIS ARCHITECTURE TO THE REST OF THE STATES !!! 
# WE STILL NEED COPY THIS ARCHITECTURE TO THE REST OF THE STATES !!! 
# WE STILL NEED COPY THIS ARCHITECTURE TO THE REST OF THE STATES !!! 
# WE STILL NEED COPY THIS ARCHITECTURE TO THE REST OF THE STATES !!! 
            if self.food == None and self.askfood == None: # First try
                response = 'What type of food do you want?'
                return response
            elif self.food == None and self.askfood =="Not Found": # When input is not recognized and levenshtein didnt find anything usefull.
                response = 'Preference not recognized, please give an alternative.'
                return response
            elif self.food != None and self.askfood =="Found": # When input is not recognized but levenshtein found a possibile answer.
                response = f'Did you mean {self.food}?'
                return response
            elif self.food != None and self.askfood =="Checked": # When input is found and checked go to ask area.
                self.state = "askarea"
        
        # ASK AREA state
        if self.state == 'askarea':
            if self.area == None:
                response = f"Got it! You want {self.food} food. In which area do you want to eat (reply with north/east/south/west/centre)?"
                return response
                
            else:
                self.state = "askpricerange"

        # ASK PRICE RANGE state
        if self.state == 'askpricerange':
            if self.price == None:
                response = f'Got it! you want {self.food} in {self.area} area. What price range do you want?'
                return response

            else:
                self.state = "recommend"

        # RECOMMEND state (if all values are filled, recommend restaurant based on restaurant csv file)
        if self.state == 'recommend':
            
            if self.possible_restaurants == None:
                self.possible_restaurants = find_restaurant(self.food, self.area, self.price)
                if len(self.possible_restaurants) > 0:
                    restaurant_row = self.possible_restaurants.pop(0)
                    restaurant_name = restaurant_row['restaurantname']
                    response = f'A restaurant that serves {self.food} food in {self.area} part of town \n and that has {self.price} price is \"{restaurant_name}\". In case you want an \n alternative, type \"alternative\", otherwise type \"restart\" to start over.'
                else: 
                    response = "I'm very sorry, but there are no restaurants that fit your preferences, would you like to start over?"
            
            else:
                if dialog_act == "reqalts" or utterance == "alternative":
                    self.state = "recommendalt"
                else:
                    self.state = "startover"
        
        # RECOMMEND ALTERNATIVE state (Keeps looping if user wants another recommendation, otherwises transitions to STARTOVER state)
        if self.state == "recommendalt":
            if dialog_act == "reqalts" or utterance == "alternative":
                if len(self.possible_restaurants) > 0:
                    restaurant_row = self.possible_restaurants.pop(0)
                    restaurant_name = restaurant_row['restaurantname']
                    response = f"An alternative restaurant could be {restaurant_name}. If you would you like an alternative, type \"alternative\"?"
                else:
                    response = "I'm very sorry, but there are no alternative restaurants that fit your preferences, would you like to start over?"
            else:
                    self.state = "startover"
        
        # STARTOVER state (Checks if user wants to terminate system or if they want a new recommendation)
        if self.state == "startover":
            if utterance in ["yes", "y", "yeah", "startover", "start over", "restart", "please", "sure"]:
                self.state = "welcome"
                self.food = None
                self.area = None
                self.price = None
                self.possible_restaurants = None
                response = "Alright we will start over, what are your new preferences?"
            else:
                self.terminate = 1
                response = "Ok, bye then"
        
        return response




    def extractor(self, utterance):
        # Lowercase the utterance for case-insensitive matching
        utterance = utterance.lower()
        dialog_act = perform_classification(utterance) # Kan efficienter

        # Define keywords for food type
        food_keywords = [
            'italian', 'chinese', 'indian', 'british', 'thai', 'french', 'bistro', 'mediterranean',
            'seafood', 'japanese', 'turkish', 'romanian', 'steakhouse', 'asian oriental', 'spanish',
            'north american', 'fast food', 'modern european', 'european', 'portuguese'
            'jamaican', 'lebanese', 'gastropub', 'cuban', 'catalan', 'maroccan', 'persian',
            'african', 'polynesian', 'traditional', 'international', 'tuscan', 'australasian', 'fusion',
            'korean', 'vietnamese'
            ]


        # Define keywords for area
        area_keywords = [
            'north', 'south', 'east', 'west', 'centre'
            ]


        # Define keywords for price range
        price_keywords = [
            'cheap', 'moderate', 'expensive'
            ]
        
        dontcare_keywords = [
            'dont care', 'any', 'doesnt matter', 'whatever', 'no preference', 'anything',
            'don\'t care', 'does not matter', 'no preference', 'no matter', 'doesn\'t matter'
        ]

        if self.askfood == "Found":
            if dialog_act == ["negate"]:
                self.food = None
                self.askfood = None
                return
            elif dialog_act == ["affirm"]:
                self.askfood = "Checked"
                return
        
        # Extract food type
        for food in food_keywords:
            if food in utterance:
                self.food = food
                self.askfood = "Checked"
                break
            else: #levenshtein
                utterance_split = utterance.split(' ')
                for word in utterance_split:
                    if ratio(word, food) > self.cutoff: #maybe we should use a different cutoff for food/area/price
                        self.food = food
                        self.askfood = "Found"
                    break
        
        if self.askfood == None:
            print('test')
            self.askfood = "Not Found"       

        # Extract area
        for area in area_keywords:
            if area in utterance:
                self.area = area
                break

         # Extract price range
        for price in price_keywords:
            if price in utterance:
                self.price = price
                break

        # Extract dontcare; this doesn't work yet --> seems to work now. Needs more testing (25 sept 21:33)
        for dontcare in dontcare_keywords:
            if dontcare in utterance:
                utterance_split = utterance.split(' ')  #let op komma's en punten
                dontcare_split = dontcare.split(' ')
                i = utterance_split.index(dontcare_split[0])
                for word in utterance_split[i:]:
                    if word == 'food':
                        self.food = 'any'
                        break
                    elif word == 'area':
                        self.area = 'any'
                        break
                    elif word == 'price':
                        self.price = 'any'
                        break                       
                                    
                    else: 
                        if self.state == 'askfoodtype':
                            self.food = 'any'
                            break
                        elif self.state == 'askarea':
                            self.area = 'any'
                            break
                        elif self.state == 'askpricerange':
                            self.price = 'any'
                            break
                        break
            
                break

        return

def main():
    dialog = dialogClass()
    print('system: Hello, welcome to the greatest MAIR restaurant system? You can ask for restaurants by area, price range or food type. How may I help you?')

    while dialog.terminate == 0:
        utterance = input('user: ').replace('?', '').replace('!', '').replace('.', '').replace(',', '').lower() #to remove punctuation. I don't know if lower() is necessary (or maybe only here)
        dialog.extractor(utterance)
        response = dialog.responder(utterance)
        print('system: ', response)

if __name__ == '__main__':
    main()
    

    