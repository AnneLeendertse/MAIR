import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from Levenshtein import ratio
import time
import configparser
import os.path

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
        # Second machine learning technique: K nearest neighbors
        kn, vectorizer, label_encoder = CreateKNearest(dataframe)
        utterance_vec = vectorizer.transform([utterance])
        label_int = kn.predict(utterance_vec) 
        label_text = label_encoder.inverse_transform(label_int)

        return label_text

#--------------------------------------------------------------

# Function that finds possible restaurants in restaurants_info_with_attributes.csv based on food type, area and price range
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


# Small function that removes pd.Series duplictates from a list
#voegt nu twee restaurants toe als ze niet hetzelfde zijn, maar als ze wel hetzelfde zijn voegt hij er maar 1 toe. Maar hier we kunnen nog wat aan doen. 
def remove_duplicate_series(series_list):
   unique_list = []
   for series in series_list:
       if not any(series.equals(unique_series) for unique_series in unique_list):
           unique_list.append(series)
   return unique_list


# Function that applies different rules to see if a restaurant fits the additional preferences
# Additional pref can be: "TOURISTIC", "NO ASSIGNED SEATS", "CHILDREN", "ROMANTIC"
# Still needs to be implemented in the recommender state
def reasoning(possible_restaurants, addpref):
   new_possible_restaurants = []
   for restaurant_row in possible_restaurants:
       # Checks which of the possible restaurants is TOURISTIC
       if "touristic" in addpref:
           if restaurant_row['food'] == 'romanian':
               pass
           elif restaurant_row['pricerange'] == 'cheap' and restaurant_row['food_quality'] == 'good':
               new_possible_restaurants.append(restaurant_row)

       # Checks which of the possible restaurants has ASSIGNED SEATS
       if "no assigned seats" in addpref:
           if restaurant_row['crowdedness'] != 'busy':
               new_possible_restaurants.append(restaurant_row)
      
       # Checks which of the possible restaurants welcomes CHILDREN
       if "child friendly" in addpref:
           if restaurant_row['length_of_stay'] != 'long':
               new_possible_restaurants.append(restaurant_row)
      
       # Checks which of the possible restaurants is ROMANTIC
       if "romantic" in addpref:
           if restaurant_row['crowdedness'] == 'busy':
               pass
           elif restaurant_row['length_of_stay'] == 'long':
               new_possible_restaurants.append(restaurant_row)


   new_possible_restaurants = remove_duplicate_series(new_possible_restaurants)

   return new_possible_restaurants if new_possible_restaurants else possible_restaurants


# Class that:
# # # # stores dialog state, food type, area, price range
# # # # responds to user based on dialog state and utterance classification
# # # # extracts preferences from user utterance

# Things to add:
# Recognize keywords based on the keywords FOOD/AREA/PRICE when no exact keywords are found and then proceed to use leventein distance or whatever.
# e.g. I’m looking for ItaliEn food → ItaliEn is the food type based on pattern {variable} food. ItaliEn is one leventein distance away from Italian -> keyword should be Italian.

class dialogClass:
    def __init__(self, config, state=None, food=None, area=None, price=None, possible_restaurants=None, terminate=None):
        self.state = "welcome"
        self.food = None
        self.area = None
        self.price = None
        self.addpref = None
        self.askfood = None
        self.askarea = None
        self.askprice = None
        self.askaddpref = None
        self.possible_restaurants = None
        self.terminate = 0
        self.config = config
        self.set_config(config)

    # Method to set global variables according to config. We do this so i can call it again in case of 'restart'/'startover'.
    def set_config(self, config):
        self.allcaps = config['allcaps']
        self.levenshtein_cutoff_food = config['levenshtein_cutoff_food']
        self.levenshtein_cutoff_area = config['levenshtein_cutoff_area']
        self.levenshtein_cutoff_price = config['levenshtein_cutoff_price']
        self.levenshtein_cutoff_additional = config['levenshtein_cutoff_additional']
        self.delay = config['delay']
        self.allow_restart = config['allow_restart']

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
                self.state = "askfood"
                  

        # ASK FOODTYPE state
        if self.state == "askfood":
            if self.food == None and self.askfood == None: # First try
                response = 'What type of food do you want?'
                return response
            elif self.food == None and self.askfood =="Not Found": # When input is not recognized and levenshtein didnt find anything useful.
                response = 'Preference for food not recognized, please give an alternative.'
                return response
            elif self.food != None and self.askfood =="Found": # When input is not recognized but levenshtein found a possible answer.
                response = f'Did you mean {self.food}?'
                return response
            elif self.food != None and self.askfood =="Checked": # When input is found and checked go to ask area.
                self.state = "askarea"
        
        # ASK AREA state
        if self.state == 'askarea':
            if self.area == None and self.askarea == None:
                response = f"Got it! You want {self.food} food. In which area do you want to eat (reply with north/east/south/west/centre)?"
                return response
            elif self.area == None and self.askarea =="Not Found": # When input is not recognized and levenshtein didnt find anything useful.
                response = 'Preference for area not recognized, please give an alternative.'
                return response
            elif self.area != None and self.askarea =="Found": # When input is not recognized but levenshtein found a possible answer.
                response = f'Did you mean {self.area}?'
                return response
            elif self.area != None and self.askarea =="Checked": # When input is found and checked go to ask area.
                self.state = "askprice"

        # ASK PRICE RANGE state
        if self.state == 'askprice':
            if self.price == None and self.askprice == None:
                response = f'Got it! you want {self.food} food in {self.area} area. What price range do you want?' 
                return response
            elif self.price == None and self.askprice =="Not Found": # When input is not recognized and levenshtein didnt find anything useful.
                response = 'Preference for price not recognized, please give an alternative.'
                return response
            elif self.price != None and self.askprice =="Found": # When input is not recognized but levenshtein found a possible answer.
                response = f'Did you mean {self.price}?'
                return response
            elif self.price != None and self.askprice =="Checked": # When input is found and checked go to ask area.
                self.state = "addpref"

        # REASONING state --> still needs to be adapted, also the extractor
        if self.state == 'addpref':
            if self.addpref == None and self.askaddpref == None:
                response = f'Got it! you want {self.food} food in {self.area} area with in the {self.price} price range. Do you have any additional details?' 
                return response
            elif self.addpref == None and self.askaddpref =="Not Found": # When input is not recognized and levenshtein didnt find anything useful.
                response = 'Preference for additional details not recognized, you can give the following additional details: .'
                return response
            elif self.addpref != None and self.askaddpref =="Found": # When input is not recognized but levenshtein found a possible answer.
                response = f'Did you mean {self.addpref}?'
                return response
            elif self.addpref != None and self.askaddpref =="Checked": # When input is found and checked go to ask area.
                self.state = "recommend"


        # RECOMMEND state (if all values are filled, recommend restaurant based on restaurant csv file)
        if self.state == 'recommend':
            if self.possible_restaurants == None:
                self.possible_restaurants = find_restaurant(self.food, self.area, self.price)
                if self.addpref:
                    self.possible_restaurants = reasoning(self.possible_restaurants, self.addpref) # Provides reasoning filter

                if len(self.possible_restaurants) > 0:
                    restaurant_row = self.possible_restaurants.pop(0)
                    restaurant_name = restaurant_row['restaurantname']
                    response = f'A restaurant that serves {self.food} food in {self.area} part of town \n and that has {self.price} price, that is {self.addpref} is \"{restaurant_name}\". In case you want an \n alternative, type \"alternative\", otherwise type \"restart\" to start over.'
                    return response
                else: 
                    response = "I'm very sorry, but there are no restaurants that fit your preferences, would you like to start over?"
                    return response
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
                    return response
                else:
                    response = "I'm very sorry, but there are no alternative restaurants that fit your preferences, would you like to start over?"
                    return response
            else:
                    self.state = "startover"
        
        # STARTOVER state (Checks if user wants to terminate system or if they want a new recommendation)
        if self.state == "startover":
            if utterance in ["yes", "y", "yeah", "startover", "start over", "restart", "please", "sure"]:
                for attr in vars(self):
                    if attr != "terminate" and attr != "config": # Resets all variables to None except terminate and the config file
                        setattr(self, attr, None)
                    self.set_config(self.config) # Resets the variables based on the config.ini file
                self.state = "welcome"
                response = "Alright we will start over, what are your new preferences?"
            
            else:
                self.terminate = 1
                response = "Ok, bye then"
            
            return response

    # Check for the asktype
    def asktype_check(self, dialog):
        # Check in case Food is found (using levenshtein)
        if self.askfood == "Found":
            if dialog == ["negate"]:
                self.food = None
                self.askfood = None
            elif dialog == ["affirm"]:
                self.askfood = "Checked"
        # Check in case Area is found (using levenshtein)
        if self.askarea == "Found": 
            if dialog == ["negate"]:
                self.area = None
                self.askarea = None
            elif dialog == ["affirm"]:
                self.askarea = "Checked"
        # Check in case Price is found (using levenshtein)
        if self.askprice == "Found":
            if dialog == ["negate"]:
                self.price = None
                self.askprice = None
            elif dialog == ["affirm"]:
                self.askprice = "Checked"
        # Check in case Additional preference is found (using levenshtein)
        if self.askaddpref == "Found":
            if dialog == ["negate"]:
                self.addpref = None
                self.askaddpref = None
            elif dialog == ["affirm"]:
                self.askaddpref = "Checked"

       
    # Method to extract type (food/area/price) from utterance and apply Levenshtein 
    def extract_type(self, keywords, cutoff, f_utterance, attr_type):
        # Extract food/area/price type
        for f_type in keywords:
            if f_type in f_utterance:
                setattr(self, attr_type, f_type)
                setattr(self, f'ask{attr_type}', "Checked")
                break
            else:  # levenshtein
                utterance_split = f_utterance.split(' ')
                for word in utterance_split:
                    if ratio(word, f_type) > cutoff:
                        setattr(self, attr_type, f_type)
                        setattr(self, f'ask{attr_type}', "Found")
                        break

        if self.state == f'ask{attr_type}' and getattr(self, f'ask{attr_type}') is None:
            setattr(self, f'ask{attr_type}', "Not Found")

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
        
        alternative_keywords = [
            'child friendly', 'touristic', 'no assigned', 'romantic', 'assigned'
        ]
        
        dontcare_keywords = [
            'dont care', 'any', 'doesnt matter', 'whatever', 'no preference', 'anything',
            'don\'t care', 'does not matter', 'no preference', 'no matter', 'doesn\'t matter'
        ]

        # Checks whether a Levenshtein suggestion was made and accepted by the user for food, area and price. 
        self.asktype_check(dialog_act)

        # Extract food type
        self.extract_type(food_keywords, self.levenshtein_cutoff_food, utterance, 'food')
        self.extract_type(area_keywords, self.levenshtein_cutoff_area, utterance, 'area')
        self.extract_type(price_keywords, self.levenshtein_cutoff_price, utterance, 'price')
        self.extract_type(alternative_keywords, self.levenshtein_cutoff_additional, utterance, 'addpref')

        # Extract dontcare; --> seems to work now. Needs more testing (25 sept 21:33)
        for dontcare in dontcare_keywords:
            if dontcare in utterance:
                utterance_split = utterance.split(' ')  #let op komma's en punten
                dontcare_split = dontcare.split(' ')
                i = utterance_split.index(dontcare_split[0])
                for word in utterance_split[i:]:
                    if word == 'food':
                        self.food = 'any'
                        self.askfood = 'Checked'
                        break
                    elif word == 'area':
                        self.area = 'any'
                        self.askarea = 'Checked'
                        break
                    elif word == 'price':
                        self.price = 'any'
                        self.askprice = 'Checked'
                        break
                    elif word == 'addpref':
                        self.addpref = 'any'
                        self.askaddpref = 'Checked'
                        break                       
                                    
                    else: 
                        if self.state == 'askfood':
                            self.food = 'any'
                            self.askfood = 'Checked'
                            break
                        elif self.state == 'askarea':
                            self.area = 'any'
                            self.askarea = 'Checked'
                            break
                        elif self.state == 'askprice':
                            self.price = 'any'
                            self.askprice = 'Checked'
                            break
                        elif self.state == 'addpref':
                            self.addpref = 'any'
                            self.askaddpref = 'Checked'
                            break
                        break
                break
        return


# Creates a config.ini file for global variables such as allcaps
def create_config():
    config = configparser.ConfigParser()

    # Add sections and key-value pairs
    config['General'] = {'allcaps': False, 
                         'levenshtein_cutoff_food': 0.8, 
                         'levenshtein_cutoff_area': 0.65, 
                         'levenshtein_cutoff_price': 0.75, 
                         'levenshtein_cutoff_additional': 0.75,
                         'delay':0.5, 
                         'allow_restart': True
                         }

    # Write the configuration to a file
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

# Reads the config.ini file for global variables such as allcaps
def read_config():
    # Creates config file if it doesn't exist
    if os.path.isfile('config.ini') == False:
        create_config()

    # Read the configuration file
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Access values from the configuration file
    allcaps = config.getboolean('General', 'allcaps')
    levenshtein_cutoff_food = config.getfloat('General', 'levenshtein_cutoff_food')
    levenshtein_cutoff_area = config.getfloat('General', 'levenshtein_cutoff_area')
    levenshtein_cutoff_price = config.getfloat('General', 'levenshtein_cutoff_price')
    levenshtein_cutoff_additional = config.getfloat('General', 'levenshtein_cutoff_additional')
    delay = config.getfloat('General', 'delay')
    allow_restart = config.getboolean('General', 'allow_restart')

    # Return a dictionary with the retrieved values
    config_values = {
        'allcaps': allcaps,
        'levenshtein_cutoff_food': levenshtein_cutoff_food,
        'levenshtein_cutoff_area': levenshtein_cutoff_area,
        'levenshtein_cutoff_price': levenshtein_cutoff_price,
        'levenshtein_cutoff_additional': levenshtein_cutoff_additional,
        'delay': delay,
        'allow_restart': allow_restart
    }

    return config_values

def main():
    config = read_config()
    dialog = dialogClass(config)

    print('system: Hello, welcome to the greatest MAIR restaurant system! You can ask for restaurants by area, price range or food type. How may I help you?') # We should still make this caps in case allcaps == True

    while dialog.terminate == 0:
        utterance = input('user: ').replace('?', '').replace('!', '').replace('.', '').replace(',', '').lower() #to remove punctuation. I don't know if lower() is necessary (or maybe only here)
        dialog.extractor(utterance)
        response = dialog.responder(utterance)

        # checks if response needs to be upper case
        if dialog.allcaps == True:
            response = response.upper()

        time.sleep(config['delay'])
        print('system: ', response)

if __name__ == '__main__':
    main()