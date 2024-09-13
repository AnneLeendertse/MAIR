import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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



# IMPLEMENT: A baseline system that, regardless of the content of the utterance, always assigns the majority class of in the data. 
# In the current dataset this is the inform label (almost 40% of all utterances).
# In both cases, think about the data you’re working with to develop your systems (i.e. think about making sure you’re not (accidentally) ‘training’ on your test data). 
# Your code should offer a prompt to enter a new utterance and classify this utterance, and repeat the prompt until the user exits.
def majority_baseline(dataframe):
    # Obtains majority label from dataframe
    majority_label = dataframe['dialog_act'].value_counts().idxmax()
    while True:
        utterance = ask_utterance()

        # Checks if user wants to exit
        if utterance == 'quit':
            break

        print('Utterance is classified as: ', majority_label, '\n')


# A baseline system that classifies an uttertance based on keywords
def keyword_baseline():
    label_keywords = {
        'inform' : ['any', 'looking'],
        'request' : ['what', 'whats'],
        'thankyou' : ['thank', 'thanks'],
        'reqalts' : [],
        'null' : [],
        'affirm' : [],
        'negate' : [],
        'bye' : ['bye', 'goodbye', 'adieu'],
        'confirm' : [],
        'repeat' : [],
        'ack' : [],
        'hello' : [],
        'deny' : [],
        'restart' : [],
        'reqmore' : []
    }

    while True:
        utterance = ask_utterance()

        # Checks if user wants to exit
        if utterance == 'quit':
            break

        # Splits prompt into words
        utterance_split = utterance.split(' ')

        # Loops through keywords and words and classifies the utterance accordingly
        keyword_found_check = False
        for label, keywords in label_keywords.items():
            for word in utterance_split:
                if word in keywords:
                    print('Utterance is classified as: ', label, '\n')
                    keyword_found_check = True
                    break
        if not keyword_found_check:        
            print('Utterance is classified as: inform \n')
            keyword_found_check = False



def main():
    label_count(df)
    #majority_baseline(test_df)
    keyword_baseline()

if __name__ == '__main__':
  main()