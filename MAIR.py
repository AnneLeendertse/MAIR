import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_table('dialog_acts.dat')

#df['dialog_act', 'utterance_content'] = df.series.str.Split(" ", expand=False)

df[['dialog_act', 'utterance_content']] = df['inform im looking for a moderately priced restaurant that serves'].str.split(" ", n=1, expand=True) # Add two additional columns;
df = df.drop('inform im looking for a moderately priced restaurant that serves', axis=1) # Remove the first column. This method loses the first row as data.

# Split the full dataset in a training part of 85% and a test part of 15%.
train_df, test_df = train_test_split(df, test_size=0.15, random_state=0)
print(train_df)

# IMPLEMENT: A baseline system that, regardless of the content of the utterance, always assigns the majority class of in the data. 
# In the current dataset this is the inform label (almost 40% of all utterances).
# In both cases, think about the data you’re working with to develop your systems (i.e. think about making sure you’re not (accidentally) ‘training’ on your test data). 
# Your code should offer a prompt to enter a new utterance and classify this utterance, and repeat the prompt until the user exits.

def new_utterance (Utterance_User = input("Utterance: ")):
    pass