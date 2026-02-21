import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('placement_data.csv')

X = df[['CGPA', 'Internships', 'Projects', 'Backlogs', 'AptitudeScore']]
y = df['Placed']

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("congrats! Model  is  trained .")