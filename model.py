# Importing Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle


# loading Dataset
df = pd.read_csv('Iris.csv')
df = df.drop(['Id'],axis=1)
print(df.head())

# Seperating predictors and response
X = df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y = df["Species"]
print(X.shape,y.shape)

# Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

# instantiating the model
classifier = RandomForestClassifier()

# Fitting the model
classifier.fit(X_train, y_train)

# making pickle for our model
pickle.dump(classifier, open("model.pkl","wb"))
