import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
data = pd.read_csv("./diabetes.csv")
print(data.head(10))

print(data.shape)

# Check for missing values
print(data.isnull().values.any())

# Rename columns for clarity
data.rename(columns={'DiabetesPedigreeFunction': 'DPF', 'BloodPressure': 'BP'}, inplace=True)
print(data.head(5))

print(data.describe())

# Correlation matrix
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10, 10))
g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()

# Print the number of zeros in each column
columns_with_zeros = ["Glucose", "BP", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]
for column in columns_with_zeros:
    print(f"Number of zeros in {column}: {data[data[column] == 0].shape[0]}")

# Replace zeros with the mean of the respective columns
for column in columns_with_zeros:
    data[column] = data[column].replace(0, data[column].mean())

# Verify that there are no zeros left in these columns
for column in columns_with_zeros:
    print(f"Number of zeros in {column}: {data[data[column] == 0].shape[0]}")

# Visualize the number of diabetic and non-diabetic people
positive_outcome = len(data.loc[data["Outcome"] == 1])
negative_outcome = len(data.loc[data["Outcome"] == 0])
print((positive_outcome, negative_outcome))

y = np.array([positive_outcome, negative_outcome])
mylabels = ["Diabetic people (268)", "Non-diabetic people (500)"]
plt.pie(y, labels=mylabels, colors=["orange", "yellow"])
plt.title("Number of diabetic and Non-diabetic persons")
plt.show()

df = {'Diabetic': positive_outcome, 'Non-diabetic': negative_outcome}
A = list(df.keys())
B = list(df.values())
plt.bar(A, B, width=0.2)
plt.title("Number of diabetic and Non-diabetic persons")
plt.show()

# Split the dataset into features and target variable
X = data.drop(columns=["Outcome"])
Y = data["Outcome"]

# Correct the train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=10)

# Train the RandomForest model
model = RandomForestClassifier(random_state=10)
model.fit(X_train, Y_train)

# Make predictions
pred = model.predict(X_test)

# Calculate accuracy
acc = metrics.accuracy_score(Y_test, pred)
print("\n\nACCURACY OF THE MODEL:", acc)

# Function to predict diabetes
def prediction_calculation(n):
    for i in range(n):
        print(f"\nENTER THE DETAILS FOR PERSON: {i + 1}")
        Gender_ip = input("\nGENDER M/F or m/f: ")
        Preg_ip = 0 if Gender_ip in ["M", "m"] else input("Number of Pregnancies: ")
        Age_ip = input("Age: ")
        Bmi_ip = input("BMI: ")
        Glucose_ip = input("Glucose level: ")
        Insulin_ip = input("Insulin level: ")
        Bp_ip = input("Blood Pressure level: ")
        St_ip = input("Skin Thickness: ")
        Dpf_ip = input("Diabetes pedigree function: ")
        
        c = np.array([Preg_ip, Glucose_ip, Bp_ip, St_ip, Insulin_ip, Bmi_ip, Dpf_ip, Age_ip], dtype=float)
        c_rs = c.reshape(1, -1)
        pred = model.predict(c_rs)
        
        if pred == 1:
            print("DIABETIC PERSON !!")
        else:
            print("NON-DIABETIC PERSON :)")

# Input the number of people for prediction
no_of_people = int(input("\nENTER NUMBER OF PEOPLE: "))
prediction_calculation(no_of_people)
