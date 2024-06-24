import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Read the CSV file into a DataFrame
csv_file_path = r"C:\Users\arnom\Desktop\BSc IT Honours\ITDAA4-12\heart.csv"
df = pd.read_csv(csv_file_path, delimiter=';')

# Step 2: Create a connection to the SQLite database
database_path = r"C:\Users\arnom\Desktop\BSc IT Honours\ITDAA4-12\heart_database.db"
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

# Step 3: Upload the DataFrame to the SQLite database as a table
table_name = "heart_data"
df.to_sql(table_name, conn, if_exists='replace', index=False)

# Step 4: Verify the data upload by querying the table
cursor.execute(f'SELECT * FROM {table_name} LIMIT 5')
rows = cursor.fetchall()

# Print first 5 rows to verify data upload
for row in rows:
    print(row)

# Step 5: Load data from the SQLite database into a DataFrame
df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

# Close the connection to the database
conn.close()

# Step 6: Preprocess the data

# Inspect the data to identify the target variable
print(df.head())
print(df.columns)

target_variable = 'target'

# Handle missing values by dropping rows with missing values
print(df.isnull().sum())
df.dropna(inplace=True)

# Convert categorical variables to appropriate data types
categorical_columns = ['cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Split data into features and target
X = df.drop(columns=[target_variable])
y = df[target_variable]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing for numerical data: impute missing values and scale the data
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
    ('scaler', StandardScaler())])  # Scale numerical features

# Preprocessing for categorical data: impute missing values and one-hot encode the data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with the most frequent value
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])  # One-hot encode categorical features

# Combine the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_columns)])

# Apply preprocessing to training data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Verify the preprocessed data
print("Preprocessed training data shape:", X_train.shape)
print("Preprocessed testing data shape:", X_test.shape)

# Plot the distribution of classes for the categorical variables based on the target variable
fig, axs = plt.subplots(4, 2, figsize=(16, 24))
for i, col in enumerate(categorical_columns):
    sns.countplot(x=col, hue=target_variable, data=df, ax=axs[i // 2, i % 2])
    axs[i // 2, i % 2].set_title(f'Distribution of {col} by {target_variable}')
plt.tight_layout()
plt.show()

# Plot the distribution of classes for the numeric variables based on the target variable
fig, axs = plt.subplots(len(numeric_features), 1, figsize=(16, 24))
for i, col in enumerate(numeric_features):
    sns.histplot(data=df, x=col, hue=target_variable, multiple='stack', ax=axs[i], kde=True)
    axs[i].set_title(f'Distribution of {col} by {target_variable}')
plt.tight_layout()
plt.show()

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy}")

# Determine the best model based on accuracy
best_model_name = max(models, key=lambda x: accuracy_score(y_test, models[x].predict(X_test)))
best_model = models[best_model_name]
print(f"Best Model: {best_model_name}")

# Save the best model to disk
model_path = "best_model.pkl"
joblib.dump(best_model, model_path)
print(f"Saved the best model to {model_path}")
