import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
df = pd.read_csv('C:/Users/mvsla/OneDrive/Documents/GitHub/Book_llm/google_books_api_dataset/google_books_api_dataset/Google_books_api_dataset.csv')

# Ensure no missing values in required columns
df = df.dropna(subset=['Title','Authors','Language','Categories','Description'])

# Encode categories into numerical labels
label_encoder = LabelEncoder()
df['Category_Labels'] = label_encoder.fit_transform(df['Categories'])

# Combine text columns into a single column for TF-IDF
df['Combined_Text'] = df['Title'].str.cat(df['Authors'], sep=' ')\
                                  .str.cat(df['Language'], sep=' ')\
                                  .str.cat(df['Description'],sep=' ')

# Prepare TF-IDF features for combined text
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(df['Combined_Text']).toarray()  # Convert sparse matrix to dense array

# Extract labels
y = to_categorical(df['Category_Labels'], num_classes=len(df['Categories'].unique()))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape TF-IDF features for LSTM (samples, time steps, features)
X_train = X_train[:, np.newaxis, :]  # Add a time step dimension
X_test = X_test[:, np.newaxis, :]

# Extract labels
y = to_categorical(df['Category_Labels'], num_classes=len(df['Categories'].unique()))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape TF-IDF features for LSTM (samples, time steps, features)
# Here, time steps = 1, since TF-IDF treats the input as a single vector
X_train = X_train[:, np.newaxis, :]  # Add a time step dimension
X_test = X_test[:, np.newaxis, :]

# Get the number of unique categories
num_classes = y_train.shape[1]

# Build the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Predict categories for test data
y_pred = model.predict(X_test)
predicted_labels = np.argmax(y_pred, axis=1)
model.save('lstm.keras')
# Map numerical predictions back to category names
predicted_categories = label_encoder.inverse_transform(predicted_labels)

# Add predictions to the DataFrame
test_titles = df.iloc[y_test.argmax(axis=1)]['Title'].reset_index(drop=True)
output_df = pd.DataFrame({'Title': test_titles, 
                          'Actual Category': label_encoder.inverse_transform(y_test.argmax(axis=1)),
                          'Predicted Category': predicted_categories})

# Print some predictions
print(output_df.head())