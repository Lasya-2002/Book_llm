import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import sqlite3
import pandas as pd

def get_data_database():
    try:
        conn=sqlite3.connect('book_recommendations.db')
        cursor=conn.cursor()
        query="""
        SELECT title_embedding,category_embedding,book_embedding,author_embedding
        FROM books
        """
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        
        for row in results:
            title_embedding,category_embedding,book_embedding,author_embedding = row
            title_embedding=np.frombuffer(title_embedding,dtype=np.float32)
            category_embedding=np.frombuffer(category_embedding,dtype=np.float32)
            book_embedding=np.frombuffer(book_embedding,dtype=np.float32)
            author_embedding=np.frombuffer(author_embedding,dtype=np.float32)
            title_category_embedding=np.concatenate([title_embedding,category_embedding])
            book_author_embedding=np.concatenate([book_embedding,author_embedding])
            
    except Exception as e:
        print(f'Error occurred :{e}')
        return []

data=get_data_database()


# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(title_category_embeddings, author_description_embeddings, test_size=0.2, random_state=42)

# Define the DNN model
model = Sequential([
    Input(shape=(1536,)),  # Input layer, 1536 for title+category embeddings
    Dense(1024, activation='relu'),  # First hidden layer with 1024 neurons
    Dense(512, activation='relu'),   # Second hidden layer with 512 neurons
    Dense(1536)  # Output layer with 1536 for the author+description embeddings
])

# Compile the model
model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Predict missing embeddings for new inputs (e.g., for a new book)
new_input_embedding = np.random.rand(1, 1536)  # Example new input
predicted_embedding = model.predict(new_input_embedding)
print(predicted_embedding)
