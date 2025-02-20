import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.cuda.empty_cache()

# Load dataset
df = pd.read_csv('C:/Users/mvsla/OneDrive/Documents/GitHub/Book_llm/google_books_dataset.csv')

# Ensure no missing values in required columns
df = df.dropna(subset=['Title','Authors','Language','Categories','Description'])

# Encode categories into numerical labels
label_encoder = LabelEncoder()
df['Category_Labels'] = label_encoder.fit_transform(df['Categories'])

# Combine text columns into a single column for TF-IDF
df['Combined_Text'] = df['Title'].str.cat(df['Authors'], sep=' ').str.cat(df['Language'], sep=' ').str.cat(df['Description'], sep=' ')

# Prepare TF-IDF features for combined text
tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
X = tfidf.fit_transform(df['Combined_Text']).toarray()

# Dimensionality reduction using Truncated SVD
svd = TruncatedSVD(n_components=500, random_state=42)
X = svd.fit_transform(X)

# Extract labels
y = label_encoder.transform(df['Categories'])
num_classes = len(np.unique(y))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)  # Add time step dimension
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# Create DataLoader for batching
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the LSTM model
class LSTMModel(nn.Module):
    def _init_(self, input_size, hidden_size, output_size):
        super(LSTMModel, self)._init_()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # Use the last hidden state
        x = self.dropout(torch.relu(self.fc1(h_n[-1])))
        x = self.fc2(x)
        return x

# Initialize the model
input_size = X_train.shape[2]
hidden_size = 256
model = LSTMModel(input_size, hidden_size, num_classes).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == y_batch).sum().item()
        total_samples += y_batch.size(0)

    accuracy = correct_predictions / total_samples * 100
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")


# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

# Save the model
torch.save(model.state_dict(), 'lstm_model.pth')

# Make predictions on test data
model.eval()
y_pred = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())

# Map predictions back to category names
predicted_categories = label_encoder.inverse_transform(y_pred)

# Create a DataFrame to display results
test_titles = df.iloc[y_test.cpu().numpy()]['Title'].reset_index(drop=True)
output_df = pd.DataFrame({
    'Title': test_titles,
    'Actual Category': label_encoder.inverse_transform(y_test.cpu().numpy()),
    'Predicted Category': predicted_categories
})

print(output_df.head())