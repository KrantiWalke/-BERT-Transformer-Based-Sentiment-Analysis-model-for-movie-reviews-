import collections
import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm
import transformers

# Set seed for reproducibility
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Load IMDb dataset
train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])

# Initialize BERT tokenizer
transformer_name = "bert-base-uncased"
tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_name)

# Function to tokenize and numericalize examples
def tokenize_and_numericalize_example(example, tokenizer):
    ids = tokenizer(example["text"], truncation=True)["input_ids"]
    return {"ids": ids}

# Tokenize and numericalize train and test data
train_data = train_data.map(
    tokenize_and_numericalize_example, fn_kwargs={"tokenizer": tokenizer}
)
test_data = test_data.map(
    tokenize_and_numericalize_example, fn_kwargs={"tokenizer": tokenizer}
)

# Define pad index and test size
pad_index = tokenizer.pad_token_id
test_size = 0.25

# Split train data into train and validation sets
train_valid_data = train_data.train_test_split(test_size=test_size)
train_data = train_valid_data["train"]
valid_data = train_valid_data["test"]

# Format train, validation, and test data
train_data = train_data.with_format(type="torch", columns=["ids", "label"])
valid_data = valid_data.with_format(type="torch", columns=["ids", "label"])
test_data = test_data.with_format(type="torch", columns=["ids", "label"])

# Define collate function and data loader function
def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids": batch_ids, "label": batch_label}
        return batch

    return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader

# Define batch size
batch_size = 8

# Create data loaders for train, validation, and test sets
train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)

# Define Transformer model class
class Transformer(nn.Module):
    def __init__(self, transformer, output_dim, freeze):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.config.hidden_size
        self.fc = nn.Linear(hidden_dim, output_dim)
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, ids):
        output = self.transformer(ids, output_attentions=True)
        hidden = output.last_hidden_state
        attention = output.attentions[-1]
        cls_hidden = hidden[:, 0, :]
        prediction = self.fc(torch.tanh(cls_hidden))
        return prediction

# Initialize BERT model and define model parameters
transformer = transformers.AutoModel.from_pretrained(transformer_name)
output_dim = len(train_data["label"].unique())
freeze = False

# Create Transformer model
model = Transformer(transformer, output_dim, freeze)

# Function to count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print the number of trainable parameters
print(f"The model has {count_parameters(model):,} trainable parameters")

# Define learning rate, optimizer, and criterion
lr = 1e-5
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Move model and criterion to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = criterion.to(device)

# Function to train the model
def train(data_loader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(data_loader, desc="Training..."):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)
        prediction = model(ids)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)

# Function to evaluate the model
def evaluate(data_loader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="Evaluating..."):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            prediction = model(ids)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)

# Function to calculate accuracy
def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy

# Define number of epochs and initialize best validation loss
n_epochs = 3
best_valid_loss = float("inf")

# Dictionary to store metrics
metrics = collections.defaultdict(list)

# Main training loop
for epoch in range(n_epochs):
    train_loss, train_acc = train(
        train_data_loader, model, criterion, optimizer, device
    )
    valid_loss, valid_acc = evaluate(valid_data_loader, model, criterion, device)
    metrics["train_losses"].append(train_loss)
    metrics["train_accs"].append(train_acc)
    metrics["valid_losses"].append(valid_loss)
    metrics["valid_accs"].append(valid_acc)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "transformer.pt")
    print(f"Epoch: {epoch}")
    print(f"Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}")
    print(f"Valid Loss: {valid_loss:.3f}, Valid Acc: {valid_acc:.3f}")

# Plot training and validation loss
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(metrics["train_losses"], label="Train Loss")
ax.plot(metrics["valid_losses"], label="Valid Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_xticks(range(n_epochs))
ax.legend()
ax.grid()

# Plot training and validation accuracy
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(metrics["train_accs"], label="Train Accuracy")
ax.plot(metrics["valid_accs"], label="Valid Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_xticks(range(n_epochs))
ax.legend()
ax.grid()

# Load best model
model.load_state_dict(torch.load("transformer.pt"))

# Evaluate model on test data
test_loss, test_acc = evaluate(test_data_loader, model, criterion, device)
print(f"Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}")

# Function to predict sentiment of a given text
def predict_sentiment(text, model, tokenizer, device):
    ids = tokenizer(text)["input_ids"]
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    return predicted_class, predicted_probability
'''
# Test sentiment prediction on sample texts
text1 = "This film is terrible!"
text2 = "This film is great!"
text3 = "This film is not terrible, it's great!"
text4 = "This film is not great, it's terrible!"

print(predict_sentiment(text1, model, tokenizer, device))
print(predict_sentiment(text2, model, tokenizer, device))
print(predict_sentiment(text3, model, tokenizer, device))
print(predict_sentiment(text4, model, tokenizer, device))
'''

def main():
    # Load the trained model and tokenizer
    transformer_name = "bert-base-uncased"
    model = transformers.AutoModel.from_pretrained(transformer_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Example text inputs
    texts = [
        "This film is terrible!",
        "This film is great!",
        "This film is not terrible, it's great!",
        "This film is not great, it's terrible!"
    ]

    # Perform sentiment analysis on each text
    for text in texts:
        predicted_class, predicted_probability = predict_sentiment(text, model, tokenizer, device)
        sentiment = "positive" if predicted_class == 1 else "negative"
        print(f"Text: '{text}' - Predicted Sentiment: {sentiment} with probability: {predicted_probability:.2f}")

if __name__ == "__main__":
    main()