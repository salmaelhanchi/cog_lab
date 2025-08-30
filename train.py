import torch
import torch.nn as nn
import torch.optim as optim
from model import MemoryRNN
from data_loader import generate_task_data

def train_model(model, num_epochs, batch_size, sequence_length, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("--- Starting Training ---")
    for epoch in range(num_epochs):
        inputs, labels = generate_task_data(batch_size, sequence_length)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 25 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    print("--- Training Complete ---")
    return model

# This block allows us to run this file directly for testing purposes
if __name__ == '__main__':
    # Define hyperparameters for a test run
    INPUT_SIZE = 3
    HIDDEN_SIZE = 16
    OUTPUT_SIZE = 3
    SEQUENCE_LENGTH = 10
    NUM_EPOCHS = 250
    BATCH_SIZE = 128
    LEARNING_RATE = 0.005

    # Create the model
    rnn_model = MemoryRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    
    # Train it
    trained_rnn_model = train_model(
        model=rnn_model,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        learning_rate=LEARNING_RATE
    )