import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def train_and_evaluate(model, train_loader, valid_loader, loss_fn, optimizer, device, epochs=5):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        valid_loss = 0
        all_predictions = []
        all_targets = []

        # Training phase
        progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{epochs}', leave=False)
        for inputs, attentions, targets in progress_bar:
            inputs, attentions, targets = inputs.to(device), attentions.to(device), targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs, attentions)
            loss = loss_fn(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({'train_loss': f'{loss.item():.4f}'})

        # Validation phase
        model.eval()
        with torch.no_grad():
            progress_bar = tqdm(valid_loader, desc='Validating', leave=False)
            for inputs, attentions, targets in progress_bar:
                inputs, attentions, targets = inputs.to(device), attentions.to(device), targets.to(device).float()

                outputs = model(inputs, attentions)
                loss = loss_fn(outputs.squeeze(), targets)
                valid_loss += loss.item()

                # Convert logits to probabilities for binary classification
                if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                    probabilities = torch.sigmoid(outputs.squeeze())
                    preds = probabilities > 0.5
                else:
                    preds = outputs.argmax(dim=1)

                all_predictions.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Calculate average losses and accuracy
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        valid_accuracy = accuracy_score(all_targets, all_predictions)

        # Print training and validation results
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}, Val Acc: {valid_accuracy:.4f}')

    return train_loss, valid_loss, valid_accuracy


def evaluate_performance(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, attentions, targets in tqdm(data_loader, desc='Accuracy Evaluation', leave=False):
            inputs = inputs.to(device)
            attentions = attentions.to(device)
            targets = targets.to(device).long()

            outputs = model(inputs, attentions)
            predicted_probs = torch.sigmoid(outputs.squeeze())
            predicted_labels = (predicted_probs > 0.5).long()  # Get binary predictions based on the threshold of 0.5

            correct += (predicted_labels == targets).sum().item()
            total += targets.size(0)

            # Store predictions and targets to compute overall metrics later
            all_predictions.extend(predicted_labels.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Compute the metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)

    return accuracy, precision, recall, f1
