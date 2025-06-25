import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

import shutil
shutil.rmtree("./visualisierung", ignore_errors=True)
writer = SummaryWriter("./visualisierung")


# 1. Daten vorbereiten
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. CNN-Modell definieren
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # [B, 32, 28, 28]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # [B, 64, 28, 28]
        self.pool = nn.MaxPool2d(2, 2)  # [B, 64, 14, 14]
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.LeakyReLU(0.01)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # automatisch richtige Batch-Größe
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. Modell, Optimierer und Loss
def start_training():
    import os
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    from sklearn.metrics import confusion_matrix

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Hyperparameter
    num_epochs = 30
    patience = 5
    best_val_loss = float("inf")
    patience_counter = 0

    train_losses = []
    val_losses = []

    # Validierungs-Subset
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=1000, shuffle=False)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validierung
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"📊 Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Early Stopping & Speichern
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print("💾 Bestes Modell gespeichert.")
        else:
            patience_counter += 1
            print(f"⏸️  Keine Verbesserung. Geduld: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("🛑 Early Stopping aktiviert.")
                break

    # Loss-Kurve plotten
    os.makedirs("plots", exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Trainings- und Validierungs-Loss")
    plt.savefig("plots/loss_curve.png")
    print("📈 Loss-Kurve gespeichert unter: plots/loss_curve.png")

    # Bestes Modell laden
    model.load_state_dict(torch.load("checkpoints/best_model.pt"))
    model.eval()

    # Klassengenauigkeit berechnen
    correct_per_class = [0] * 10
    total_per_class = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for label, pred in zip(labels, predicted):
                total_per_class[label.item()] += 1
                if label == pred:
                    correct_per_class[label.item()] += 1

    print("\n🎯 Klassengenauigkeit:")
    for i in range(10):
        accuracy = 100 * correct_per_class[i] / total_per_class[i]
        print(f"Ziffer {i}: {accuracy:.2f}% ({correct_per_class[i]}/{total_per_class[i]})")

    total_correct = sum(correct_per_class)
    total = sum(total_per_class)
    print(f"\n✅ Gesamt-Testgenauigkeit: {100 * total_correct / total:.2f}%")

    # Beispielbild + Vorhersage anzeigen
    examples = iter(test_loader)
    example_images, example_labels = next(examples)
    example_images = example_images.to(device)

    with torch.no_grad():
        output = model(example_images)
        _, predicted = torch.max(output, 1)

    # Plot des ersten Bildes
    plt.imshow(example_images[0].cpu().squeeze(), cmap="gray")
    plt.title(f" Vorhersage: {predicted[0].item()} | Tatsächlich: {example_labels[0].item()}")
    plt.axis("off")
    plt.show()


def visualize_network():
    import torchvision  # notwendig für make_grid()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    writer = SummaryWriter("./visualisierung")

    # Dummy-Daten (z. B. erster Batch aus dem echten Loader)
    dummy_loader = iter(train_loader)
    images, labels = next(dummy_loader)
    images, labels = images.to(device), labels.to(device)

    # 1. Netzwerkgraph
    writer.add_graph(model, images)

    # 2. Beispielbilder
    img_grid = torchvision.utils.make_grid(images[:16])
    writer.add_image("Beispiel-Input", img_grid)

    # 3. Dummy-Trainingsschritt für Scalar- und Histogram-Visualisierung
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    outputs = model(images)
    loss = criterion(outputs, labels)
    writer.add_scalar("Loss/Dummy", loss.item(), 0)

    # 4. Histogramme der Parameter (Weights & Biases)
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, 0)

    writer.close()
    print("✅ TensorBoard-Visualisierung erstellt. Starte: tensorboard --logdir=./visualisierung")



def count_trainable_params(model):
    print("\n📊 Übersicht der trainierbaren Parameter pro Layer:\n")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            shape_str = str(tuple(param.shape))  # ✅ Umwandlung in String
            print(f"{name:30} | Shape: {shape_str:20} | Anzahl: {num_params:,}")
            total_params += num_params
    print("-" * 80)
    print(f"✅ Gesamtanzahl trainierbarer Parameter: {total_params:,}\n")
    return total_params




if __name__ == "__main__":
    start_training()
    count_trainable_params(CNN())
    visualize_network()
