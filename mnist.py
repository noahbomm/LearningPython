import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
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

# 2. Einfaches neuronales Netz definieren
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. Modell, Loss, Optimierer
def start_training():
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Optional: TensorBoard Writer für Trainingsmetriken
    # writer = SummaryWriter(log_dir="runs/mnist_training")

    for epoch in range(5):  # z.B. 5 Epochen
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Optional: loss loggen
            # writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + batch_idx)

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # writer.close()

    # Test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    # Beispielbild anzeigen
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    plt.imshow(example_data[1][0], cmap="gray")
    plt.title(f"Beispielhafte Vorhersage: {example_targets[1].item()}")
    plt.show()

# 4. Netzwerkgraph visualisieren
def visualize_network():
    import torchvision

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN().to(device)
    model.eval()

    writer = SummaryWriter(log_dir="visualisierung")

    # Dummy-Daten für Visualisierung
    example_data, example_labels = next(iter(DataLoader(train_dataset, batch_size=64)))
    example_data, example_labels = example_data.to(device), example_labels.to(device)

    # 1. Graph visualisieren
    try:
        writer.add_graph(model, example_data)
        print("✅ Graph erfolgreich in TensorBoard gespeichert.")
    except Exception as e:
        print("❌ Fehler beim Hinzufügen des Graphen:", e)

    # 2. Beispielbild loggen
    img_grid = torchvision.utils.make_grid(example_data[:16])
    writer.add_image("Beispielbilder", img_grid, global_step=0)

    # 3. Dummy-Loss berechnen und loggen
    criterion = nn.CrossEntropyLoss()
    outputs = model(example_data)
    loss = criterion(outputs, example_labels)
    writer.add_scalar("Loss/Dummy", loss.item(), 0)

    # 4. Gewichts-Histogramme loggen
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, 0)

    writer.close()
    print("✅ TensorBoard-Logs gespeichert. Starte mit: tensorboard --logdir=visualisierung")

# 5. Hauptprogramm
if __name__ == "__main__":
    # start_training()  # Optional ausführen
    visualize_network()

# Parameter berechnen

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
    count_trainable_params(SimpleNN())
    visualize_network()