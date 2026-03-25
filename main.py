import mlflow.pytorch
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser(description="A simple greeting program.")
parser.add_argument(
    "epochs", nargs="?", type=int, default=50, help="Number of training epochs."
)
args = parser.parse_args()
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import mlflow
import os

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Assignment3_MarkSamuel")
# Load data
data = np.load("data/shorts.npy")  # shape: (N, 784)

# Normalize to [-1, 1] (important for GAN with Tanh)
data = data.astype(np.float32) / 255.0
data = (data - 0.5) * 2

# Convert to tensor
tensor_data = torch.tensor(data)

dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Tanh(),  # because data is [-1,1]
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

noise_dim = 100
G = Generator(noise_dim).to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()

G_learning_rate = 0.0001
D_learning_rate = 0.0000001

optimizer_G = optim.Adam(G.parameters(), lr=G_learning_rate)
optimizer_D = optim.Adam(D.parameters(), lr=D_learning_rate)


with mlflow.start_run():
    epochs = args.epochs
    mlflow.log_param("G_learning_rate", G_learning_rate)
    mlflow.log_param("D_learning_rate", D_learning_rate)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("noise_dim", 100)
    mlflow.log_param("batch_size", 32)
    mlflow.set_tag("student_id", "202201857")

    for epoch in range(epochs):
        for real_batch in dataloader:
            real_images = real_batch[0].to(device)
            batch_size = real_images.size(0)

            # =====================
            # Train Discriminator
            # =====================
            optimizer_D.zero_grad()

            # Real labels = 1
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Real loss
            outputs_real = D(real_images)
            loss_real = criterion(outputs_real, real_labels)

            # Fake images
            z = torch.randn(batch_size, noise_dim).to(device)
            fake_images = G(z)

            outputs_fake = D(fake_images.detach())
            loss_fake = criterion(outputs_fake, fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            # =====================
            # Train Generator
            # =====================
            optimizer_G.zero_grad()

            outputs = D(fake_images)
            loss_G = criterion(outputs, real_labels)  # trick D

            loss_G.backward()
            optimizer_G.step()
        mlflow.log_metric("loss_D", loss_D.item(), step=epoch)
        mlflow.log_metric("loss_G", loss_G.item(), step=epoch)
        mlflow.log_metric(
            "D_accuracy",
            ((outputs_real > 0.5).float().mean() + (outputs_fake < 0.5).float().mean())
            / 2,
            step=epoch,
        )
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}"
        )
    G_input = torch.randn(2, noise_dim).cpu().numpy()
    mlflow.pytorch.log_model(
        G, export_model=False, name=f"generator_epoch_{epoch+1}", input_example=G_input
    )
    D_input = torch.randn(2, 784).cpu().numpy()
    mlflow.pytorch.log_model(
        D,
        export_model=False,
        name=f"discriminator_epoch_{epoch+1}",
        input_example=D_input,
    )
    run_id = mlflow.active_run().info.run_id

G.eval()

with open("model_info.txt", "w") as f:
    f.write(run_id)
with torch.no_grad():
    z = torch.randn(16, noise_dim).to(device)
    samples = G(z).cpu().numpy()

samples = (samples + 1) / 2  # back to [0,1]

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(samples[i].reshape(28, 28), cmap="gray")
    ax.axis("off")
plt.show()
