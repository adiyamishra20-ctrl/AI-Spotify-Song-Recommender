import numpy as np
import torch
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity

class SongAutoencoder(nn.Module):
    def __init__(self, input_dim=10, latent_dim=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,32), nn.ReLU(),
            nn.Linear(32,16), nn.ReLU(),
            nn.Linear(16,latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,16), nn.ReLU(),
            nn.Linear(16,32), nn.ReLU(),
            nn.Linear(32,input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

def train_autoencoder(X, epochs=120, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    model = SongAutoencoder(X.shape[1], min(5, X.shape[1]//2))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X_tensor), X_tensor)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        Z = model.encode(X_tensor).numpy()
    return model, Z

def recommend(embeddings, query_vector, top_n=5):
    sims = cosine_similarity(embeddings, query_vector.reshape(1,-1)).ravel()
    idx = np.argsort(-sims)[:top_n]
    return idx, sims[idx]
