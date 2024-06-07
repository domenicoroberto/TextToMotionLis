import logging
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc
import math

from dataset import dataset
from TextGestureDiffusion import DiffusionModel
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Rimani coerente con la definizione della funzione custom_loss
def custom_loss(denoised_gestures, true_gestures):
    mse_loss = torch.nn.MSELoss()
    return mse_loss(denoised_gestures, true_gestures)

def forward_diffusion_process(gestures, t, betas):
    # Rimani coerente con la definizione originale della funzione
    noise = torch.randn_like(gestures)
    batch_size, seq_len, gesture_dim = gestures.size()
    betas = betas.to(device)

    sqrt_alphas_cumprod_t = torch.sqrt(torch.cumprod(1 - betas, dim=0)[t])
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - torch.cumprod(1 - betas, dim=0)[t])

    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1).unsqueeze(-1).expand(batch_size, seq_len, gesture_dim)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1).unsqueeze(-1).expand(batch_size, seq_len, gesture_dim)

    corrupted_gestures = sqrt_alphas_cumprod_t * gestures + sqrt_one_minus_alphas_cumprod_t * noise
    return corrupted_gestures, noise

def reverse_conditional_generation(model, text, text_attention_mask, corrupted_gestures, time_embedding):
    return model(text, text_attention_mask, corrupted_gestures[0], corrupted_gestures[1], time_embedding)

def get_sinusoidal_embeddings(seq_len, dim, device):
    if dim % 2 != 0:
        raise ValueError("The dimension (dim) must be even.")
    embeddings = torch.zeros(seq_len, dim, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / dim))
    embeddings[:, 0::2] = torch.sin(position * div_term)
    embeddings[:, 1::2] = torch.cos(position * div_term)
    return embeddings

def train_model(train_loader, model, optimizer, epochs, betas, device, accumulation_steps):
    model.train()
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

    for epoch in range(epochs):
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, batch in enumerate(train_loader):
            text, attention_mask, hand_keypoints, arm_keypoints = batch
            text = text.to(device)
            attention_mask = attention_mask.to(device)
            hand_keypoints = hand_keypoints.to(device)
            arm_keypoints = arm_keypoints.to(device)

            batch_size, hand_seq_len, _ = hand_keypoints.size()
            _, arm_seq_len, _ = arm_keypoints.size()



            embedding_dim = 512  # Must be even

            max_seq_len = max(hand_seq_len, arm_seq_len)

            # Sinusoidal time embeddings
            time_embeddings = get_sinusoidal_embeddings(max_seq_len, embedding_dim, device).unsqueeze(0).repeat(batch_size, 1, 1)
            t = torch.randint(0, len(betas), (batch_size,), device=device).long()

            # Perform diffusion process for hands and arms separately
            corrupted_hand_gestures, noise_hand = forward_diffusion_process(hand_keypoints, t, betas)
            corrupted_arm_gestures, noise_arm = forward_diffusion_process(arm_keypoints, t, betas)
            time_embedding = time_embeddings[:, t, :]

            with torch.cuda.amp.autocast():  # For mixed precision training
                # Generate denoised gestures for hands and arms
                denoised_hand_gestures, denoised_arm_gestures = reverse_conditional_generation(
                    model, text, attention_mask, (corrupted_hand_gestures, corrupted_arm_gestures), time_embedding
                )

                # Calculate loss separately for hands and arms
                loss_hand = custom_loss(denoised_hand_gestures, noise_hand)
                loss_arm = custom_loss(denoised_arm_gestures, noise_arm)
                loss = loss_hand + loss_arm

                loss = loss / accumulation_steps

            scaler.scale(loss).backward()  # For mixed precision training

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)  # For mixed precision training
                scaler.update()  # For mixed precision training
                optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
            gc.collect()
            torch.cuda.empty_cache()

        progress_bar.close()
        if (epoch + 1) % 10 == 0:
            checkpoint_dir=r"D:\Lis\progetto\models"
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            logger.info(f'Model checkpoint saved at {checkpoint_path}')

if __name__ == '__main__':
    gesture_dim = 3  # Dimension of coordinates (x, y, z)
    text_dim = 768  # Dimension of BERT embeddings
    hidden_dim = 512  # Dimension of reduced transformer hidden layers
    num_layers = 4    # Number of layers in the reduced transformer

    model = DiffusionModel(text_dim, hidden_dim, gesture_dim, num_layers)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader = dataset()

    betas = torch.linspace(0.0001, 0.02, 1000)  # Example beta values for diffusion process
    accumulation_steps = 2  # Number of steps to accumulate gradients

    def print_gpu_memory():
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2} MB")

    print_gpu_memory()


    train_model(train_loader, model, optimizer, epochs=100, betas=betas, device=device, accumulation_steps=accumulation_steps)
