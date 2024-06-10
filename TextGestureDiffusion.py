import torch
from torch import nn
from transformers import BertModel

# Definizione del modello di diffusione
class DiffusionModel(nn.Module):
    def __init__(self, text_dim, hidden_dim, gesture_dim, num_layers):
        super(DiffusionModel, self).__init__()
        # Inizializza il modello BERT pre-addestrato
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        # Proiettore per il testo codificato
        self.text_projector = nn.Linear(text_dim, hidden_dim)
        # Transformer Encoder per i gesti
        self.gesture_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=num_layers
        )
        # Proiettori iniziali per i gesti corrotti
        self.initial_pose_projector_hands = nn.Linear(gesture_dim, hidden_dim)
        self.initial_pose_projector_arms = nn.Linear(gesture_dim, hidden_dim)
        # Proiettori per denoisare i gesti
        self.denoise_projector_hands = nn.Linear(hidden_dim, gesture_dim)
        self.denoise_projector_arms = nn.Linear(hidden_dim, gesture_dim)

    def forward(self, text, text_attention_mask, corrupted_hand_gestures, corrupted_arm_gestures, time_embedding):
        batch_size, seq_len_hands, _ = corrupted_hand_gestures.size()
        _, seq_len_arms, _ = corrupted_arm_gestures.size()
        device = text.device

        # Codifica del testo e proiezione
        text_output = self.text_encoder(text, attention_mask=text_attention_mask)
        text_embeds = text_output['last_hidden_state']
        text_embeds = self.text_projector(text_embeds)
        #print(f"text_embeds shape: {text_embeds.shape}")

        # Proiezione iniziale dei gesti corrotti
        corrupted_hand_gestures = self.initial_pose_projector_hands(corrupted_hand_gestures)
        corrupted_arm_gestures = self.initial_pose_projector_arms(corrupted_arm_gestures)
        #print(f"corrupted_hand_gestures shape after projection: {corrupted_hand_gestures.shape}")
        #print(f"corrupted_arm_gestures shape after projection: {corrupted_arm_gestures.shape}")

        # Assicura che text_embeds e time_embedding abbiano la lunghezza corretta
        max_seq_len = max(seq_len_hands, seq_len_arms)
        text_embeds = text_embeds[:, :max_seq_len, :]
        time_embedding = time_embedding[:, :max_seq_len, :]

        # Combina gli input
        combined_input_hands = torch.cat((text_embeds, corrupted_hand_gestures, time_embedding), dim=1)
        combined_input_hands = combined_input_hands.permute(1, 0, 2)
        #print(f"combined_input_hands shape: {combined_input_hands.shape}")

        combined_input_arms = torch.cat((text_embeds, corrupted_arm_gestures, time_embedding), dim=1)
        combined_input_arms = combined_input_arms.permute(1, 0, 2)
        #print(f"combined_input_arms shape: {combined_input_arms.shape}")

        # Codifica tramite Transformer
        transformer_output_hands = self.gesture_transformer(combined_input_hands)
        transformer_output_hands = transformer_output_hands.permute(1, 0, 2)
        #print(f"transformer_output_hands shape: {transformer_output_hands.shape}")

        transformer_output_arms = self.gesture_transformer(combined_input_arms)
        transformer_output_arms = transformer_output_arms.permute(1, 0, 2)
        #print(f"transformer_output_arms shape: {transformer_output_arms.shape}")

        # Proiezione per denoisare i gesti
        denoised_gestures_hands = self.denoise_projector_hands(transformer_output_hands)
        #print(f"denoised_gestures_hands shape before view: {denoised_gestures_hands.shape}")

        denoised_gestures_arms = self.denoise_projector_arms(transformer_output_arms)
        #print(f"denoised_gestures_arms shape before view: {denoised_gestures_arms.shape}")

        # Ridimensionamento per corrispondere alla lunghezza della sequenza
        if denoised_gestures_hands.size(1) > seq_len_hands:
            denoised_gestures_hands = denoised_gestures_hands[:, :seq_len_hands, :]
        elif denoised_gestures_hands.size(1) < seq_len_hands:
            pad_amount = seq_len_hands - denoised_gestures_hands.size(1)
            padding = torch.zeros(batch_size, pad_amount, denoised_gestures_hands.size(2), device=device)
            denoised_gestures_hands = torch.cat((denoised_gestures_hands, padding), dim=1)

        if denoised_gestures_arms.size(1) > seq_len_arms:
            denoised_gestures_arms = denoised_gestures_arms[:, :seq_len_arms, :]
        elif denoised_gestures_arms.size(1) < seq_len_arms:
            pad_amount = seq_len_arms - denoised_gestures_arms.size(1)
            padding = torch.zeros(batch_size, pad_amount, denoised_gestures_arms.size(2), device=device)
            denoised_gestures_arms = torch.cat((denoised_gestures_arms, padding), dim=1)

        return denoised_gestures_hands, denoised_gestures_arms

'''
text_embeds shape: torch.Size([1, 512, 512])
corrupted_hand_gestures shape after projection: torch.Size([1, 46851, 512])
corrupted_arm_gestures shape after projection: torch.Size([1, 6750, 512])
combined_input_hands shape: torch.Size([47364, 1, 512])
combined_input_arms shape: torch.Size([7263, 1, 512])
transformer_output_hands shape: torch.Size([1, 47364, 512])
transformer_output_arms shape: torch.Size([1, 7263, 512])
denoised_gestures_hands shape before view: torch.Size([1, 47364, 3])
denoised_gestures_arms shape before view: torch.Size([1, 7263, 3])
'''