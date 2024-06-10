import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

# Definisce una classe Dataset per PyTorch, per gestire trascrizioni e punti chiave delle mani e delle braccia.
class GestureDataset(Dataset):
    def __init__(self, transcripts, hands, arms, tokenizer, max_length_hands=None, max_length_arms=None):
        self.transcripts = transcripts
        self.hands = hands
        self.arms = arms
        self.tokenizer = tokenizer
        self.max_length_hands = max_length_hands
        self.max_length_arms = max_length_arms

    # Restituisce la lunghezza del dataset (numero di trascrizioni).
    def __len__(self):
        return len(self.transcripts)

    # Restituisce un campione del dataset alla posizione data (trascrizione, punti chiave delle mani e delle braccia).
    def __getitem__(self, idx):
        transcript = self.transcripts[idx]
        hand_keypoints = self.hands[idx]
        arm_keypoints = self.arms[idx]

        # Tokenizza la trascrizione usando BERT.
        encoded_inputs = self.tokenizer(transcript, padding='max_length', truncation=True, return_tensors='pt')

        input_ids = encoded_inputs['input_ids'].squeeze(0)
        attention_mask = encoded_inputs['attention_mask'].squeeze(0)

        # Pad (riempie) i punti chiave delle mani e delle braccia fino alla lunghezza massima.
        hand_keypoints = self.pad_keypoints(hand_keypoints, self.max_length_hands)
        arm_keypoints = self.pad_keypoints(arm_keypoints, self.max_length_arms)

        return input_ids, attention_mask, hand_keypoints, arm_keypoints

    # Funzione per riempire (pad) i punti chiave fino alla lunghezza massima specificata.
    def pad_keypoints(self, keypoints, max_length):
        padded_keypoints = torch.zeros((max_length, 3), dtype=torch.float32)
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        padded_keypoints[:len(keypoints)] = keypoints
        return padded_keypoints

# Carica le trascrizioni da un file Excel.
def load_transcription(file_path):
    df_transcription = pd.read_excel(file_path, sheet_name='Sheet1', header=None)
    transcriptions = df_transcription.iloc[:, 2].tolist()
    return transcriptions

# Carica i punti chiave delle mani e delle braccia da un file Excel, aggiornando i contatori.
def load_keypoints(args, counters):
    folder, file_name, input_dir = args
    keypoint_file_path = os.path.join(input_dir, folder, file_name)

    if os.path.exists(keypoint_file_path):
        df_keypoints = pd.read_excel(keypoint_file_path, sheet_name='Sheet1')

        valori_numerici = []
        valori_numerici2 = []

        for index, row in df_keypoints.iterrows():
            stringa_dizionari = row[2]
            stringa_ = row[1]

            lista_dizionari = eval(stringa_dizionari)
            lista = eval(stringa_)

            for dizionario in lista_dizionari:
                for punto in dizionario:
                    valori_numerici.append((punto['x'], punto['y'], punto['z']))
                    counters['arms'] += 1  # Aggiorna il contatore dei punti chiave delle braccia
            for dizionario in lista:
                for punto in dizionario:
                    valori_numerici2.append((punto['x'], punto['y'], punto['z']))
                    counters['hands'] += 1  # Aggiorna il contatore dei punti chiave delle mani

        return valori_numerici2, valori_numerici

    return None, None

# Carica i dati delle trascrizioni e dei punti chiave, e calcola le lunghezze massime delle sequenze di punti chiave.
def load_data(input_dir, transcription_dir, num_keypoint_folders, num_transcription_files):
    hands_array = []
    arms_array = []
    transcriptions_array = []
    keypoints_files_processed = []
    transcription_files_processed = []
    counters = {'hands': 0, 'arms': 0}  # Inizializza i contatori

    # Ottiene le cartelle dei punti chiave e i file di trascrizione.
    folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    transcription_files = [f for f in os.listdir(transcription_dir) if f.endswith('.xlsx')]

    # Carica le trascrizioni.
    with tqdm(total=num_transcription_files, desc='Loading transcriptions') as pbar:
        for file_name in transcription_files[:num_transcription_files]:
            file_path = os.path.join(transcription_dir, file_name)
            transcription_files_processed.append(file_name)

            transcriptions = load_transcription(file_path)
            transcriptions_array.extend(transcriptions)
            pbar.update(1)

    # Calcola il numero totale di punti chiave da caricare.
    total_keypoints = len(transcriptions_array) * num_keypoint_folders
    with tqdm(total=total_keypoints, desc='Loading keypoints') as pbar:
        for file_name in transcription_files[:num_transcription_files]:
            file_path = os.path.join(transcription_dir, file_name)

            for i in range(len(transcriptions_array)):
                for folder in folders:
                    keypoint_file_path = os.path.join(input_dir, folder, f"{file_name.split('.')[0]}_{i}.xlsx")
                    hands_data, arms_data = load_keypoints((folder, f"{file_name.split('.')[0]}_{i}.xlsx", input_dir), counters)
                    if hands_data is not None and arms_data is not None:
                        hands_array.append(hands_data)
                        arms_array.append(arms_data)
                        pbar.update(1)
    print(f"Total hand keypoints added: {counters['hands']}")
    print(f"Total arm keypoints added: {counters['arms']}")

    # Calcola le lunghezze massime delle sequenze di punti chiave.
    max_seq_length_hands = max(len(seq) for seq in hands_array)
    max_seq_length_arms = max(len(seq) for seq in arms_array)

    return hands_array, arms_array, transcriptions_array, keypoints_files_processed, transcription_files_processed, max_seq_length_hands, max_seq_length_arms

# Percorsi ai dati di input.
input_dir = r"D:\Lis\dataset\keypoint_video_tagliati"
transcription_dir = r"D:\Lis\dataset\trascrizioni_xlsx"

# Funzione principale per caricare il dataset e creare un DataLoader.
def dataset():
    num_keypoint_folders = 1  # e.g., prendere solo la prima cartella
    num_transcription_files = 105  # e.g., prendere solo i primi 105 file di trascrizione

    hands_array, arms_array, transcriptions_array, keypoints_files_processed, transcription_files_processed, max_seq_length_hands, max_seq_length_arms = load_data(
        input_dir, transcription_dir, num_keypoint_folders, num_transcription_files)

    print(f"Number of loaded data sets (hands): {len(hands_array)}")
    print(f"Number of loaded data sets (arms): {len(arms_array)}")
    print(f"Number of loaded transcriptions: {len(transcriptions_array)}")

    print(f"Processed keypoint files: {keypoints_files_processed}")
    print(f"Processed transcription files: {transcription_files_processed}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    gesture_dataset = GestureDataset(transcriptions_array, hands_array, arms_array, tokenizer, max_length_hands=max_seq_length_hands, max_length_arms=max_seq_length_arms)
    train_loader = DataLoader(gesture_dataset, batch_size=1, shuffle=True)

    return train_loader


if __name__ == '__main__':
    train_loader = dataset()

    for input_ids, attention_mask, hands, arms in train_loader:
        print("Hands size:", hands.size())
        print("Attention mask size:", attention_mask.size())
        print("Arms size:", arms.size())
