import io
import paramiko
import os
import csv
import pandas as pd

# Configurazione del server e delle credenziali
hostname = '193.204.187.47'
port = 22  # Porta predefinita per SFTP/SSH
username = 'droberto'
password = 'droberto'
remote_directory = '/ext/emanuele/LISDataset/transcripts'
local_directory = r'D:\Lis\dataset\trascrizioni_xlsx'  # Cambia con il percorso corretto

# Creazione del client SSH
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())


def convert_txt_to_xlsx(local_txt_path, local_xlsx_path):
    with open(local_txt_path, 'r', encoding='utf-8') as txt_file:
        lines = txt_file.readlines()

    data = [line.strip().split('--') for line in lines]

    df = pd.DataFrame(data)
    df.to_excel(local_xlsx_path, index=False, header=False)


try:
    # Controlla se la directory locale esiste, altrimenti la crea
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)

    # Connessione al server
    ssh.connect(hostname, port, username, password)
    sftp = ssh.open_sftp()

    print("Connessione stabilita!")

    # Elenca i file nella directory remota
    files = sftp.listdir(remote_directory)
    print(f"File nella directory remota ({remote_directory}):")
    for file in files:
        print(file)

        # Scarica il file dalla directory remota alla directory locale
        remote_file_path = os.path.join(remote_directory, file)
        remote_file_path = remote_file_path.replace('\\', '/')  # Sostituisci tutte le barre rovesciate con barre normali
        local_txt_path = os.path.join(local_directory, file)
        sftp.get(remote_file_path, local_txt_path)

        # Converti il file di testo scaricato in XLSX
        local_xlsx_path = os.path.splitext(local_txt_path)[0] + '.xlsx'
        convert_txt_to_xlsx(local_txt_path, local_xlsx_path)

        print(f"File {file} scaricato e convertito con successo in {local_xlsx_path}!")

    # Chiusura della connessione SFTP
    sftp.close()

except Exception as e:
    print(f"Errore durante la connessione o il trasferimento dei file: {e}")

finally:
    # Chiusura della connessione SSH
    ssh.close()

import os

# Definisci la cartella di input


# Elenco dei file nella cartella di input
files = os.listdir(local_directory)

# Itera su ogni file nella cartella di input
for file in files:
    # Verifica se il file è un file .txt
    if file.endswith('.txt'):
        # Crea il percorso completo per il file
        percorso = os.path.join(local_directory, file)

        # Rimuovi il file .txt
        os.remove(percorso)

print("Operazione completata!")

import os
from openpyxl import load_workbook

# Definisci la cartella di input
cartella = local_directory

# Elenco dei file nella cartella di input
files = os.listdir(cartella)

# Itera su ogni file nella cartella di input
for file in files:
    # Verifica se il file è un file .xlsx
    if file.endswith('.xlsx'):
        # Crea il percorso completo per il file
        percorso = os.path.join(cartella, file)

        # Carica il workbook
        wb = load_workbook(percorso)

        # Seleziona il primo foglio di lavoro
        foglio = wb.active

        # Rimuovi la prima riga
        foglio.delete_rows(1)

        # Salva il workbook modificato sovrascrivendo il file originale
        wb.save(percorso)

print("Operazione completata!")
