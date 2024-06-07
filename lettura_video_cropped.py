import os
import paramiko
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip

# Configurazione del server e delle credenziali
hostname = '193.204.187.47'
port = 22  # Porta predefinita per SFTP/SSH
username = 'droberto'
password = 'droberto'
remote_directory = '/ext/emanuele/LISDataset/video_cropped'
local_directory = r'D:\Lis\dataset\trascrizioni_xlsx'  # Cambia con il percorso corretto
output_folder = r'D:\Lis\dataset\video_tagliati'  # Cambia con il percorso della cartella di output desiderata

# Creazione del client SSH
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())


def convert_txt_to_xlsx(local_txt_path, local_xlsx_path):
    with open(local_txt_path, 'r', encoding='utf-8') as txt_file:
        lines = txt_file.readlines()

    data = [line.strip().split('--') for line in lines]

    df = pd.DataFrame(data)
    df.to_excel(local_xlsx_path, index=False, header=False)


def taglia_video(video, start_time, end_time, output_path):
    subclip = video.subclip(start_time, end_time)
    subclip.write_videofile(output_path)
    subclip.close()


try:
    # Controlla se la directory locale esiste, altrimenti la crea
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)

    # Controlla se la cartella di output esiste, altrimenti la crea
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Connessione al server
    ssh.connect(hostname, port, username, password)
    sftp = ssh.open_sftp()

    print("Connessione stabilita!")

    # Elenca i file nella directory remota
    files = sftp.listdir(remote_directory)
    print(f"File nella directory remota ({remote_directory}):")
    for file in files:
        print(file)
        remote_file_path = os.path.join(remote_directory, file)
        remote_file_path = remote_file_path.replace('\\', '/')  # Sostituisci tutte le barre rovesciate con barre normali

        # Se il file è un video
        if file.endswith('.mp4'):
            # Rimuovi "_cropped.mp4" dal nome del video
            video_name = os.path.splitext(file)[0].replace("_cropped", "")


            # Crea una cartella per il video nella cartella di output
            video_output_folder = os.path.join(output_folder, video_name)
            if not os.path.exists(video_output_folder):
                os.makedirs(video_output_folder)


            # Trova i file xlsx con lo stesso nome nella directory locale
            for xlsx_file in os.listdir(local_directory):
                if xlsx_file.endswith('.xlsx') and os.path.splitext(xlsx_file)[0] == video_name:
                    print(f"Trovato un file xlsx corrispondente per il video {video_name}: {xlsx_file}")

                    # Leggi il file XLSX per ottenere i tempi di start e end
                    xlsx_path = os.path.join(local_directory, xlsx_file)
                    df = pd.read_excel(xlsx_path, header=None)

                    # Per ogni riga nella trascrizione
                    for index, row in df.iterrows():
                        start_time = row[0]  # Tempo di start
                        end_time = row[1]    # Tempo di end
                        remote_file_path = os.path.join(remote_directory, file)
                        remote_file_path = remote_file_path.replace('\\', '/')
                        # Scarica temporaneamente il video in un file locale
                        video_path = remote_file_path
                        local_video_path = os.path.join(local_directory, file)
                        sftp.get(video_path, local_video_path)

                        # Carica il video da file temporaneo e taglia il video
                        video_clip = VideoFileClip(local_video_path)
                        output_filename = f"{video_name}_{index}.mp4"  # Aggiungi un numero di sequenza al nome del file
                        output_path = os.path.join(video_output_folder, output_filename)
                        taglia_video(video_clip, start_time, end_time, output_path)

                        # Elimina il file temporaneo del video
                        os.remove(local_video_path)

    print("Taglio dei video completato!")

except Exception as e:
    print(f"Errore durante la connessione o il taglio dei video: {e}")

finally:
    # Chiusura della connessione SFTP
    sftp.close()
    # Chiusura della connessione SSH
    ssh.close()
