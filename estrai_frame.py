import os
from moviepy.editor import VideoFileClip

def crea_cartelle_frames(input_dir, output_dir):
    # Lista delle cartelle nella directory di input
    folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

    # Per ogni cartella nella directory di input
    for folder in folders:
        # Crea una cartella con lo stesso nome nella directory di output
        output_folder_path = os.path.join(output_dir, folder)
        os.makedirs(output_folder_path)

        # Cartella di input per il video
        video_input_folder = os.path.join(input_dir, folder)

        # Lista dei file video nella cartella di input
        video_files = [f for f in os.listdir(video_input_folder) if f.endswith('.mp4')]

        # Per ogni video nella cartella di input
        for video_file in video_files:
            video_path = os.path.join(video_input_folder, video_file)

            # Crea una cartella per i frame nella cartella di output corrispondente
            frame_folder_path = os.path.join(output_folder_path, os.path.splitext(video_file)[0])
            os.makedirs(frame_folder_path)

            # Estrae i frame dal video e li salva nella cartella dei frame
            clip = VideoFileClip(video_path)
            clip.write_images_sequence(os.path.join(frame_folder_path, "frame%04d.png"))
            clip.close()

# Esempio di utilizzo
input_directory = r"D:\Lis\dataset\video_tagliati"
output_directory = r"D:\Lis\dataset\frame_video"

crea_cartelle_frames(input_directory, output_directory)
