import os
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm
import numpy as np

# Inizializza Mediapipe Hands e Pose
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils


# Funzione per calcolare gli angoli di rotazione
def calculate_wrist_rotation(hand_landmarks):
    wrist = hand_landmarks[0]  # id=0 per il polso
    index_mcp = hand_landmarks[5]  # id=5 per l'articolazione MCP dell'indice
    pinky_mcp = hand_landmarks[17]  # id=17 per l'articolazione MCP del mignolo

    # Calcolare i vettori dal polso
    vec_index = np.array([index_mcp.x - wrist.x, index_mcp.y - wrist.y, index_mcp.z - wrist.z])
    vec_pinky = np.array([pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z])

    # Calcolare il vettore normale al piano formato da index e pinky
    normal = np.cross(vec_index, vec_pinky)

    # Calcolare gli angoli di rotazione
    rot_x = np.arctan2(normal[1], normal[2])
    rot_y = np.arctan2(normal[0], normal[2])
    rot_z = np.arctan2(normal[0], normal[1])

    return rot_x, rot_y, rot_z


# Funzione per estrarre i keypoints da un singolo frame
def extract_keypoints(frame):
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_results = hands.process(imgRGB)
    pose_results = pose.process(imgRGB)
    frame_keypoints = {'frame': [], 'hands': [], 'arms': [], 'wrist_rotation': []}

    # Se vengono rilevate mani
    if hands_results.multi_hand_landmarks:
        for handLms in hands_results.multi_hand_landmarks:
            hand_keypoints = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                hand_keypoints.append({'id': id, 'x': lm.x, 'y': lm.y, 'z': lm.z})
            frame_keypoints['hands'].append(hand_keypoints)

            # Calcolare la rotazione del polso
            rot_x, rot_y, rot_z = calculate_wrist_rotation(handLms.landmark)
            frame_keypoints['wrist_rotation'].append({'rot_x': rot_x, 'rot_y': rot_y, 'rot_z': rot_z})

    # Se vengono rilevate pose (braccia)
    if pose_results.pose_landmarks:
        arm_keypoints = []
        for id in [11, 13, 15, 12, 14, 16]:  # Keypoints per spalla, gomito, polso sinistro e destro
            lm = pose_results.pose_landmarks.landmark[id]
            arm_keypoints.append({'id': id, 'x': lm.x, 'y': lm.y, 'z': lm.z})
        frame_keypoints['arms'].append(arm_keypoints)

    return frame_keypoints


# Funzione per estrarre i keypoints da un video e salvare in un DataFrame
def extract_video_keypoints(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []

    try:
        # Ottieni la lunghezza totale del video per la barra di avanzamento
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = tqdm(total=total_frames, desc=os.path.basename(video_path), unit='frame')

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_keypoints = extract_keypoints(frame)
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            frame_keypoints['frame'] = frame_number
            keypoints_list.append(frame_keypoints)

            # Aggiorna la barra di avanzamento
            progress_bar.update(1)

    except KeyboardInterrupt:
        print("Interruzione dall'utente.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Chiudi la barra di avanzamento
        progress_bar.close()

    # Creazione del DataFrame
    df = pd.DataFrame(keypoints_list)
    # Salvataggio in un file Excel nella cartella di output appropriata
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    excel_file = os.path.join(output_folder, f"{video_name}.xlsx")
    df.to_excel(excel_file, index=False)


# Percorso della cartella contenente i video
input_folder = r"D:\Lis\dataset\video_tagliati"
# Cartella di output
output_folder = r"D:\Lis\dataset\keypoint_video_tagliati"

# Ciclo su ogni cartella nella directory di input
for root, dirs, files in os.walk(input_folder):
    for dir_name in dirs:
        # Percorso della cartella di input
        input_dir_path = os.path.join(root, dir_name)
        # Percorso della cartella di output corrispondente
        output_dir_path = os.path.join(output_folder, dir_name)
        # Crea la cartella di output se non esiste già
        os.makedirs(output_dir_path, exist_ok=True)

        # Ciclo su ogni file video nella cartella di input
        for file_name in os.listdir(input_dir_path):
            if file_name.endswith('.mp4'):
                # Percorso completo del video di input
                video_path = os.path.join(input_dir_path, file_name)
                # Estrai i keypoints e salva l'Excel nella cartella di output corrispondente
                extract_video_keypoints(video_path, output_dir_path)
