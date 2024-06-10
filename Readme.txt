Panoramica Generale
Il progetto sviluppa un modello per generare movimenti delle mani e delle braccia basati su trascrizioni di testo. Utilizza un encoder BERT per il testo e un Transformer per il trattamento delle sequenze di gesti, denoizzando i gesti corrotti attraverso un processo di diffusione. L'obiettivo è apprendere una rappresentazione congiunta testo-gesti che permetta di generare gesti realistici a partire da un input testuale.

Struttura del Progetto
Dataset
Trascrizioni di testo: Frasi o dialoghi che servono come input testuale. Caricate da file Excel.
Sequenze di gesti: Coordinate 3D (x, y, z) delle mani e delle braccia. Caricate da file Excel e processate per ottenere le sequenze di gesti.
Caricamento dei Dati
Funzione load_transcription: Carica le trascrizioni da file Excel.
Funzione load_keypoints: Carica le sequenze di gesti dalle coordinate 3D, suddividendole in mani e braccia.
Funzione load_data: Coordina il caricamento di trascrizioni e sequenze di gesti, gestisce la lunghezza massima delle sequenze.
Modello: DiffusionModel
Text Encoder: Utilizza BERT per ottenere rappresentazioni dense del testo.
Text Projector: Proietta gli embeddings del testo in uno spazio latente di dimensione nascosta.
Gesture Transformer: Un Transformer Encoder che elabora le sequenze di gesti (corrotti).
Initial Pose Projector: Proietta le sequenze di gesti iniziali (mani e braccia) nello spazio latente.
Denoise Projector: Proietta le sequenze di gesti latenti denoizzati nello spazio delle coordinate originali.
Processo di Diffusione
Forward Diffusion: Introduce rumore ai dati gestuali utilizzando parametri di diffusione. La funzione forward_diffusion_process aggiunge rumore progressivamente alle sequenze di gesti.
Reverse Conditional Generation: Utilizza il modello per denoizzare i gesti corrotti, condizionandosi sugli embeddings del testo e sui parametri temporali.
Funzionamento del Modello
Text Encoding e Proiezione:

BERT genera embeddings testuali.
Gli embeddings vengono proiettati in uno spazio latente.
Proiezione delle Posi Iniziali:

Le sequenze di gesti corrotti vengono proiettate nello spazio latente.
Combinazione degli Input:

Gli embeddings testuali, i gesti corrotti e gli embeddings temporali vengono combinati.
Trasformazione tramite Transformer:

Il Transformer Encoder elabora le sequenze combinate, catturando le dipendenze temporali.
Proiezione di Denoising:

Gli output del Transformer vengono proiettati indietro nello spazio delle coordinate originali, ottenendo le sequenze di gesti denoizzati.
Addestramento del Modello
Preparazione dei Dati:

Caricamento di batch di dati dal DataLoader.
Generazione degli embeddings temporali.
Applicazione del processo di diffusione per ottenere sequenze di gesti corrotti.
Forward Pass:

Passaggio dei dati attraverso il modello per ottenere i gesti denoizzati.
Calcolo della Loss:

La funzione di perdita (MPJPE) calcola l'errore tra i gesti denoizzati e il rumore aggiunto.
Backward Pass:

Calcolo dei gradienti e aggiornamento dei parametri del modello utilizzando un ottimizzatore (AdamW).
Salvataggio del Modello:

I checkpoint vengono salvati periodicamente durante l'addestramento.
Workflow del Progetto
Preparazione dei Dati:

Caricamento e preprocessing delle trascrizioni e delle sequenze di gesti.
Divisione in batch per l'addestramento.
Inizializzazione del Modello:

Creazione del modello di diffusione e inizializzazione dei suoi parametri.
Addestramento del Modello:

