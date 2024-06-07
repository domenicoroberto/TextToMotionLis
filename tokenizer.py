import openpyxl


import openpyxl

def modifica_intestazione(file_path):
    # Carica il file Excel
    workbook = openpyxl.load_workbook(file_path)
    # Seleziona il primo foglio di lavoro
    sheet = workbook.active

    # Modifica la seconda cella della prima riga
    sheet.cell(row=1, column=2).value = 'hands'

    # Salva il file Excel con le modifiche
    workbook.save(file_path)


# Esempio di utilizzo
file_path = r"C:\Users\Domenico\Desktop\Tg_Noi_Lis_03_05_2023_112.xlsx"  # Sostituisci con il percorso del tuo file
modifica_intestazione(file_path)
