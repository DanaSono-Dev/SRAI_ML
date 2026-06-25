import itertools
import os

import pandas as pd
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

                                            # CONFIGURACION

DATA_DIR       = os.getenv("DATA_DIR")
PROJECT        = os.getenv("PROJECT")
MODEL_WEIGHTS  = os.getenv("MODEL_WEIGHTS", "yolov8s-cls.pt")

EPOCHS_LIST    = [int(x)   for x in os.getenv("EPOCHS_LIST",   "50,100").split(",")]
LR_LIST        = [float(x) for x in os.getenv("LR_LIST",       "0.01,0.001").split(",")]
DROPOUT_LIST   = [float(x) for x in os.getenv("DROPOUT_LIST",  "0.3,0.5").split(",")]

IMGSZ          = int(os.getenv("IMGSZ",   "224"))
BATCH          = int(os.getenv("BATCH",   "32"))
DEVICE         = int(os.getenv("DEVICE",  "0"))
EXIST_OK       = os.getenv("EXIST_OK", "False").lower() == "true"

                                            # MAIN - GRILLA

if __name__ == '__main__':
    os.makedirs(PROJECT, exist_ok=True)

    combinaciones = list(itertools.product(EPOCHS_LIST, LR_LIST, DROPOUT_LIST))
    total = len(combinaciones)
    print(f"Total de combinaciones: {total}\n")

    resultados = []

    for i, (epochs, lr, dropout) in enumerate(combinaciones, 1):
        nombre_run = f"yolov8s_ep{epochs}_lr{lr}_drop{dropout}"
        print(f"\n[{i}/{total}] {nombre_run}")

        model = YOLO(MODEL_WEIGHTS)

        model.train(
            data=DATA_DIR,
            epochs=epochs,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            lr0=lr,
            dropout=dropout,
            project=PROJECT,
            name=nombre_run,
            exist_ok=EXIST_OK
        )

# Obtener metricas de validacion
        metrics = model.val(
            data=DATA_DIR,
            split="test"
        )

        resultado = {
            "nombre":   nombre_run,
            "epochs":   epochs,
            "lr":       lr,
            "dropout":  dropout,
            "accuracy": round(metrics.top1, 4),
        }
        resultados.append(resultado)
        print(f"  Accuracy test: {metrics.top1:.4f}")

# Guardar resultados parciales
        pd.DataFrame(resultados).to_csv(
            os.path.join(PROJECT, "resultados_grilla_yolov8s.csv"), index=False
        )

# Resumen final
    df = pd.DataFrame(resultados).sort_values("accuracy", ascending=False)
    print(f"\n{'='*50}")
    print("TOP 5 MEJORES CONFIGURACIONES YOLOV8s:")
    print('='*50)
    print(df.head(5).to_string(index=False))
    print(f"\nResultados guardados en: {PROJECT}/resultados_grilla_yolov8s.csv")
