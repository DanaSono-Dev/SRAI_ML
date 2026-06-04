# IMPORTAR LIBRERÍAS Y CONFIGURACIÓN

import os
import json
import time
import random
from pathlib import Path
from itertools import product

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# LEER VARIABLES DE ENTORNO

DATA_DIR    = Path(os.getenv("DATA_DIR"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "resultados"))
MODELS_DIR  = Path(os.getenv("MODELS_DIR",  "modelos"))
EXPORT_DIR  = Path(os.getenv("EXPORT_DIR",  "exportado"))

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

CLASES     = os.getenv("CLASES").split(",")
NUM_CLASES = len(CLASES)

IMG_SIZE   = int(os.getenv("IMG_SIZE",   224))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))

CAPAS_OPCIONES = [
    [int(n) for n in opcion.split("-")]
    for opcion in os.getenv("CAPAS_OPCIONES").split("|")
]

DROPOUT_OPCIONES    = [float(x) for x in os.getenv("DROPOUT_OPCIONES").split(",")]
LR_OPCIONES         = [float(x) for x in os.getenv("LR_OPCIONES").split(",")]
EPOCAS_OPCIONES     = [int(x)   for x in os.getenv("EPOCAS_OPCIONES").split(",")]
ACTIVACION_OPCIONES = os.getenv("ACTIVACION_OPCIONES").split(",")

SEED_BASE = int(os.getenv("SEED_BASE", 42))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("  RED NEURONAL — ENFERMEDADES FOLIARES EN TOMATE")
print("=" * 60)
print(f"  Dispositivo  : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU          : {torch.cuda.get_device_name(0)}")
print(f"  Dataset      : {DATA_DIR}")
print(f"  Resultados   : {RESULTS_DIR}")
print(f"  Modelos      : {MODELS_DIR}")
print(f"  Exportado    : {EXPORT_DIR}")
print(f"  Clases ({NUM_CLASES})  : {CLASES}")
print(f"  IMG_SIZE     : {IMG_SIZE}")
print(f"  BATCH_SIZE   : {BATCH_SIZE}")
print(f"  Capas        : {CAPAS_OPCIONES}")
print(f"  Dropout      : {DROPOUT_OPCIONES}")
print(f"  LR           : {LR_OPCIONES}")
print(f"  Épocas       : {EPOCAS_OPCIONES}")
print(f"  Activaciones : {ACTIVACION_OPCIONES}")
print(f"  Semilla base : {SEED_BASE}")
print("=" * 60)



# CARGAR DATOS + SHUFFLE


transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def cargar_datos(data_dir, seed):
    """
    Carga train, val y test desde las carpetas del dataset.
    Cada experimento recibe su propia semilla para el shuffle.
    """
    train_dataset = datasets.ImageFolder(root=data_dir / "train",
                                         transform=transform_train)
    val_dataset   = datasets.ImageFolder(root=data_dir / "val",
                                         transform=transform_val_test)
    test_dataset  = datasets.ImageFolder(root=data_dir / "test",
                                         transform=transform_val_test)

    generador = torch.Generator()
    generador.manual_seed(seed)

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              generator=generador)
    val_loader   = DataLoader(val_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=4,
                              pin_memory=True)
    test_loader  = DataLoader(test_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=4,
                              pin_memory=True)

    print(f"  Train : {len(train_dataset):,} imágenes")
    print(f"  Val   : {len(val_dataset):,} imágenes")
    print(f"  Test  : {len(test_dataset):,} imágenes")

    return train_loader, val_loader, test_loader



# ARQUITECTURA

class RedTomate(nn.Module):
    def __init__(self, capas_densas, dropout, activacion):
        """
        capas_densas : lista de neuronas, ej: [512,256] o [256,128] o [512,256,128]
        dropout      : valor entre 0 y 1
        activacion   : "ReLU" o "GELU"
        """
        super(RedTomate, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU() if activacion == "ReLU" else nn.GELU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU() if activacion == "ReLU" else nn.GELU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU() if activacion == "ReLU" else nn.GELU(),
            nn.MaxPool2d(2, 2)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        capas = []
        entrada = 128

        for neuronas in capas_densas:
            capas.append(nn.Linear(entrada, neuronas))
            capas.append(nn.ReLU() if activacion == "ReLU" else nn.GELU())
            capas.append(nn.Dropout(dropout))
            entrada = neuronas

        capas.append(nn.Linear(entrada, NUM_CLASES))
        self.clasificador = nn.Sequential(*capas)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.clasificador(x)
        x = self.softmax(x)
        return x


def crear_modelo(capas_densas, dropout, activacion, seed):
    """Crea un modelo nuevo con su propia semilla para comparación justa."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    modelo = RedTomate(capas_densas, dropout, activacion).to(DEVICE)

    total_params = sum(p.numel() for p in modelo.parameters())
    print(f"  Parámetros totales: {total_params:,}")

    return modelo



# CICLO DE ENTRENAMIENTO


def entrenar_modelo(modelo, train_loader, val_loader,
                    epochs, lr, nombre_experimento):
    """
    Entrena el modelo época por época.
    Guarda el mejor modelo según val_loss.
    """
    criterio    = nn.CrossEntropyLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=lr)
    scheduler   = optim.lr_scheduler.ReduceLROnPlateau(
        optimizador, mode="min", patience=5, factor=0.5, verbose=False
    )

    historial = {"train_loss": [], "train_acc": [],
                 "val_loss":   [], "val_acc":   []}

    mejor_val_loss    = float("inf")
    ruta_mejor_modelo = MODELS_DIR / f"{nombre_experimento}_mejor.pt"

    for epoca in range(1, epochs + 1):

        modelo.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imagenes, etiquetas in train_loader:
            imagenes  = imagenes.to(DEVICE)
            etiquetas = etiquetas.to(DEVICE)

            optimizador.zero_grad()
            predicciones = modelo(imagenes)
            loss         = criterio(predicciones, etiquetas)
            loss.backward()
            optimizador.step()

            train_loss    += loss.item()
            _, predicted   = predicciones.max(1)
            train_total   += etiquetas.size(0)
            train_correct += predicted.eq(etiquetas).sum().item()

        train_loss_avg = train_loss / len(train_loader)
        train_acc_avg  = train_correct / train_total

        modelo.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for imagenes, etiquetas in val_loader:
                imagenes  = imagenes.to(DEVICE)
                etiquetas = etiquetas.to(DEVICE)

                predicciones = modelo(imagenes)
                loss         = criterio(predicciones, etiquetas)

                val_loss    += loss.item()
                _, predicted = predicciones.max(1)
                val_total   += etiquetas.size(0)
                val_correct += predicted.eq(etiquetas).sum().item()

        val_loss_avg = val_loss / len(val_loader)
        val_acc_avg  = val_correct / val_total

        scheduler.step(val_loss_avg)

        if val_loss_avg < mejor_val_loss:
            mejor_val_loss = val_loss_avg
            torch.save(modelo.state_dict(), ruta_mejor_modelo)
            marca = "✓"
        else:
            marca = ""

        historial["train_loss"].append(train_loss_avg)
        historial["train_acc"].append(train_acc_avg)
        historial["val_loss"].append(val_loss_avg)
        historial["val_acc"].append(val_acc_avg)

        print(f"  Época {epoca:3d}/{epochs} | "
              f"train_loss: {train_loss_avg:.4f} | "
              f"train_acc: {train_acc_avg:.4f} | "
              f"val_loss: {val_loss_avg:.4f} | "
              f"val_acc: {val_acc_avg:.4f} {marca}")

    print(f"\n  Mejor val_loss: {mejor_val_loss:.4f}")
    return historial, ruta_mejor_modelo



# EVALUACIÓN Y MÉTRICAS


def graficar_historial(historial, nombre_experimento):
    """Guarda las curvas de loss y accuracy por época."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(historial["train_loss"], label="Train Loss")
    ax1.plot(historial["val_loss"],   label="Val Loss")
    ax1.set_title("Loss por época")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(historial["train_acc"], label="Train Acc")
    ax2.plot(historial["val_acc"],   label="Val Acc")
    ax2.set_title("Accuracy por época")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(nombre_experimento)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{nombre_experimento}_curvas.png", dpi=150)
    plt.close()


def evaluar_modelo(modelo, test_loader, ruta_modelo, nombre_experimento):
    """
    Carga el mejor modelo y lo evalúa con el test set.
    Genera métricas completas y matriz de confusión.
    """
    modelo.load_state_dict(torch.load(ruta_modelo, map_location=DEVICE))
    modelo.eval()

    todas_predicciones = []
    todas_etiquetas    = []
    tiempos_inferencia = []

    with torch.no_grad():
        for imagenes, etiquetas in test_loader:
            imagenes  = imagenes.to(DEVICE)
            etiquetas = etiquetas.to(DEVICE)

            inicio  = time.time()
            salidas = modelo(imagenes)
            fin     = time.time()

            tiempos_inferencia.append((fin - inicio) / len(imagenes))
            _, predicted = salidas.max(1)
            todas_predicciones.extend(predicted.cpu().numpy())
            todas_etiquetas.extend(etiquetas.cpu().numpy())

    y_pred = np.array(todas_predicciones)
    y_true = np.array(todas_etiquetas)

    accuracy   = accuracy_score(y_true, y_pred)
    precision  = precision_score(y_true, y_pred,
                                 average="weighted", zero_division=0)
    recall     = recall_score(y_true, y_pred,
                              average="weighted", zero_division=0)
    f1         = f1_score(y_true, y_pred,
                          average="weighted", zero_division=0)
    tiempo_ms  = np.mean(tiempos_inferencia) * 1000
    tamanio_mb = os.path.getsize(ruta_modelo) / (1024 * 1024)

    print(f"\n{'═'*60}")
    print(f"  RESULTADOS: {nombre_experimento}")
    print(f"{'═'*60}")
    print(f"  Accuracy          : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision         : {precision:.4f}")
    print(f"  Recall            : {recall:.4f}")
    print(f"  F1-Score          : {f1:.4f}")
    print(f"  Tiempo inferencia : {tiempo_ms:.2f} ms/imagen")
    print(f"  Tamaño del modelo : {tamanio_mb:.2f} MB")
    print(f"{'═'*60}")
    print(classification_report(y_true, y_pred, target_names=CLASES))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASES, yticklabels=CLASES)
    plt.title(f"Matriz de Confusión\n{nombre_experimento}")
    plt.ylabel("Etiqueta Real")
    plt.xlabel("Predicción")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{nombre_experimento}_confusion.png", dpi=150)
    plt.close()

    return {
        "nombre":     nombre_experimento,
        "accuracy":   accuracy,
        "precision":  precision,
        "recall":     recall,
        "f1":         f1,
        "tiempo_ms":  tiempo_ms,
        "tamanio_mb": tamanio_mb
    }



# MODELO A ONNX


def exportar_a_onnx(modelo, ruta_modelo_pt, nombre_experimento):
    """
    Exporta el mejor modelo a ONNX optimizado para CPU.
    ONNX permite inferencia 2-3x más rápida en la RPI5 vs PyTorch puro.
    """
    print(f"\n  Exportando a ONNX para Raspberry Pi 5...")

    modelo.load_state_dict(torch.load(ruta_modelo_pt, map_location="cpu"))
    modelo.eval()
    modelo.to("cpu")

    imagen_dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    ruta_onnx    = EXPORT_DIR / f"{nombre_experimento}.onnx"

    torch.onnx.export(
        modelo,
        imagen_dummy,
        ruta_onnx,
        export_params=True,
        opset_version=11,
        input_names=["imagen"],
        output_names=["probabilidades"],
        dynamic_axes={
            "imagen":         {0: "batch_size"},
            "probabilidades": {0: "batch_size"}
        }
    )

    tamanio_onnx = os.path.getsize(ruta_onnx) / (1024 * 1024)
    print(f"  Modelo ONNX guardado : {ruta_onnx}")
    print(f"  Tamaño ONNX          : {tamanio_onnx:.2f} MB")

    ruta_clases = EXPORT_DIR / "clases.json"
    with open(ruta_clases, "w") as f:
        json.dump({"clases": CLASES, "img_size": IMG_SIZE}, f, indent=2)
    print(f"  Clases guardadas en  : {ruta_clases}")

    return ruta_onnx



# GRILLA DE EXPERIMENTOS


def correr_grilla():

    combinaciones = list(product(
        CAPAS_OPCIONES,
        DROPOUT_OPCIONES,
        LR_OPCIONES,
        EPOCAS_OPCIONES,
        ACTIVACION_OPCIONES
    ))

    total    = len(combinaciones)
    semillas = [SEED_BASE * (i + 1) for i in range(total)]

    print(f"\nTotal de experimentos : {total}")
    print(f"Resultados parciales  : {RESULTS_DIR}/resultados_grilla.csv\n")

    resultados_grilla = []
    mejor_f1_global   = 0.0
    mejor_info        = None

    for i, (capas, dropout, lr, epochs, activacion) in enumerate(combinaciones):

        capas_str = "-".join(str(c) for c in capas)
        nombre    = (f"ep{epochs}_lr{lr}_drop{dropout}"
                     f"_{activacion}_{capas_str}")
        seed      = semillas[i]

        print(f"\n{'─'*60}")
        print(f"Experimento {i+1}/{total}: {nombre}")
        print(f"Semilla: {seed}")
        print(f"{'─'*60}")

        train_loader, val_loader, test_loader = cargar_datos(DATA_DIR, seed)
        modelo   = crear_modelo(capas, dropout, activacion, seed)
        historial, ruta_modelo = entrenar_modelo(
            modelo, train_loader, val_loader, epochs, lr, nombre
        )
        metricas = evaluar_modelo(
            modelo, test_loader, ruta_modelo, nombre
        )
        graficar_historial(historial, nombre)

        metricas["epochs"]     = epochs
        metricas["lr"]         = lr
        metricas["dropout"]    = dropout
        metricas["activacion"] = activacion
        metricas["capas"]      = capas_str
        metricas["seed"]       = seed

        resultados_grilla.append(metricas)

        pd.DataFrame(resultados_grilla).to_csv(
            RESULTS_DIR / "resultados_grilla.csv", index=False
        )

        if metricas["f1"] > mejor_f1_global:
            mejor_f1_global = metricas["f1"]
            mejor_info = {
                "nombre":      nombre,
                "capas":       capas,
                "dropout":     dropout,
                "activacion":  activacion,
                "ruta_modelo": ruta_modelo,
                "seed":        seed,
                "metricas":    metricas
            }

        print(f"\n  Resultados parciales guardados ({i+1}/{total})")

    df = pd.DataFrame(resultados_grilla).sort_values("f1", ascending=False)
    df.to_csv(RESULTS_DIR / "resultados_grilla_final.csv", index=False)

    print(f"\n{'═'*60}")
    print(f"  RESUMEN FINAL — TOP 5 MODELOS (ordenados por F1)")
    print(f"{'═'*60}")
    print(df[["nombre", "accuracy", "f1",
              "tiempo_ms", "tamanio_mb"]].head(5).to_string(index=False))

    mejor = df.iloc[0]
    print(f"\n  🏆 MEJOR MODELO: {mejor['nombre']}")
    print(f"     F1-Score  : {mejor['f1']:.4f}")
    print(f"     Accuracy  : {mejor['accuracy']:.4f}")
    print(f"     Recall    : {mejor['recall']:.4f}")
    print(f"     Precision : {mejor['precision']:.4f}")
    print(f"     Tiempo    : {mejor['tiempo_ms']:.2f} ms/imagen (en PC)")
    print(f"     Tamaño    : {mejor['tamanio_mb']:.2f} MB")

    print(f"\n{'═'*60}")
    print(f"  EXPORTANDO MEJOR MODELO PARA RASPBERRY PI 5")
    print(f"{'═'*60}")

    mejor_capas  = [int(n) for n in mejor["capas"].split("-")]
    modelo_final = crear_modelo(
        mejor_capas,
        float(mejor["dropout"]),
        mejor["activacion"],
        int(mejor["seed"])
    )

    ruta_onnx = exportar_a_onnx(
        modelo_final,
        MODELS_DIR / f"{mejor['nombre']}_mejor.pt",
        mejor["nombre"]
    )

    print(f"\n{'═'*60}")
    print(f"  LISTO PARA RASPBERRY PI 5")
    print(f"{'═'*60}")
    print(f"  Archivo ONNX : {ruta_onnx}")
    print(f"  Clases       : {EXPORT_DIR}/clases.json")
    print(f"  Estos archivos se usarán en el Docker de la RPI5.")
    print(f"{'═'*60}")

    return df

# PUNTO DE ENTRADA

if __name__ == "__main__":
    correr_grilla()