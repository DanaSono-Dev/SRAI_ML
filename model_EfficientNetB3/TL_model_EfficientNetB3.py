import os
import ast
import time
from dotenv import load_dotenv

load_dotenv()  # Carga las variables desde .env
import itertools
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# ============================================================
# VARIABLES DE ENTORNO
# ============================================================
def _parse_list(raw, cast=str):
    """Parsea 'a,b,c' o '["a","b"]' como lista."""
    raw = raw.strip()
    if raw.startswith("["):
        return ast.literal_eval(raw)
    return [cast(x.strip()) for x in raw.split(",")]

def _parse_nested_list(raw):
    """Parsea '(512,256),(256,128)' o '[[512,256],[256,128]]' como lista de tuplas."""
    raw = raw.strip()
    if raw.startswith("["):
        parsed = ast.literal_eval(raw)
        return [tuple(t) for t in parsed]
    # formato: (512,256),(256,128),(512,256,128)
    import re
    grupos = re.findall(r'\([\d,\s]+\)', raw)
    return [tuple(int(x) for x in g.strip("()").split(",")) for g in grupos]

# Rutas
DATA_DIR = os.environ.get("DATA_DIR",
    "C:/Users/Dan/Documents/DAN/Proyecto Dana/Repositorio/ML/Parametros")
PROJECT  = os.environ.get("PROJECT",
    "C:/Users/Dan/Documents/DAN/Proyecto Dana/Repositorio/ML/model_EfficientNetB3")

# Parámetros fijos
IMGSZ  = int(os.environ.get("IMGSZ",  "224"))
BATCH  = int(os.environ.get("BATCH",  "32"))
CLASES = int(os.environ.get("CLASES", "4"))

# Grilla de hiperparámetros
ARQUITECTURAS = _parse_list(os.environ.get("ARQUITECTURAS", "efficientnet_b3"))
NEURONAS_LIST = _parse_nested_list(
    os.environ.get("NEURONAS_LIST", "(512,256),(256,128),(512,256,128)"))
DROPOUT_LIST  = _parse_list(os.environ.get("DROPOUT_LIST",  "0.3,0.5"),  float)
LR_LIST       = _parse_list(os.environ.get("LR_LIST",       "0.01,0.001"), float)
EPOCHS_LIST   = _parse_list(os.environ.get("EPOCHS_LIST",   "50,100"),   int)
ACTIVACIONES  = _parse_list(os.environ.get("ACTIVACIONES",  "relu,gelu"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODELO
# ============================================================
def crear_modelo(arquitectura, neuronas, dropout, activacion, num_clases=CLASES):
    backbone = timm.create_model(arquitectura, pretrained=True)
    num_caracteristicas = backbone.classifier.in_features
    backbone.classifier = nn.Identity()

    act_fn = nn.ReLU if activacion == "relu" else nn.GELU

    capas = []
    entrada = num_caracteristicas
    for n in neuronas:
        capas.append(nn.Linear(entrada, n))
        capas.append(act_fn())
        capas.append(nn.Dropout(dropout))
        entrada = n

    capas.append(nn.Linear(entrada, num_clases))
    capas.append(nn.Softmax(dim=1))

    class ModeloCompleto(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.cabeza = nn.Sequential(*capas)

        def forward(self, x):
            x = self.backbone(x)
            x = self.cabeza(x)
            return x

    return ModeloCompleto()


# ============================================================
# DATOS
# ============================================================
def cargar_datos():
    transform = transforms.Compose([
        transforms.Resize((IMGSZ, IMGSZ)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", transform=transform)
    val_dataset   = datasets.ImageFolder(f"{DATA_DIR}/val",   transform=transform)
    test_dataset  = datasets.ImageFolder(f"{DATA_DIR}/test",  transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH, shuffle=False, num_workers=4)

    clases = train_dataset.classes
    print(f"Clases detectadas: {clases}")
    return train_loader, val_loader, test_loader, clases


# ============================================================
# ENTRENAMIENTO
# ============================================================
def entrenar(modelo, train_loader, val_loader, epochs, lr, nombre_run):
    modelo = modelo.to(DEVICE)
    criterio    = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)

    mejor_accuracy = 0.0
    ruta_mejor = os.path.join(PROJECT, nombre_run, "best.pt")
    os.makedirs(os.path.join(PROJECT, nombre_run), exist_ok=True)

    for epoch in range(1, epochs + 1):
        modelo.train()
        loss_train, correctas_train, total_train = 0.0, 0, 0
        for imagenes, etiquetas in train_loader:
            imagenes, etiquetas = imagenes.to(DEVICE), etiquetas.to(DEVICE)
            optimizador.zero_grad()
            salidas = modelo(imagenes)
            loss = criterio(salidas, etiquetas)
            loss.backward()
            optimizador.step()
            loss_train += loss.item()
            _, preds = torch.max(salidas, 1)
            correctas_train += (preds == etiquetas).sum().item()
            total_train += etiquetas.size(0)

        modelo.eval()
        loss_val, correctas_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for imagenes, etiquetas in val_loader:
                imagenes, etiquetas = imagenes.to(DEVICE), etiquetas.to(DEVICE)
                salidas = modelo(imagenes)
                loss = criterio(salidas, etiquetas)
                loss_val += loss.item()
                _, preds = torch.max(salidas, 1)
                correctas_val += (preds == etiquetas).sum().item()
                total_val += etiquetas.size(0)

        acc_val = correctas_val / total_val
        print(f"  Época {epoch}/{epochs} | "
              f"Loss train: {loss_train/len(train_loader):.4f} | "
              f"Acc train: {correctas_train/total_train:.4f} | "
              f"Loss val: {loss_val/len(val_loader):.4f} | "
              f"Acc val: {acc_val:.4f}")

        if acc_val > mejor_accuracy:
            mejor_accuracy = acc_val
            torch.save(modelo.state_dict(), ruta_mejor)
            print(f"  ✓ Mejor modelo guardado (acc_val: {acc_val:.4f})")

    return ruta_mejor


# ============================================================
# EVALUACIÓN
# ============================================================
def evaluar(modelo, test_loader, clases, nombre_run):
    modelo.eval()
    todas_etiquetas, todas_predicciones, tiempos = [], [], []

    with torch.no_grad():
        for imagenes, etiquetas in test_loader:
            imagenes = imagenes.to(DEVICE)
            inicio = time.time()
            salidas = modelo(imagenes)
            tiempos.append((time.time() - inicio) / imagenes.size(0))
            _, preds = torch.max(salidas, 1)
            todas_etiquetas.extend(etiquetas.numpy())
            todas_predicciones.extend(preds.cpu().numpy())

    accuracy  = accuracy_score(todas_etiquetas, todas_predicciones)
    precision = precision_score(todas_etiquetas, todas_predicciones, average='weighted')
    recall    = recall_score(todas_etiquetas, todas_predicciones, average='weighted')
    f1        = f1_score(todas_etiquetas, todas_predicciones, average='weighted')
    tiempo_ms = np.mean(tiempos) * 1000

    ruta_modelo = os.path.join(PROJECT, nombre_run, "best.pt")
    tamanio_mb  = os.path.getsize(ruta_modelo) / (1024 * 1024)

    cm = confusion_matrix(todas_etiquetas, todas_predicciones)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=clases, yticklabels=clases)
    plt.title(f'Matriz de Confusión - {nombre_run}')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.savefig(os.path.join(PROJECT, nombre_run, "confusion_matrix.png"))
    plt.close()

    print(f"\n{'='*50}")
    print(f"Resultados: {nombre_run}")
    print(f"{'='*50}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1-score:          {f1:.4f}")
    print(f"Tiempo inferencia: {tiempo_ms:.2f} ms/imagen")
    print(f"Tamaño modelo:     {tamanio_mb:.2f} MB")

    return {
        "nombre": nombre_run, "accuracy": accuracy,
        "precision": precision, "recall": recall,
        "f1": f1, "tiempo_ms": tiempo_ms, "tamanio_mb": tamanio_mb
    }


# ============================================================
# MAIN - GRILLA
# ============================================================
if __name__ == '__main__':
    print("Configuración activa:")
    print(f"  DATA_DIR      = {DATA_DIR}")
    print(f"  PROJECT       = {PROJECT}")
    print(f"  IMGSZ         = {IMGSZ}")
    print(f"  BATCH         = {BATCH}")
    print(f"  CLASES        = {CLASES}")
    print(f"  ARQUITECTURAS = {ARQUITECTURAS}")
    print(f"  NEURONAS_LIST = {NEURONAS_LIST}")
    print(f"  DROPOUT_LIST  = {DROPOUT_LIST}")
    print(f"  LR_LIST       = {LR_LIST}")
    print(f"  EPOCHS_LIST   = {EPOCHS_LIST}")
    print(f"  ACTIVACIONES  = {ACTIVACIONES}")
    print(f"  DEVICE        = {DEVICE}\n")

    os.makedirs(PROJECT, exist_ok=True)
    train_loader, val_loader, test_loader, clases = cargar_datos()

    combinaciones = list(itertools.product(
        ARQUITECTURAS, NEURONAS_LIST, DROPOUT_LIST, LR_LIST, EPOCHS_LIST, ACTIVACIONES
    ))
    total = len(combinaciones)
    print(f"Total de combinaciones: {total}\n")

    resultados = []

    for i, (arq, neuronas, dropout, lr, epochs, activacion) in enumerate(combinaciones, 1):
        n_str = "x".join(str(n) for n in neuronas)
        nombre_run = f"{arq}_n{n_str}_drop{dropout}_lr{lr}_ep{epochs}_act{activacion}"
        print(f"\n[{i}/{total}] {nombre_run}")

        modelo = crear_modelo(arq, neuronas, dropout, activacion)
        ruta_mejor = entrenar(modelo, train_loader, val_loader, epochs, lr, nombre_run)

        modelo.load_state_dict(torch.load(ruta_mejor))
        resultado = evaluar(modelo, test_loader, clases, nombre_run)
        resultados.append(resultado)

        pd.DataFrame(resultados).to_csv(os.path.join(PROJECT, "resultados_grilla.csv"), index=False)

    df = pd.DataFrame(resultados).sort_values("f1", ascending=False)
    print(f"\n{'='*50}")
    print("TOP 5 MEJORES CONFIGURACIONES:")
    print('='*50)
    print(df.head(5).to_string(index=False))
    df.to_csv(os.path.join(PROJECT, "resultados_grilla.csv"), index=False)
    print(f"\nResultados guardados en: {PROJECT}/resultados_grilla.csv")