import os
import json
import time
from pathlib import Path
from itertools import product

from dotenv import load_dotenv
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


# Carga de datos

def cargar_datos(data_dir, seed, batch_size, use_pin_memory, img_size,
                 normalize_mean, normalize_std):
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    transform_val_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    train_dataset = datasets.ImageFolder(root=data_dir / "train",
                                         transform=transform_train)
    val_dataset   = datasets.ImageFolder(root=data_dir / "val",
                                         transform=transform_val_test)
    test_dataset  = datasets.ImageFolder(root=data_dir / "test",
                                         transform=transform_val_test)

    generador = torch.Generator()
    generador.manual_seed(seed)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=use_pin_memory,
                              generator=generador)
    val_loader   = DataLoader(val_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=use_pin_memory)
    test_loader  = DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=use_pin_memory)

    print(f"  Train : {len(train_dataset):,} imagenes  |  "
          f"Val : {len(val_dataset):,}  |  "
          f"Test : {len(test_dataset):,}")

    return train_loader, val_loader, test_loader


# Arquitectura

class RedTomate(nn.Module):
    def __init__(self, capas_densas, dropout, activacion, num_clases):
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
        capas.append(nn.Linear(entrada, num_clases))
        self.clasificador = nn.Sequential(*capas)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.clasificador(x)
        return x


def crear_modelo(capas_densas, dropout, activacion, seed, num_clases, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    modelo = RedTomate(capas_densas, dropout, activacion, num_clases).to(device)
    total_params = sum(p.numel() for p in modelo.parameters())
    print(f"  Parametros totales : {total_params:,}")
    return modelo


# Entrenamiento con checkpoint

def entrenar_modelo(modelo, train_loader, val_loader, epochs, lr,
                    nombre_experimento, models_dir, device,
                    scheduler_patience, scheduler_factor):
    criterio    = nn.CrossEntropyLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=lr)
    scheduler   = optim.lr_scheduler.ReduceLROnPlateau(
        optimizador, mode="min", patience=scheduler_patience, factor=scheduler_factor
    )

    historial = {"train_loss": [], "train_acc": [],
                 "val_loss":   [], "val_acc":   []}

    mejor_val_loss    = float("inf")
    ruta_mejor_modelo = models_dir / f"{nombre_experimento}_mejor.pt"
    ruta_checkpoint   = models_dir / f"{nombre_experimento}_ckpt.pt"
    epoca_inicio      = 1

    # Reanudacion desde checkpoint
    if ruta_checkpoint.exists():
        print(f"  Checkpoint encontrado, reanudando...")
        ckpt = torch.load(ruta_checkpoint, map_location=device)
        modelo.load_state_dict(ckpt["modelo"])
        optimizador.load_state_dict(ckpt["optimizador"])
        scheduler.load_state_dict(ckpt["scheduler"])
        historial      = ckpt["historial"]
        mejor_val_loss = ckpt["mejor_val_loss"]
        epoca_inicio   = ckpt["epoca"] + 1
        print(f"  Reanudando desde epoca {epoca_inicio}/{epochs}")

    total_imgs = len(train_loader.dataset)
    n_batches  = len(train_loader)

    for epoca in range(epoca_inicio, epochs + 1):

        modelo.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        t0 = time.time()

        for batch_idx, (imagenes, etiquetas) in enumerate(train_loader, 1):
            imagenes  = imagenes.to(device)
            etiquetas = etiquetas.to(device)

            optimizador.zero_grad()
            predicciones = modelo(imagenes)
            loss         = criterio(predicciones, etiquetas)
            loss.backward()
            optimizador.step()

            train_loss    += loss.item()
            _, predicted   = predicciones.max(1)
            train_total   += etiquetas.size(0)
            train_correct += predicted.eq(etiquetas).sum().item()

            loss_parcial = train_loss / batch_idx
            acc_parcial  = train_correct / train_total
            imgs_vistas  = min(batch_idx * imagenes.size(0), total_imgs)
            mem_str      = f"  {torch.cuda.memory_reserved() / 1e9:.2f}G" if device.type == "cuda" else ""

            print(f"  {epoca}/{epochs}{mem_str}  "
                  f"loss: {loss_parcial:.4f}  acc: {acc_parcial:.4f}  "
                  f"{imgs_vistas}/{total_imgs}",
                  end="\r", flush=True)

        print(" " * 80, end="\r")
        train_loss_avg = train_loss / n_batches
        train_acc_avg  = train_correct / train_total

        modelo.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for imagenes, etiquetas in val_loader:
                imagenes  = imagenes.to(device)
                etiquetas = etiquetas.to(device)

                predicciones = modelo(imagenes)
                loss         = criterio(predicciones, etiquetas)

                val_loss    += loss.item()
                _, predicted = predicciones.max(1)
                val_total   += etiquetas.size(0)
                val_correct += predicted.eq(etiquetas).sum().item()

        val_loss_avg = val_loss / len(val_loader)
        val_acc_avg  = val_correct / val_total
        t_epoca      = time.time() - t0

        scheduler.step(val_loss_avg)

        if val_loss_avg < mejor_val_loss:
            mejor_val_loss = val_loss_avg
            torch.save(modelo.state_dict(), ruta_mejor_modelo)
            marca = "  *"
        else:
            marca = ""

        historial["train_loss"].append(train_loss_avg)
        historial["train_acc"].append(train_acc_avg)
        historial["val_loss"].append(val_loss_avg)
        historial["val_acc"].append(val_acc_avg)

        # Checkpoint al final de cada epoca
        torch.save({
            "epoca":          epoca,
            "modelo":         modelo.state_dict(),
            "optimizador":    optimizador.state_dict(),
            "scheduler":      scheduler.state_dict(),
            "historial":      historial,
            "mejor_val_loss": mejor_val_loss,
        }, ruta_checkpoint)

        print(f"  Epoca {epoca:3d}/{epochs}  "
              f"loss: {train_loss_avg:.4f}  acc: {train_acc_avg:.4f}  "
              f"val_loss: {val_loss_avg:.4f}  val_acc: {val_acc_avg:.4f}  "
              f"{t_epoca:.1f}s{marca}")

    # Checkpoint eliminado al completar el experimento
    if ruta_checkpoint.exists():
        ruta_checkpoint.unlink()

    print(f"  Mejor val_loss: {mejor_val_loss:.4f}")
    return historial, ruta_mejor_modelo


# Graficas

def graficar_historial(historial, nombre_experimento, results_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(historial["train_loss"], label="Train Loss")
    ax1.plot(historial["val_loss"],   label="Val Loss")
    ax1.set_title("Loss por epoca")
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(historial["train_acc"], label="Train Acc")
    ax2.plot(historial["val_acc"],   label="Val Acc")
    ax2.set_title("Accuracy por epoca")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(nombre_experimento)
    plt.tight_layout()
    plt.savefig(results_dir / f"{nombre_experimento}_curvas.png", dpi=150)
    plt.close()


# Evaluacion

def evaluar_modelo(modelo, test_loader, ruta_modelo, nombre_experimento,
                   clases, results_dir, device):
    modelo.load_state_dict(torch.load(ruta_modelo, map_location=device))
    modelo.eval()

    todas_predicciones = []
    todas_etiquetas    = []
    tiempos_inferencia = []

    with torch.no_grad():
        for imagenes, etiquetas in test_loader:
            imagenes  = imagenes.to(device)
            etiquetas = etiquetas.to(device)

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
    precision  = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall     = recall_score(y_true, y_pred,    average="weighted", zero_division=0)
    f1         = f1_score(y_true, y_pred,         average="weighted", zero_division=0)
    tiempo_ms  = np.mean(tiempos_inferencia) * 1000
    tamanio_mb = os.path.getsize(ruta_modelo) / (1024 * 1024)

    print(classification_report(y_true, y_pred, target_names=clases))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=clases, yticklabels=clases)
    plt.title(f"Matriz de Confusion\n{nombre_experimento}")
    plt.ylabel("Etiqueta Real")
    plt.xlabel("Prediccion")
    plt.tight_layout()
    plt.savefig(results_dir / f"{nombre_experimento}_confusion.png", dpi=150)
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


# Exportar a ONNX

def exportar_a_onnx(modelo, ruta_modelo_pt, nombre_experimento,
                    export_dir, clases, img_size, opset_version):
    modelo.load_state_dict(torch.load(ruta_modelo_pt, map_location="cpu"))
    modelo.eval()
    modelo.to("cpu")

    imagen_dummy = torch.randn(1, 3, img_size, img_size)
    ruta_onnx    = export_dir / f"{nombre_experimento}.onnx"

    torch.onnx.export(
        modelo,
        imagen_dummy,
        ruta_onnx,
        export_params=True,
        opset_version=opset_version,
        dynamo=False,
        input_names=["imagen"],
        output_names=["logits"],
        dynamic_axes={
            "imagen":  {0: "batch_size"},
            "logits":  {0: "batch_size"}
        }
    )

    tamanio_onnx = os.path.getsize(ruta_onnx) / (1024 * 1024)
    print(f"  ONNX guardado : {ruta_onnx}  ({tamanio_onnx:.2f} MB)")

    ruta_clases = export_dir / "clases.json"
    with open(ruta_clases, "w") as f:
        json.dump({"clases": clases, "img_size": img_size}, f, indent=2)
    print(f"  Clases        : {ruta_clases}")

    return ruta_onnx


# Grilla de experimentos con reanudacion

def correr_grilla(cfg):
    combinaciones = list(product(
        cfg["capas_opciones"],
        cfg["dropout_opciones"],
        cfg["lr_opciones"],
        cfg["epocas_opciones"],
        cfg["activacion_opciones"]
    ))

    total    = len(combinaciones)
    semillas = [cfg["seed_base"] * (i + 1) for i in range(total)]

    # Resultados previos
    ruta_csv_parcial = cfg["results_dir"] / "resultados_grilla.csv"

    if ruta_csv_parcial.exists():
        df_previo         = pd.read_csv(ruta_csv_parcial)
        nombres_completos = set(df_previo["nombre"].tolist())
        resultados_grilla = df_previo.to_dict("records")
        print(f"  Reanudando grilla: {len(nombres_completos)}/{total} experimentos previos")
    else:
        nombres_completos = set()
        resultados_grilla = []

    mejor_f1_global = max((r["f1"] for r in resultados_grilla), default=0.0)
    mejor_info      = None

    print(f"\nTotal de combinaciones: {total}\n")

    for i, (capas, dropout, lr, epochs, activacion) in enumerate(combinaciones):

        capas_str = "-".join(str(c) for c in capas)
        nombre    = f"ep{epochs}_lr{lr}_drop{dropout}_{activacion}_{capas_str}"
        seed      = semillas[i]

        # Saltar experimentos ya completados
        if nombre in nombres_completos:
            print(f"  [{i+1}/{total}] Saltando {nombre}")
            continue

        print(f"\n[{i+1}/{total}] {nombre}")

        train_loader, val_loader, test_loader = cargar_datos(
            cfg["data_dir"], seed, cfg["batch_size"], cfg["use_pin_memory"],
            cfg["img_size"], cfg["normalize_mean"], cfg["normalize_std"]
        )
        modelo = crear_modelo(
            capas, dropout, activacion, seed, cfg["num_clases"], cfg["device"]
        )
        historial, ruta_modelo = entrenar_modelo(
            modelo, train_loader, val_loader, epochs, lr,
            nombre, cfg["models_dir"], cfg["device"],
            cfg["scheduler_patience"], cfg["scheduler_factor"]
        )
        metricas = evaluar_modelo(
            modelo, test_loader, ruta_modelo, nombre,
            cfg["clases"], cfg["results_dir"], cfg["device"]
        )
        graficar_historial(historial, nombre, cfg["results_dir"])

        metricas["epochs"]     = epochs
        metricas["lr"]         = lr
        metricas["dropout"]    = dropout
        metricas["activacion"] = activacion
        metricas["capas"]      = capas_str
        metricas["seed"]       = seed

        resultados_grilla.append(metricas)
        pd.DataFrame(resultados_grilla).to_csv(ruta_csv_parcial, index=False)

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

        print(f"  Accuracy test: {metricas['accuracy']:.4f}  F1: {metricas['f1']:.4f}")

    df = pd.DataFrame(resultados_grilla).sort_values("f1", ascending=False)
    df.to_csv(cfg["results_dir"] / "resultados_grilla_final.csv", index=False)

    print(f"\nTOP 5 MEJORES CONFIGURACIONES:")
    print(df[["nombre", "accuracy", "f1", "tiempo_ms", "tamanio_mb"]].head(5).to_string(index=False))

    mejor = df.iloc[0]
    print(f"\nMejor modelo: {mejor['nombre']}")
    print(f"  Accuracy  : {mejor['accuracy']:.4f}")
    print(f"  F1-Score  : {mejor['f1']:.4f}")
    print(f"  Recall    : {mejor['recall']:.4f}")
    print(f"  Precision : {mejor['precision']:.4f}")
    print(f"  Tiempo    : {mejor['tiempo_ms']:.4f} ms/imagen")
    print(f"  Tamanio   : {mejor['tamanio_mb']:.2f} MB")

    mejor_capas  = [int(n) for n in mejor["capas"].split("-")]
    modelo_final = crear_modelo(
        mejor_capas, float(mejor["dropout"]), mejor["activacion"],
        int(mejor["seed"]), cfg["num_clases"], cfg["device"]
    )

    ruta_onnx = exportar_a_onnx(
        modelo_final,
        cfg["models_dir"] / f"{mejor['nombre']}_mejor.pt",
        mejor["nombre"],
        cfg["export_dir"],
        cfg["clases"],
        cfg["img_size"],
        cfg["opset_version"]
    )

    print(f"\nResultados : {cfg['results_dir']}/resultados_grilla_final.csv")
    print(f"ONNX       : {ruta_onnx}")
    print(f"Clases     : {cfg['export_dir']}/clases.json")

    return df


if __name__ == "__main__":

    load_dotenv(dotenv_path=Path(__file__).parent / ".env")

    # Seleccion de GPU, debe aplicarse antes de inicializar CUDA
    cuda_device_idx = os.getenv("CUDA_DEVICE_INDEX", None)
    if cuda_device_idx is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device_idx

    # Configuracion

    data_dir    = Path(os.getenv("DATA_DIR"))
    results_dir = Path(os.getenv("RESULTS_DIR", "resultados"))
    models_dir  = Path(os.getenv("MODELS_DIR",  "modelos"))
    export_dir  = Path(os.getenv("EXPORT_DIR",  "exportado"))

    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    export_dir.mkdir(exist_ok=True)

    clases     = os.getenv("CLASES").split(",")
    num_clases = len(clases)
    img_size   = int(os.getenv("IMG_SIZE",   224))
    batch_size = int(os.getenv("BATCH_SIZE", 32))

    capas_opciones = [
        [int(n) for n in opcion.split("-")]
        for opcion in os.getenv("CAPAS_OPCIONES").split("|")
    ]

    dropout_opciones    = [float(x) for x in os.getenv("DROPOUT_OPCIONES").split(",")]
    lr_opciones         = [float(x) for x in os.getenv("LR_OPCIONES").split(",")]
    epocas_opciones     = [int(x)   for x in os.getenv("EPOCAS_OPCIONES").split(",")]
    activacion_opciones = os.getenv("ACTIVACION_OPCIONES").split(",")
    seed_base           = int(os.getenv("SEED_BASE", 42))

    normalize_mean     = [float(x) for x in os.getenv("IMG_NORMALIZE_MEAN", "0.485,0.456,0.406").split(",")]
    normalize_std      = [float(x) for x in os.getenv("IMG_NORMALIZE_STD",  "0.229,0.224,0.225").split(",")]
    scheduler_patience = int(os.getenv("LR_SCHEDULER_PATIENCE", 5))
    scheduler_factor   = float(os.getenv("LR_SCHEDULER_FACTOR", 0.5))
    opset_version      = int(os.getenv("ONNX_OPSET_VERSION", 11))

    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = (device.type == "cuda")

    print(f"  Dispositivo  : {device}")
    if device.type == "cuda":
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print(f"  Dataset      : {data_dir}")
    print(f"  Resultados   : {results_dir}")
    print(f"  Modelos      : {models_dir}")
    print(f"  Exportado    : {export_dir}")
    print(f"  Clases ({num_clases})  : {clases}")
    print(f"  IMG_SIZE     : {img_size}")
    print(f"  BATCH_SIZE   : {batch_size}")
    print(f"  Capas        : {capas_opciones}")
    print(f"  Dropout      : {dropout_opciones}")
    print(f"  LR           : {lr_opciones}")
    print(f"  Epocas       : {epocas_opciones}")
    print(f"  Activaciones : {activacion_opciones}")
    print(f"  Semilla base : {seed_base}")
    print(f"  Norm. media  : {normalize_mean}")
    print(f"  Norm. std    : {normalize_std}")
    print(f"  Sched. pat.  : {scheduler_patience}  factor: {scheduler_factor}")
    print(f"  ONNX opset   : {opset_version}")

    cfg = {
        "data_dir":            data_dir,
        "results_dir":         results_dir,
        "models_dir":          models_dir,
        "export_dir":          export_dir,
        "clases":              clases,
        "num_clases":          num_clases,
        "img_size":            img_size,
        "batch_size":          batch_size,
        "capas_opciones":      capas_opciones,
        "dropout_opciones":    dropout_opciones,
        "lr_opciones":         lr_opciones,
        "epocas_opciones":     epocas_opciones,
        "activacion_opciones": activacion_opciones,
        "seed_base":           seed_base,
        "normalize_mean":      normalize_mean,
        "normalize_std":       normalize_std,
        "scheduler_patience":  scheduler_patience,
        "scheduler_factor":    scheduler_factor,
        "opset_version":       opset_version,
        "device":              device,
        "use_pin_memory":      use_pin_memory,
    }

    correr_grilla(cfg)