import os
import numpy as np
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Configuracion
BASE_DIR    = os.getenv('BASE_DIR', str(Path(__file__).resolve().parent.parent))
NPZ_DIR     = os.getenv('NPZ_DIR',  os.path.join(BASE_DIR, 'NPZ'))
NPZ_FILE    = os.getenv('NPZ_FILE', os.path.join(NPZ_DIR, 'DB.npz'))
IMAGE_SIZE  = int(os.getenv('IMAGE_SIZE', 256))
SEED        = int(os.getenv('RANDOM_SEED', 0))
SPLIT_TRAIN = float(os.getenv('SPLIT_TRAIN', 0.70))
SPLIT_TEST  = float(os.getenv('SPLIT_TEST',  0.15))
SPLIT_VAL   = float(os.getenv('SPLIT_VAL',   0.15))

np.random.seed(SEED)

# Directorio raiz del dataset
# Estructura esperada:
#   dataset/
#     clase_0/  img1.jpg  img2.jpg ...
#     clase_1/  ...
#     clase_2/  ...
#     clase_3/  ...
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')


def cargar_clase(ruta_clase, image_size):
    imagenes = []
    extensiones = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    for archivo in sorted(Path(ruta_clase).iterdir()):
        if archivo.suffix.lower() not in extensiones:
            continue
        try:
            img = Image.open(archivo).convert('RGB')
            img = img.resize((image_size, image_size), Image.LANCZOS)
            imagenes.append(np.array(img, dtype=np.uint8))
        except Exception as e:
            print(f'Error leyendo {archivo}: {e}')
    return imagenes


def balancear(clases_data):
    # Menor cantidad de muestras entre todas las clases
    minimo = min(len(c) for c in clases_data)
    print(f'Muestras por clase tras balanceo: {minimo}')
    balanceadas = []
    for c in clases_data:
        idx = np.random.permutation(len(c))[:minimo]
        balanceadas.append([c[i] for i in idx])
    return balanceadas, minimo


def split(imagenes, etiqueta, n_train, n_test, n_val):
    arr = np.array(imagenes, dtype=np.uint8)
    idx = np.random.permutation(len(arr))
    arr = arr[idx]

    x_train = arr[:n_train]
    x_test  = arr[n_train:n_train + n_test]
    x_val   = arr[n_train + n_test:n_train + n_test + n_val]

    y_train = np.full(len(x_train), etiqueta, dtype=np.int32)
    y_test  = np.full(len(x_test),  etiqueta, dtype=np.int32)
    y_val   = np.full(len(x_val),   etiqueta, dtype=np.int32)

    return x_train, y_train, x_test, y_test, x_val, y_val


def construir_db():
    os.makedirs(NPZ_DIR, exist_ok=True)

    # Leer clases (carpetas ordenadas = etiquetas 0, 1, 2, ...)
    clases_dirs = sorted([
        d for d in Path(DATASET_DIR).iterdir()
        if d.is_dir()
    ])
    print(f'Clases encontradas ({len(clases_dirs)}):')
    for i, d in enumerate(clases_dirs):
        print(f'  {i}: {d.name}')

    # Carga
    clases_data = []
    for d in clases_dirs:
        imgs = cargar_clase(d, IMAGE_SIZE)
        print(f'  {d.name}: {len(imgs)} imagenes')
        clases_data.append(imgs)

    # Balanceo
    clases_data, minimo = balancear(clases_data)

    # Calcular splits
    n_train = int(minimo * SPLIT_TRAIN)
    n_test  = int(minimo * SPLIT_TEST)
    n_val   = minimo - n_train - n_test

    print(f'\nSplit por clase -> train: {n_train}  test: {n_test}  val: {n_val}')

    # Construir arrays globales
    X_train, Y_train = [], []
    X_test,  Y_test  = [], []
    X_val,   Y_val   = [], []

    for etiqueta, imgs in enumerate(clases_data):
        xt, yt, xte, yte, xv, yv = split(imgs, etiqueta, n_train, n_test, n_val)
        X_train.append(xt);  Y_train.append(yt)
        X_test.append(xte);  Y_test.append(yte)
        X_val.append(xv);    Y_val.append(yv)

    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)
    X_test  = np.concatenate(X_test)
    Y_test  = np.concatenate(Y_test)
    X_val   = np.concatenate(X_val)
    Y_val   = np.concatenate(Y_val)

    # Barajar conjuntos
    for X, Y in [(X_train, Y_train), (X_test, Y_test), (X_val, Y_val)]:
        idx = np.random.permutation(len(X))
        X[:] = X[idx]
        Y[:] = Y[idx]

    print(f'\nShapes finales:')
    print(f'  x_train: {X_train.shape}  y_train: {Y_train.shape}')
    print(f'  x_test:  {X_test.shape}   y_test:  {Y_test.shape}')
    print(f'  x_val:   {X_val.shape}    y_val:   {Y_val.shape}')

    # Guardar NPZ
    np.savez_compressed(
        NPZ_FILE,
        x_train=X_train,
        y_train=Y_train,
        x_test=X_test,
        y_test=Y_test,
        x_val=X_val,
        y_val=Y_val,
    )
    print(f'\nGuardado en: {NPZ_FILE}')


if __name__ == '__main__':
    construir_db()
