import os, time, shutil, csv, random, logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logger = logging.getLogger('tensorflow')
logger.setLevel(logging.ERROR)

import tensorflow as tf

from tensorflow.keras.backend import clear_session
from keras.utils import plot_model, to_categorical

from concurrent.futures import ThreadPoolExecutor # Para guardado asíncrono

from datetime import date
from script.Modelo import Clasificador

# Después de tus imports
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"--- GPU Detectada: {gpus[0].name} ---")
else:
    print("--- Entrenando en CPU ---")
    
# Funcion para crear un archivo ".txt" del summary del modelo cargado
def CreaSummary(modelo, 
                nombre,
                hoy,
                ):
    hoy = str(date.today())+'/'
    if not os.path.exists(hoy):
        os.mkdir(hoy)
    
    dir_model = hoy + 'Model/'
    if not os.path.exists(dir_model):
        os.mkdir(dir_model)
        
    with open(dir_model+'Summary_%s.txt' % nombre, 'w') as f:
        modelo.summary(print_fn=lambda x: f.write(x + '\n'))

    plot_model(modelo,
               dir_model+"Grafo_%s.png" % nombre,
               show_shapes=True,
               show_layer_names=True,
               )

def Split(datos, 
          b_size,
          ):    
    T = len(datos)
    
    batches = np.arange(b_size, 
                        T, 
                        b_size,
                        )    
    datos = np.split(datos, 
                     batches,
                     )
    
    return datos




class TrainingManager:
    """Clase para gestionar carpetas y guardado de archivos sin ensuciar el loop."""
    def __init__(self, 
                 model_name,
                 ):
        self.hoy = f"{hoy}/"
        self.paths = {
            "model": f"{self.hoy}Model/",
            "metrics": f"{self.hoy}pltimagenes/datos/",
            "plots": f"{self.hoy}pltimagenes/"
        }
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
            
    def log_metrics_to_csv(self,
                           epoch,
                           loss,
                           acc, 
                           val_loss,
                           val_acc,
                           filename,
                           ):
        """Escribe una nueva fila en el CSV de métricas."""
        # full_path = os.path.join(self.paths["metrics"], filename)
        # Podemos usar el executor para que la escritura en disco no bloquee el entrenamiento
        def __write():
            with open(filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                # if not os.path.isfile(filename):
                #     writer.writerow(['epoch', 'loss', 'acc', 'val_loss', 'val_acc'])
                    
                writer.writerow([
                    epoch, 
                    f"{loss:.6f}", 
                    f"{acc:.6f}", 
                    f"{val_loss:.6f}", 
                    f"{val_acc:.6f}", 
                ])
        executor.submit(__write)

    def save_plot_async(self, 
                        epochs, 
                        acc, 
                        acc_v, 
                        loss,
                        loss_v, 
                        filename,
                        ):
        """Lanza el guardado de la imagen en un hilo separado."""
        def __save():
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, acc, label='Acc')
            plt.plot(epochs, acc_v, label='Val Acc')
            plt.plot(epochs, loss, label='Loss')
            plt.plot(epochs, loss_v, label='Val Loss')
            plt.ylim(-0.1, 2)
            plt.legend()
            plt.grid(True)
            plt.savefig(filename)
            plt.close()
        executor.submit(__save)

def print_async(epoch, n_epochs, step, total_steps, lr, loss, acc):
    """
    Envía la impresión a un hilo secundario para no bloquear el entrenamiento.
    """
    def __do_print():
        # Construimos el bloque de texto primero para que la salida sea atómica
        mensaje = (
            f"Época: {epoch}/{n_epochs} - Step: {step}/{total_steps} - l_r: {lr:.10f}\n"
            f"Loss: {loss:.6f} - Acc: {acc:.6f}\n"
        )
        print(mensaje) # Usamos print normal aquí ya que va en su propio hilo

    # 'executor' es el ThreadPoolExecutor que definimos al inicio del script
    executor.submit(__do_print)
    
def train(x_train,
          y_train, 
          x_val,
          y_val, 
          model, 
          n_epochs=100,
          batch_size=128,
          ):
    manager = TrainingManager("Clasificador")
    
    if os.path.isfile(f"{manager.paths['metrics']}metricas.csv"):
        os.remove(f"{manager.paths['metrics']}metricas.csv")
        with open(f"{manager.paths['metrics']}metricas.csv", mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'acc', 'val_loss', 'val_acc'])
        
    x_trainB = Split(x_train, batch_size)
    y_trainB = Split(y_train, batch_size)
    
    x_valB = Split(x_val, batch_size*2)
    y_valB = Split(y_val, batch_size*2)
        
    num_batches_train = len(x_trainB)
    num_batches_val = len(x_valB)
    
    metrics = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0

    for epoch in range(n_epochs):
        # Barajar datos al inicio de cada época        
        epoch_loss, epoch_acc = [], []
        
        shuffleBatch = np.arange(num_batches_train)
        np.random.shuffle(shuffleBatch)
        # print(shuffleBatch)
        for step in shuffleBatch:
            # El set de entrenamiento usa el step normal
            x_trainB0 = x_trainB[step]/ 255.0
            y_trainB0 = y_trainB[step]
            # print(idx_val)
            
            # Entrenamiento
            loss, acc = model.train_on_batch(x_trainB0, y_trainB0)
            epoch_loss.append(loss)
            epoch_acc.append(acc)
            
            lr = model.optimizer.learning_rate.numpy()
            # # Impresión limpia (usando \r para actualizar la misma línea)
            # print(f"Época: {epoch+1}/{n_epochs} - Step: {step}/{num_batches_train} - l_r: {lr:.8f}", end='\n')
            # print(f"Loss: {loss:.4f} - Acc: {acc:.4f}", end='\n\n')
            
            print_async(epoch+1, n_epochs, step, num_batches_train, lr, loss, acc)
        
        epoch_val_loss, epoch_val_acc = [], []
        for vali0, vali  in enumerate(x_valB):
            v_loss, v_acc = model.evaluate(vali/255.0, y_valB[vali0], verbose=0)
            epoch_val_loss.append(v_loss)
            epoch_val_acc.append(v_acc)
        
        loss = np.mean(epoch_loss)
        acc = np.mean(epoch_acc)
        v_loss = np.mean(epoch_val_loss)
        v_acc = np.mean(epoch_val_acc)
        print(f"Época: {epoch}/{n_epochs} - Step: {step}/{num_batches_train} - l_r: {lr:.10f}\n")
        print(f"Loss: {np.mean(loss):.6f} - Acc: {np.mean(acc):.6f}\n")
        print(f"Val Loss: {v_loss:.6f} - Val Acc: {v_acc:.6f}", end='\n\n')
        
        # 3. GUARDADO EN CSV (Asíncrono)
        manager.log_metrics_to_csv(
            epoch + 1, 
            loss, 
            acc, 
            v_loss, 
            v_acc,
            f"{manager.paths['metrics']}metricas.csv"
            )
        
        # Guardar historial
        metrics["loss"].append(loss)
        metrics["acc"].append(acc)
        metrics["val_loss"].append(v_loss)
        metrics["val_acc"].append(v_acc)

        # Lógica de guardado del "Mejor Modelo"
        if v_acc > best_acc:
            best_acc = v_acc
            model.save(f"{manager.hoy}model_best.keras")

        # Guardado de gráficas cada 25 épocas (Asíncrono)
        if (epoch + 1) % 25 == 0:
            manager.save_plot_async(
                range(len(metrics["acc"])), 
                metrics["acc"], metrics["val_acc"],
                metrics["loss"], metrics["val_loss"],
                f"{manager.paths['plots']}epoch_{epoch+1}.png"
            )
            
            model.save(f"{manager.hoy}model_{epoch + 1:04d}.keras")
        
        if np.mean(metrics["val_loss"][-20:]) <= 0.002:
            break
        
    print(f"\nEntrenamiento finalizado. Mejor Accuracy: {best_acc:.4f}")

# Configuración de hilos para E/S
executor = ThreadPoolExecutor(max_workers=2)

np.random.seed(0)
inicio = time.time()
hoy = str(date.today())

if not os.path.exists(hoy+'/'):
    os.mkdir(hoy+'/')
     
shutil.copyfile('Train.py', hoy+'/Train.py')
# directory = 'OLD npz/'

#DB = np.load('NPZ/Aff224_DB.npz',
DB = np.load('NPZ/DB.npz',
#DB = np.load('NPZ/RAF224_DB.npz',
             allow_pickle=True)
print(DB.keys())

x_train = DB['x_train']
x_val = DB['x_val']

y_train = DB['y_train']
numCla = len(np.unique(y_train))
y_train = to_categorical(y_train, numCla)
y_val = DB['y_val']
y_val = to_categorical(y_val, numCla)
    
num_epocas = 2000
batch0 = 32
Efdecay = 50
fdecay = Efdecay*(x_train.shape[0]//batch0)

model = Clasificador(x_train[0].shape, fdecay)
#model.load_weights("GMF/pesos_Mayor.weights.h5")
CreaSummary(model, 'Clasificador', hoy)

#Grueso = Model(model.inputs, model.get_layer(name='CoarseOutput').output)
#CreaSummary(Grueso, 'Grueso', hoy)
#Mediano = Model(model.inputs, model.get_layer(name='MediumOutput').output)
#CreaSummary(Mediano, 'Mediano', hoy)
#Fino = Model(model.inputs, model.get_layer(name='FineOutput').output)
#CreaSummary(Fino, 'Fino', hoy)

#del Grueso, Mediano, Fino
clear_session()

# print(x_train[0], y_train[0])
print(len(x_train))
print(fdecay)
train(x_train, y_train,
      x_val, y_val,
      model,
      num_epocas, batch0)
