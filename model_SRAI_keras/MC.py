import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

# Configuración desde variables de entorno
BASE_DIR        = os.getenv('BASE_DIR', str(Path(__file__).resolve().parent))
MODEL_DATE      = os.getenv('MODEL_DATE', '2026-06-22')
MODEL_PATH      = os.getenv('MODEL_PATH', os.path.join(BASE_DIR, MODEL_DATE, 'model_best.keras'))
NPZ_DIR         = os.getenv('NPZ_DIR',   os.path.join(BASE_DIR, 'NPZ'))
NPZ_FILE        = os.getenv('NPZ_FILE',  os.path.join(NPZ_DIR, 'DB.npz'))
PRED_BATCH_SIZE = int(os.getenv('PRED_BATCH_SIZE', 16))
CLASS_LABELS    = os.getenv('CLASS_LABELS', 'Campa Ver,Coraz Azu,Engra Ama,Engra Ama').split(',')
CM_OUTPUT_NAME  = os.getenv('CM_OUTPUT_NAME', 'MC_Aumen')
CM_OUTPUT_DPI   = int(os.getenv('CM_OUTPUT_DPI', 600))
CM_FIGURE_DPI   = int(os.getenv('CM_FIGURE_DPI', 300))
CM_CMAP         = os.getenv('CM_CMAP', 'Purples')

print(MODEL_PATH)

model = load_model(MODEL_PATH, compile=False)
print(model.summary())

DB = np.load(NPZ_FILE, allow_pickle=True)
x_test = DB['x_test'] / 255.

y_test = DB['y_test']

y_predict = model.predict(x_test, batch_size=PRED_BATCH_SIZE)
y_predict = np.argmax(y_predict, axis=1)

print('y_predict', y_predict.shape)
print('y_test', y_test.shape)

print(classification_report(y_test, y_predict, target_names=CLASS_LABELS))

plt.figure(figsize=[5.0, 5.0], tight_layout=True, dpi=CM_FIGURE_DPI)
cm_display = ConfusionMatrixDisplay.from_predictions(y_test,
                                                     y_predict,
                                                     display_labels=CLASS_LABELS,
                                                     xticks_rotation="vertical",
                                                     cmap=CM_CMAP,
                                                     normalize='true',
                                                     )

plt.xlabel('')
plt.ylabel('')
plt.savefig(os.path.join(BASE_DIR, CM_OUTPUT_NAME),
            dpi=CM_OUTPUT_DPI,
            bbox_inches='tight',
            pad_inches=0.1)
plt.close()
