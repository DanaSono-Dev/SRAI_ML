from ultralytics import YOLO
from PIL import Image
import sys

# Carga el modelo (extrae best.pt del zip primero, o usa la ruta directa)
model = YOLO("yolov8s_ep50_lr0.001_drop0.3/weights/best.pt")

# Ruta a tu imagen de prueba
image_path = "Prueba_TYLCV.webp"  # ← cambia esto

# Inferencia
results = model.predict(source=image_path, imgsz=224)

# Mostrar resultados
result = results[0]
probs = result.probs  # objeto con probabilidades por clase

print("=== Resultado ===")
print(f"Clase predicha:  {result.names[probs.top1]}")
print(f"Confianza:       {probs.top1conf:.2%}")
print()
print("Top 5 clases:")
for i, (cls_idx, conf) in enumerate(zip(probs.top5, probs.top5conf)):
    print(f"  {i+1}. {result.names[cls_idx]:<20} {conf:.2%}")

# Mostrar imagen con resultado (opcional)
result.show()