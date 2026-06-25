import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def random_rotation(img):
    angle = random.uniform(-30, 30)
    return img.rotate(angle, expand=True, fillcolor=(128, 128, 128))

def random_flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT) if random.random() > 0.5 else img

def random_brightness(img):
    return ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5))

def random_contrast(img):
    return ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.5))

def random_noise(img):
    arr = np.array(img).astype(np.int16)
    noise = np.random.randint(-25, 25, arr.shape, dtype=np.int16)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

def random_blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))

def random_crop(img, min_ratio=0.7):
    w, h = img.size
    crop_w = int(random.uniform(min_ratio, 1.0) * w)
    crop_h = int(random.uniform(min_ratio, 1.0) * h)
    x = random.randint(0, w - crop_w)
    y = random.randint(0, h - crop_h)
    return img.crop((x, y, x + crop_w, y + crop_h)).resize((w, h), Image.LANCZOS)


TRANSFORMS = [random_rotation, random_flip, random_brightness,
              random_contrast, random_noise, random_blur, random_crop]


def augment(img):
    for transform in random.sample(TRANSFORMS, k=random.randint(2, len(TRANSFORMS))):
        img = transform(img)
    return img


def process_image(image_path, out_dir, base_name, index, num_augmentations):
    img = Image.open(image_path).convert("RGB")
    suffix = image_path.suffix

    new_name = f"{base_name}_{index}"

    # Guardar original con nuevo nombre
    original_out = out_dir / f"{new_name}{suffix}"
    img.save(original_out)
    print(f"  original -> {original_out}")

    # Guardar aumentos
    for i in range(1, num_augmentations + 1):
        out_path = out_dir / f"{new_name}_aug_{i}{suffix}"
        augment(img.copy()).save(out_path)
        print(f"  aug_{i} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Genera versiones aumentadas de imágenes y las guarda en una carpeta destino."
    )
    parser.add_argument("carpeta_origen", help="Ruta a la carpeta con las imágenes originales")
    parser.add_argument("carpeta_destino", help="Ruta a la carpeta donde se guardarán las imágenes")
    parser.add_argument("--iniciar-en", type=int, default=1, metavar="N",
                        help="Número desde el que empezar a nombrar las imágenes (por defecto: 1)")
    parser.add_argument("--cantidad", type=int, default=10, metavar="N",
                        help="Versiones aumentadas por imagen (por defecto: 10)")

    args = parser.parse_args()
    input_path = Path(args.carpeta_origen)
    output_path = Path(args.carpeta_destino)

    if not input_path.is_dir():
        print(f"Error: '{args.carpeta_origen}' no es una carpeta válida.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    # El nombre base se toma del nombre de la carpeta destino
    base_name = output_path.name

    images = sorted(p for p in input_path.iterdir()
                    if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)

    if not images:
        print(f"No se encontraron imágenes en '{args.carpeta_origen}'.")
        return

    print(f"Imágenes encontradas: {len(images)}")
    print(f"Nombre base: {base_name}")
    print(f"Iniciando desde: {args.iniciar_en}\n")

    for i, image_path in enumerate(images):
        index = args.iniciar_en + i
        print(f"Procesando: {image_path.name} -> {base_name}_{index}")
        try:
            process_image(image_path, output_path, base_name, index, args.cantidad)
        except Exception as e:
            print(f"  Error al procesar {image_path.name}: {e}")
        print()

    print("Proceso completado.")


if __name__ == "__main__":
    main()