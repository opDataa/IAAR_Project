# Rutas del drive
DATASET_FOLDER__PATH = "C:/Users/Jtorr/PycharmProjects/IAAR_Project/dataset"
IMAGES_FOLDER__PATH = DATASET_FOLDER__PATH + "/images"
LABELS_FOLDER__PATH = DATASET_FOLDER__PATH + "/labels"
########################################################################################################################


import os
# Carga las imagenes y labels en listas
images_list = os.listdir(IMAGES_FOLDER__PATH)
labels_list = os.listdir(LABELS_FOLDER__PATH)

#print(f"Total de imágenes: {len(images_list)}")
#print(f"\nTotal de etiquetas: {len(labels_list)}")

########################################################################################################################
# DEBUG ONLY
"""
# Verificar que cada imagen tenga su archivo de etiqueta correspondiente
i = 0
for img in images_list:
    img_name = os.path.splitext(img)[0].strip().lower()  # Eliminar espacios y convertir a minúsculas
    # Revisamos si el archivo de etiqueta correspondiente existe
    if not any(os.path.splitext(lbl)[0].strip().lower() == img_name for lbl in labels_list):
        print(f"Falta la etiqueta para la imagen: {img}")
        i += 1

# Verificar que cada etiqueta tenga su archivo de imagen correspondiente
j = 0
for lbl in labels_list:
    lbl_name = os.path.splitext(lbl)[0].strip().lower()  # Eliminar espacios y convertir a minúsculas
    # Revisamos si la imagen correspondiente existe
    if not any(os.path.splitext(img)[0].strip().lower() == lbl_name for img in images_list):
        print(f"Falta la imagen para la etiqueta: {lbl}")
        j += 1

# Imprimir total de imágenes y etiquetas sin coincidencias
print(f"\nTotal de imágenes sin etiqueta: {i}")
print(f"Total de etiquetas sin imagen: {j}")
"""


########################################################################################################################
"""
# [X] Dividimos las imanges y los labels en 2 grupos
import random
# Establecemos un porcentaje para entrenamiento
TRAIN_PERCENTAGE = 0.8 # range(0 - 1)
train_size = int(len(images_list) * TRAIN_PERCENTAGE)
test_size = len(images_list) - train_size
# Mezclamos las listas
combined = list(zip(images_list, labels_list))
random.shuffle(combined)
# Guardamos un conjunto de entrenamiento y otro de validacion
train_images, train_labels = zip(*combined[:train_size])
val_images, val_labels = zip(*combined[train_size:])
"""
# [X] Guardamos cada grupo en su carpeta correspondiente en el dataset

# Rutas de las carpetas a crear
IMAGES_TRAIN_FOLDER__PATH = IMAGES_FOLDER__PATH + "/train"
IMAGES_VAL_FOLDER__PATH = IMAGES_FOLDER__PATH + "/val"
LABELS_TRAIN_FOLDER__PATH = LABELS_FOLDER__PATH + "/train"
LABELS_VAL_FOLDER__PATH = LABELS_FOLDER__PATH + "/val"

"""
import shutil
# Función para reiniciar una carpeta
def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)  # Elimina carpeta y todo su contenido
    os.makedirs(path)        # La vuelve a crear vacía
# Resetear las carpetas de imágenes y etiquetas (las borra y las crea si no existe)
reset_dir(IMAGES_TRAIN_FOLDER__PATH)
reset_dir(IMAGES_VAL_FOLDER__PATH)
reset_dir(LABELS_TRAIN_FOLDER__PATH)
reset_dir(LABELS_VAL_FOLDER__PATH)

# Función para copiar los pares
def copiar_dataset(image_list, images_dest_dir, labels_dest_dir):
    for img_file in image_list:
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + '.txt'

        # Rutas completas
        src_img = os.path.join(IMAGES_FOLDER__PATH, img_file)
        src_lbl = os.path.join(LABELS_FOLDER__PATH, label_file)
        dst_img = os.path.join(images_dest_dir, img_file)
        dst_lbl = os.path.join(labels_dest_dir, label_file)

        # Copiar si existen ambos
        if os.path.exists(src_img) and os.path.exists(src_lbl):
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)
        else:
            print(f"⚠️ Falta imagen o etiqueta: {img_file}")

# Copiar los archivos en las carpetas
copiar_dataset(train_images, IMAGES_TRAIN_FOLDER__PATH, LABELS_TRAIN_FOLDER__PATH)
copiar_dataset(val_images, IMAGES_VAL_FOLDER__PATH, LABELS_VAL_FOLDER__PATH)
print("✅ Archivos copiados correctamente.")
"""

########################################################################################################################

"""
# Verificar que cada imagen tenga su archivo de etiqueta correspondiente en las carpetas 'train' y 'eval'
def verificar_integridad(dataset_type):
    images_list = os.listdir(os.path.join(IMAGES_FOLDER__PATH, dataset_type))
    labels_list = os.listdir(os.path.join(LABELS_FOLDER__PATH, dataset_type))

    i = 0
    # Verificar que cada imagen tenga su archivo de etiqueta correspondiente
    for img in images_list:
        img_name = os.path.splitext(img)[0].strip().lower()  # Eliminar espacios y convertir a minúsculas
        # Revisamos si el archivo de etiqueta correspondiente existe
        if not any(os.path.splitext(lbl)[0].strip().lower() == img_name for lbl in labels_list):
            print(f"Falta la etiqueta para la imagen: {img}")
            i += 1

    j = 0
    # Verificar que cada etiqueta tenga su archivo de imagen correspondiente
    for lbl in labels_list:
        lbl_name = os.path.splitext(lbl)[0].strip().lower()  # Eliminar espacios y convertir a minúsculas
        # Revisamos si la imagen correspondiente existe
        if not any(os.path.splitext(img)[0].strip().lower() == lbl_name for img in images_list):
            print(f"Falta la imagen para la etiqueta: {lbl}")
            j += 1

    # Imprimir total de imágenes y etiquetas sin coincidencias
    print(f"\nTotal de imágenes sin etiqueta en {dataset_type}: {i}")
    print(f"Total de etiquetas sin imagen en {dataset_type}: {j}")


# Verificación para las carpetas 'train' y 'eval'
verificar_integridad('train')
verificar_integridad('val')  # Si tienes una carpeta 'eval' también
"""
########################################################################################################################
# [X] Generar YAML de configuracion para YOLO

YAML_FILE__PATH = DATASET_FOLDER__PATH + "/data.yaml"
def gen_yaml_configuration_file():
    import yaml

    # Datos del yaml de configuracion
    yaml_data = {
        'train': IMAGES_TRAIN_FOLDER__PATH,
        'val': IMAGES_VAL_FOLDER__PATH,
        'nc': 5,
        'names': ["bola_cerca", "bola_lejos", "linea_cerca", "hexapodo_propio", "hexapodo_ajeno"]
    }

    # Guardamos el yaml
    with open(YAML_FILE__PATH, 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file)

    print(f"Archivo YAML guardado en: {YAML_FILE__PATH}")

# gen_yaml_configuration_file()
########################################################################################################################
# [X] Lanzar entrenamiento

from ultralytics import YOLO

# Cargar modelo base preentrenado (por ejemplo yolov8n.pt)
model = YOLO("yolov8n.pt")  # Puedes cambiar a yolov8s.pt, yolov8m.pt, etc.

# Entrenamiento
model.train(
    # Ruta del yaml de configuracion
    data=YAML_FILE__PATH,

    # Épocas: cuántas veces verá el modelo todo el conjunto de entrenamiento (Mas epocas = mas aprendizaje y riesgo de sobrecalentamiento)
    epochs=25,

    # Dimensiones de las imagenes [EJ] imgsz=640 := 640x640 .
    # YOLO se encarga de redimensionar si la imagen no cumple ese tamaño
    imgsz=640,

    # Tamaño del lote: cuantas imagenes se procesan de golpe durante el entrenamiento
    batch=16,

    # Nombre del modelo entrenado
    name="modelo_1"
)




########################################################################################################################



