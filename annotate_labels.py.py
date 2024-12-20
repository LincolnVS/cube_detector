import cv2
import os
import numpy as np
import tqdm


NAMES = [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
         'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
         'U', 'V', 'W', 'X', 'Y', 'Z']


# Diretórios das imagens e labels
images_dir = "images"
labels_dir = "labels"
output_dir = "annotated_images"

# Criar o diretório de saída se não existir
os.makedirs(output_dir, exist_ok=True)

# Função para desenhar bounding boxes
for label_file in tqdm.tqdm(os.listdir(labels_dir), desc="Anotando imagens"): # Para cada arquivo de labels os.listdir(labels_dir):
    # Caminhos dos arquivos
    image_file = os.path.join(images_dir, label_file.replace(".txt", ".jpg"))
    label_path = os.path.join(labels_dir, label_file)

    # Verifica se a imagem existe
    if not os.path.exists(image_file):
        print(f"Imagem {image_file} não encontrada. Pulando.")
        continue

    # Carrega a imagem
    image = cv2.imread(image_file)
    h, w, _ = image.shape

    # Lê o arquivo de labels
    with open(label_path, "r") as f:
        for idx, line in enumerate(f):
            if line.strip():
                parts = line.split()
                if len(parts) >= 6:
                    class_id = int(parts[0])  # ID da classe 
                    class_name = NAMES[class_id]
                    x_center, y_center, bbox_width, bbox_height, angle = map(float, parts[1:6])

                    # Converter valores normalizados para coordenadas de pixel
                    x1 = int((x_center - bbox_width / 2) * w)
                    y1 = int((y_center - bbox_height / 2) * h)
                    x2 = int((x_center + bbox_width / 2) * w)
                    y2 = int((y_center + bbox_height / 2) * h)

                    # Coordenadas do centro do bounding box
                    center = (int(x_center * w), int(y_center * h))

                    # Dimensões em pixels
                    bbox_width_px = int(bbox_width * w)
                    bbox_height_px = int(bbox_height * h)

                    # Matriz de rotação
                    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

                    # Pontos do retângulo
                    rect_points = np.array([
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2]
                    ], dtype=np.float32)

                    # Rotacionar os pontos
                    rotated_points = cv2.transform(np.array([rect_points]), rotation_matrix)[0]
                    rotated_points = rotated_points.astype(np.int32)

                    # Desenha o bounding box rotacionado
                    cv2.polylines(image, [rotated_points], isClosed=True, color=(0, 255, 0), thickness=2)
                    # Escreve acima do quadrado o nome da classe
                    cv2.putText(image, f"{class_name} - {idx}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Salva a imagem anotada
    output_path = os.path.join(output_dir, os.path.basename(image_file))
    cv2.imwrite(output_path, image)

print(f"Imagens anotadas salvas em {output_dir}")
