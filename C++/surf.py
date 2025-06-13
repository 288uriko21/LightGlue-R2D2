import subprocess
import torch
import numpy as np
import cv2


def compute_matching_score_torch(points1, points2, homography):
    """
    Вычисление Matching Score.
    """
    keypoints1 = points1["keypoints"].squeeze(0)  # torch.Size([2048, 2])
    keypoints2 = points2["keypoints"].squeeze(0)  # torch.Size([2048, 2])

      # Преобразуем точки из первого изображения через гомографию
    keypoints1_h = torch.cat([keypoints1, torch.ones((keypoints1.shape[0], 1))], dim=1)  # (x, y, 1)
    projected_keypoints1 = torch.matmul(homography, keypoints1_h.T).T  # [N, 3]
    projected_keypoints1 = projected_keypoints1[:, :2] / projected_keypoints1[:, 2:3]  # Нормализуем

    correct_matches = 0
    for proj_kp1 in projected_keypoints1:
        dists = torch.norm(keypoints2 - proj_kp1, dim=1)
        if torch.min(dists) < 10:
            correct_matches += 1
    matching_score = correct_matches / len(projected_keypoints1)
    return matching_score

def create_scaling_homography(scale):
    """
    Создает гомографию для масштабирования.
    :param scale: Коэффициент масштабирования (например, 1/1.5).
    :return: Матрица гомографии 3x3 в формате torch.tensor.
    """
    return torch.tensor([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ], dtype=torch.float32)

def create_horizontal_flip_homography(image_width):
    """
    Создает матрицу гомографии для зеркального отражения по вертикальной оси.
    """
    return torch.tensor([
        [-1.0,  0.0, image_width],
        [ 0.0,  1.0, 0.0],
        [ 0.0,  0.0, 1.0]
    ], dtype=torch.float32)

def run_surf(image_path, output_file):
    subprocess.run(["./surf_extractor", image_path, output_file])

def load_surf_points(file_path):
    with open(file_path, "rb") as f:
        num_points = np.frombuffer(f.read(4), dtype=np.int32)[0]
        points = np.frombuffer(f.read(num_points * 2 * 4), dtype=np.float32).reshape(num_points, 2)
        descriptors = np.frombuffer(f.read(), dtype=np.float32).reshape(num_points, -1)
    return {
        "keypoints": torch.tensor(points, dtype=torch.float32),
        "descriptors": torch.tensor(descriptors, dtype=torch.float32)
    }

# Пример использования
image_path = "COCO_train2014_000000508467.jpg"
output_file = "surf_output.bin"

# Извлечение ключевых точек SURF
run_surf(image_path, output_file)

#
# print("Форма ключевых точек:", surf_points["keypoints"].shape)
# print("Форма дескрипторов:", surf_points["descriptors"].shape)



# scale = 4
# scaled_path = "img_scaled.png"
#
# img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# scaled_img = cv2.resize(img, None, fx=1/scale, fy=1/scale)
# cv2.imwrite(scaled_path, scaled_img)
#
#
# # Запускаем SURF для оригинального и масштабированного изображения
# run_surf(image_path, "surf_original.bin")
# run_surf(scaled_path, "surf_scaled.bin")
#
# # Загружаем ключевые точки
# points1 = load_surf_points("surf_original.bin")
# points2 = load_surf_points("surf_scaled.bin")
#
# # Создаём гомографию и вычисляем метрики
# H = create_scaling_homography(scale)
#
# print("matching score: ", compute_matching_score_torch(points2, points1, H))

# Пример использования
image_path = "COCO_train2014_000000573966.jpg"
flipped_path = "img_flipped.png"

# Загружаем и создаем зеркальное изображение
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
flipped_img = cv2.flip(img, 1)  # Горизонтальное отражение
cv2.imwrite(flipped_path, flipped_img)

# Извлечение ключевых точек SURF
run_surf(image_path, "surf_original.bin")
run_surf(flipped_path, "surf_flipped.bin")

# Загружаем ключевые точки
points1 = load_surf_points("surf_original.bin")
points2 = load_surf_points("surf_flipped.bin")

# Создаем гомографию для зеркального отражения
H = create_horizontal_flip_homography(img.shape[1])

# Вычисляем matching score
print("matching score: ", compute_matching_score_torch(points2, points1, H))
