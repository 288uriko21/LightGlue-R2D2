import torch
import cv2
from pathlib import Path

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ZippyPoint, R2D2
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from statistics import median
import os

def compute_precision(points1, points2, matches, homography, threshold=10):
    """
    Precision = TP / (TP + FP)
    Точность: доля корректных матчей среди всех предсказанных.
    """
    matches = matches.tolist() if isinstance(matches, torch.Tensor) else matches
    homography = torch.inverse(homography)

    true_positives = 0
    total_predicted = 0

    for match in matches:
        kp1 = points1[match[0]]
        kp2 = points2[match[1]]
        kp1_h = torch.cat([kp1, torch.tensor([1.0])])
        projected_kp2 = torch.matmul(homography, kp1_h)
        projected_kp2 = projected_kp2[:2] / projected_kp2[2]

        dist = torch.norm(kp2 - projected_kp2)

        if dist < threshold:
            true_positives += 1
        total_predicted += 1

    return true_positives / total_predicted if total_predicted > 0 else 0.0


def compute_recall(points1, points2, matches, homography, threshold=10):
    """
    Recall = TP / (TP + FN)
    Полнота: сколько из возможных GT-сопоставлений модель нашла.
    """
    matches = matches.tolist() if isinstance(matches, torch.Tensor) else matches
    homography = torch.inverse(homography)

    projected_points = []
    for kp1 in points1:
        kp1_h = torch.cat([kp1, torch.tensor([1.0])])
        projected_kp2 = torch.matmul(homography, kp1_h)
        projected_kp2 = projected_kp2[:2] / projected_kp2[2]
        projected_points.append(projected_kp2)
    projected_points = torch.stack(projected_points)

    # для каждой проекции проверяем, есть ли в предсказанных матчах ближайшая точка
    matched_gt = torch.zeros(len(points1), dtype=torch.bool)

    for match in matches:
        idx1 = match[0]
        idx2 = match[1]
        kp2 = points2[idx2]
        projected = projected_points[idx1]
        dist = torch.norm(kp2 - projected)
        if dist < threshold:
            matched_gt[idx1] = True

    recall = matched_gt.sum().item() / len(points1)
    return recall

def compute_f1(points1, points2, matches, homography, threshold=10):
    """
    F1-score: гармоническое среднее precision и recall.
    """
    precision = compute_precision(points1, points2, matches, homography, threshold)
    recall = compute_recall(points1, points2, matches, homography, threshold)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def apply_homography_cv2(image, H_torch):
    """
    Применяет гомографию к изображению с помощью OpenCV.
    :param image: numpy-массив изображения.
    :param H_torch: гомография в формате torch.tensor (3x3).
    :return: преобразованное изображение.
    """
    h, w = image.shape[:2]
    H_np = H_torch.numpy()
    warped = cv2.warpPerspective(image, H_np, (w, h), flags=cv2.INTER_LINEAR)
    return warped

import torch
import math

def create_rotation_homography_full_view(image_width, image_height, degrees):
    """
    Создает гомографию поворота изображения вокруг центра на заданный угол (по часовой стрелке),
    и возвращает откорректированную матрицу, чтобы изображение полностью влезало без обрезки.

    :param image_width: ширина исходного изображения
    :param image_height: высота исходного изображения
    :param degrees: угол поворота по часовой стрелке
    :return: гомография 3x3 в формате torch.tensor, учитывающая смещение
    """
    theta = math.radians(degrees)  # по часовой стрелке

    cx, cy = image_width / 2, image_height / 2

    # Гомография: поворот вокруг центра
    T1 = torch.tensor([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1]
    ], dtype=torch.float32)

    R = torch.tensor([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta),  math.cos(theta), 0],
        [0, 0, 1]
    ], dtype=torch.float32)

    T2 = torch.tensor([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0, 1]
    ], dtype=torch.float32)

    H = T2 @ R @ T1  # базовая гомография

    # Рассчитаем смещение, чтобы изображение влезло
    H_np = H.numpy()

    # исходные углы изображения
    corners = np.array([
        [0, 0],
        [image_width - 1, 0],
        [image_width - 1, image_height - 1],
        [0, image_height - 1]
    ], dtype=np.float32).reshape(-1, 1, 2)

    transformed_corners = cv2.perspectiveTransform(corners, H_np)
    x_coords = transformed_corners[:, 0, 0]
    y_coords = transformed_corners[:, 0, 1]
    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()

    offset_x = -min_x
    offset_y = -min_y

    # Смещающая матрица
    translation = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float32)

    H_adjusted = torch.from_numpy(translation @ H_np)

    return H_adjusted


def create_horizontal_flip_homography(image_width):
    """
    Создает матрицу гомографии для зеркального отражения изображения по вертикальной оси.
    """
    H = torch.tensor([
        [-1.0,  0.0, image_width],
        [ 0.0,  1.0, 0.0],
        [ 0.0,  0.0, 1.0]
    ])
    return H

# 1. Функция для создания гомографии при масштабировании
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


# 2. Функция для вычисления MMA
def compute_mma_at_threshold(points1, points2, matches, homography, threshold=3):
    """
    Вычисление Mean Matching Accuracy на пороговом значении (например, 3 пикселя).
    """
    keypoints1 = points1
    keypoints2 = points2

    correct_matches = 0
    total_matches = 0

    matches = matches.tolist() if isinstance(matches, torch.Tensor) else matches

    homography = torch.inverse(homography)

    for match in matches:
        kp1 = keypoints1[match[0]]  # Точка на первом изображении (x1, y1)
        kp2 = keypoints2[match[1]]  # Точка на втором изображении (x2, y2)

        # Преобразуем точку kp1 в гомогенные координаты
        kp1_homogeneous = torch.cat([kp1, torch.tensor([1.0])])  # (x1, y1, 1)

        projected_kp2 = torch.matmul(homography, kp1_homogeneous)  # (3,)

        projected_kp2_normalized = projected_kp2[:2] / projected_kp2[2]

        # Проверка, насколько близка проекция и настоящая точка на втором изображении
        dist = torch.norm(kp2 - projected_kp2_normalized)

        if dist < threshold:
            correct_matches += 1

        total_matches += 1

    return correct_matches / total_matches if total_matches > 0 else 0.0


import cv2
import numpy as np
import torch

def apply_homography_cv2_full_view(image, H_torch):
    """
    Применяет гомографию к изображению, корректируя размер выходного изображения,
    чтобы оно полностью влезло и не обрезалось.
    :param image: numpy-массив изображения.
    :param H_torch: гомография в формате torch.tensor (3x3).
    :return: преобразованное изображение.
    """
    h, w = image.shape[:2]
    H_np = H_torch.numpy()

    # Углы исходного изображения
    corners = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32).reshape(-1, 1, 2)

    # Преобразуем углы
    transformed_corners = cv2.perspectiveTransform(corners, H_np)

    # Находим границы нового изображения
    x_coords = transformed_corners[:, 0, 0]
    y_coords = transformed_corners[:, 0, 1]
    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()

    # Смещение, чтобы координаты начинались с (0, 0)
    offset_x = -min_x
    offset_y = -min_y

    # Размеры выходного изображения
    new_w = int(np.ceil(max_x - min_x))
    new_h = int(np.ceil(max_y - min_y))

    # Смещающая матрица
    translation = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ])

    # Комбинируем смещение с исходной гомографией
    H_adjusted = translation @ H_np

    # Применяем трансформацию
    warped = cv2.warpPerspective(image, H_adjusted, (new_w, new_h), flags=cv2.INTER_LINEAR)

    return warped




def compute_matching_score_torch(points1, points2, matches, homography, threshold=10):

    keypoints1 = points1
    keypoints2 = points2

    correct_matches = 0

    matches = matches.tolist() if isinstance(matches, torch.Tensor) else matches

    homography = torch.inverse(homography)

    for match in matches:
        kp1 = keypoints1[match[0]]  # Точка на первом изображении (x1, y1)
        kp2 = keypoints2[match[1]]  # Точка на втором изображении (x2, y2)

        # Преобразуем ключевую точку kp1 в гомогенные координаты
        kp2_homogeneous = torch.cat([kp1, torch.tensor([1.0])])  # (x1, y1, 1)

        projected_kp2 = torch.matmul(homography, kp2_homogeneous)  # (3,)

        projected_kp2_normalized = projected_kp2[:2] / projected_kp2[2]

        dist = torch.norm(kp2 - projected_kp2_normalized)

        if dist < threshold:
            correct_matches += 1

    return correct_matches / len(matches) if len(matches) > 0 else 0.0



# 4. Функция для вычисления Repeatability
def compute_repeatability(points1, points2, matches, homography, threshold=3):
    """
    Вычисление Repeatability с учетом гомографии и без повторного использования точек.
    """
    keypoints1 = points1
    keypoints2 = points2

    repeatable_points = 0
    total_points = 0

    matches = matches.tolist() if isinstance(matches, torch.Tensor) else matches

    homography = torch.inverse(homography)

    used_kp2 = set()  # Храним индексы уже использованных точек на втором изображении

    for match in matches:
        idx1, idx2 = match
        if idx2 in used_kp2:
            continue  # Уже использована — пропускаем

        kp1 = keypoints1[idx1]
        kp2 = keypoints2[idx2]

        kp1_homogeneous = torch.cat([kp1, torch.tensor([1.0], device=kp1.device)])
        projected_kp2 = torch.matmul(homography, kp1_homogeneous)
        projected_kp2_normalized = projected_kp2[:2] / projected_kp2[2]

        dist = torch.norm(kp2 - projected_kp2_normalized)

        if dist < threshold:
            repeatable_points += 1
            used_kp2.add(idx2)  # Отмечаем, что использовали

        total_points += 1

    return repeatable_points / total_points if total_points > 0 else 0.0




########################################################################################################################

torch.set_grad_enabled(False)
images = Path("30 images")
folder_path = Path("30 images")
lstms = []
lstmma = []
lstrp = []

lstpr = []
lstre = []
lstf1 = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor = R2D2(max_num_keypoints=600)  # load the extractor
#extractor = DISK(max_num_keypoints=600).eval().to(device)
matcher = LightGlue(features="r2d2").eval().to(device)
# state_dict = torch.load("checkpoint_best_180.tar", map_location="cpu")['model']
# matcher.load_state_dict(state_dict, strict=False)
torch.cuda.is_available()

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        scale = 1.5

        print(filename)

        img = cv2.imread(images / filename, cv2.IMREAD_GRAYSCALE)

        image_width = img.shape[1]

        flipped_image = cv2.flip(img, 1)
        #plt.imshow(flipped_image)
        print(flipped_image.shape)


        cv2.imwrite("img.png", flipped_image)

        # scaled_img = cv2.resize(img, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_LINEAR)
        # #plt.imshow(scaled_img)
        # cv2.imwrite("img.png", scaled_img)
        #
        #
        # image0 = load_image(images/filename)
        # image1 = load_image("img.png")
        # H = create_rotation_homography_full_view(image_width=img.shape[1], image_height=img.shape[0], degrees=15)

        # Применяем
        # rotated_image = apply_homography_cv2_full_view(img, H)
        # cv2.imwrite("img.png", rotated_image)


        image0 = load_image(images/filename)
        image1 = load_image("img.png")

        feats0 = extractor.extract(images/filename)
        feats1 = extractor.extract("img.png")

        # feats0 = extractor.extract(image0.to(device))
        # feats1 = extractor.extract(image1.to(device))

        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension
        #
        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]


        H = create_horizontal_flip_homography(image_width)
        #H = create_scaling_homography(scale)
        # H = torch.inverse(H)
        # rotated_image = apply_homography_cv2_full_view(rotated_image, H)
        # cv2.imwrite("img.png", rotated_image)

        ms = compute_matching_score_torch(kpts0, kpts1, matches, H)
        #print("Matching score: ", ms)
        lstms.append(ms)
        print("Медиана ms", median(lstms))

        rp = compute_repeatability(kpts0, kpts1, matches, H)
        #print("Matching score: ", rp)
        lstrp.append(rp)
        print("Медиана rp", median(lstrp))

        mma = compute_mma_at_threshold(kpts0, kpts1, matches, H)
        #print("Matching score: ", mma)
        lstmma.append(mma)
        print("Медиана mma", median(lstmma))

        # rec = compute_recall(kpts0, kpts1, matches, H)
        # lstre.append(rec)
        # print("Медиана recall ", median(lstre))
        #
        # pre = compute_precision(kpts0, kpts1, matches, H)
        # lstpr.append(pre)
        # print("Медиана precision ", median(lstpr))
        #
        # f1 = compute_f1(kpts0, kpts1, matches, H)
        # lstf1.append(f1)
        # print("Медиана f1 ", median(lstf1))

# print("Repeatability: ", compute_repeatability(kpts0, kpts1, matches, H))
#print("Медиана", median(lst))











