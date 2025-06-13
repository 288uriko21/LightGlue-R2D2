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


def compute_matching_score_torch(points1, points2, homography):
    """
    Вычисление Matching Score.
    """
    keypoints1 = points1 # torch.Size([2048, 2])
    keypoints2 = points2 # torch.Size([2048, 2])

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

def compute_repeatability_torch(points1, points2, homography, threshold=5):
    keypoints1 = points1
    keypoints2 = points2

    keypoints1_h = torch.cat([keypoints1, torch.ones((keypoints1.shape[0], 1))], dim=1)
    projected_keypoints1 = torch.matmul(homography, keypoints1_h.T).T
    projected_keypoints1 = projected_keypoints1[:, :2] / projected_keypoints1[:, 2:3]

    matched2 = torch.zeros(len(keypoints2), dtype=torch.bool)  # Отмечаем, какие точки уже были использованы
    correct_repeatability = 0

    for proj_kp1 in projected_keypoints1:
        dists = torch.norm(keypoints2 - proj_kp1, dim=1)
        min_dist, min_idx = torch.min(dists, dim=0)
        if min_dist < threshold and not matched2[min_idx]:
            correct_repeatability += 1
            matched2[min_idx] = True  # Помечаем, что точка уже "занята"

    repeatability = correct_repeatability / min(len(keypoints1), len(keypoints2))
    return repeatability



def compute_mma_torch(points1, points2, homography, thresholds=[3]):
    """
    Вычисление MMA для данных, предоставленных в формате PyTorch.
    """
    keypoints1 = points1# torch.Size([2048, 2])
    keypoints2 = points2 # torch.Size([2048, 2])

    keypoints1_h = torch.cat([keypoints1, torch.ones((keypoints1.shape[0], 1))], dim=1)  # (x, y, 1)
    projected_keypoints1 = torch.matmul(homography, keypoints1_h.T).T  # [N, 3]
    projected_keypoints1 = projected_keypoints1[:, :2] / projected_keypoints1[:, 2:3]  # Нормализуем

    mma_scores = []
    for thresh in thresholds:
        correct_matches = 0
        for proj_kp1 in projected_keypoints1:
            # Расстояния до всех точек на втором изображении
            dists = torch.norm(keypoints2 - proj_kp1, dim=1)
            if torch.min(dists) < thresh:
                correct_matches += 1
        mma_scores.append(correct_matches / len(projected_keypoints1))

    mma = torch.tensor(mma_scores).mean().item()

    return mma

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



########################################################################################################################

torch.set_grad_enabled(False)
images = Path("30 images")
folder_path = Path("30 images")
lstms = []
lstrp = []
lstmma = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor = DISK(max_num_keypoints=600)  # load the extractor
#extractor = DISK(max_num_keypoints=600).eval().to(device)
matcher = LightGlue(features="r2d2").eval().to(device)
torch.cuda.is_available()

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        scale = 2.3

        print(filename)

        img = cv2.imread(images / filename, cv2.IMREAD_GRAYSCALE)

        image_width = img.shape[1]

        # flipped_image = cv2.flip(img, 1)
        # #plt.imshow(flipped_image)
        # print(flipped_image.shape)
        #
        #
         # cv2.imwrite("img.png", flipped_image)

        # scaled_img = cv2.resize(img, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_LINEAR)
        # #plt.imshow(scaled_img)


        H = create_rotation_homography_full_view(image_width=img.shape[1], image_height=img.shape[0], degrees=40)

        # Применяем
        rotated_image = apply_homography_cv2_full_view(img, H)
        cv2.imwrite("img.png", rotated_image)


        image0 = load_image(images/filename)
        image1 = load_image("img.png")

        # feats0 = extractor.extract(images/filename)
        # feats1 = extractor.extract("img.png")



        feats0 = extractor.extract(image0.to(device))
        feats1 = extractor.extract(image1.to(device))

        #H = create_horizontal_flip_homography(image_width)


        #print(feats0)


        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension
        #
        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]


        #H = create_horizontal_flip_homography(image_width)
        #H = create_scaling_homography(1/scale)
        #ms = compute_matching_score_torch(kpts0, kpts1, H)
        #rp = compute_repeatability_torch(kpts0, kpts1, H)
        mma = compute_mma_torch(kpts0, kpts1, H)
        # print("Matching score: ", ms)
        # lstms.append(ms)
        # print("repeatability: ", rp)
        # lstrp.append(rp)
        print("mma: ", mma)
        lstmma.append(mma)
        # print("Медиана ms", median(lstms))
        #print("Медиана rp", median(lstrp))
        print("Медиана mma", median(lstmma))

#         ms = compute_matching_score_torch(kpts0, kpts1, matches, H)
#         #mma =
#         print("Matching score: ", ms)
#         lst.append(ms)
#         print("Медиана", median(lst))
# # print("Repeatability: ", compute_repeatability(kpts0, kpts1, matches, H))
# print("Медиана", median(lst))











