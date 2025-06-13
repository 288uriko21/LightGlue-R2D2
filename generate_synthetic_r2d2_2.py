import os
import cv2
import numpy as np
import torch
from pathlib import Path
from lightglue import R2D2
from lightglue.utils import load_image
from tqdm import tqdm

# Параметры
input_folder = Path("30 images")  # <-- сюда положи свои изображения
output_folder = Path("synthetic_r2d2")
max_keypoints = 1024
image_size = (480, 640)  # можно адаптировать

# Инициализация
output_folder.mkdir(parents=True, exist_ok=True)
(output_folder / "feats").mkdir(exist_ok=True)
(output_folder / "matches").mkdir(exist_ok=True)
pair_list = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#extractor = R2D2(max_num_keypoints=max_keypoints).eval().to(device)
extractor = R2D2(max_num_keypoints=max_keypoints)

# ⚙️ Функция для создания случайной гомографии
def sample_homography(shape, perturbation=0.1):
    h, w = shape
    margin = 0.1
    pts1 = np.array([
        [margin, margin],
        [1 - margin, margin],
        [1 - margin, 1 - margin],
        [margin, 1 - margin]
    ])
    pts2 = pts1 + (np.random.rand(4, 2) - 0.5) * 2 * perturbation
    pts1 *= np.array([[w, h]])
    pts2 *= np.array([[w, h]])
    H, _ = cv2.findHomography(pts1, pts2)
    return H

# Обработка изображений
images = sorted([p for p in input_folder.glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])

for i, img_path in enumerate(tqdm(images, desc="Генерация пар")):
    base_name = f"image_{i:04d}"

    # Загрузка и ресайз
    image = load_image(img_path, resize=image_size)
    image_np = image.permute(1, 2, 0).cpu().numpy()

    # Гомография
    H = sample_homography(image_np.shape[:2])
    warped = cv2.warpPerspective(image_np, H, image_size[::-1], flags=cv2.INTER_LINEAR)

    # Сохранение и фичи
    tmp_img_path = output_folder / f"{base_name}_tmp.png"
    tmp_warp_path = output_folder / f"{base_name}_warp_tmp.png"
    cv2.imwrite(str(tmp_img_path), (image_np * 255).astype(np.uint8))
    cv2.imwrite(str(tmp_warp_path), (warped * 255).astype(np.uint8))

    # R2D2-фичи
    try:
        f0 = extractor.extract(tmp_img_path)
        f1 = extractor.extract(tmp_warp_path)
    except Exception as e:
        print(f"⚠️  Ошибка на паре {base_name}: {e}")
        continue

    # Удаление временных файлов
    tmp_img_path.unlink()
    tmp_warp_path.unlink()

    # Сохранение фичей
    def save_feat(name, feat):
        np.savez_compressed(
            output_folder / "feats" / f"{name}.npz",
            keypoints=feat["keypoints"][0].cpu().numpy(),
            descriptors=feat["descriptors"][0].cpu().numpy(),
            scores=feat["keypoint_scores"][0].cpu().numpy(),
            image_size=feat["image_size"][0].cpu().numpy(),
        )

    save_feat(base_name, f0)
    save_feat(f"{base_name}_warp", f1)

    # Генерация матчей через обратную гомографию
    kpts0 = f0["keypoints"][0].cpu().numpy()
    kpts1 = f1["keypoints"][0].cpu().numpy()
    kpts1_proj = cv2.perspectiveTransform(kpts0[None], H)[0]

    dists = np.linalg.norm(kpts1[None] - kpts1_proj[:, None], axis=-1)
    idx1 = dists.argmin(-1)
    dist = dists[np.arange(len(idx1)), idx1]
    valid = dist < 3  # пиксельная точность
    matches = np.stack([np.arange(len(idx1))[valid], idx1[valid]], axis=-1)

    # Генерация ground truth матчей (gt_matches0 и gt_matches1)
    gt_matches0 = np.arange(len(f0["keypoints"][0]))[valid]  # Индексы для первого изображения
    gt_matches1 = idx1[valid]  # Индексы для второго изображения

    np.savez_compressed(
        output_folder / "matches" / f"{base_name}___{base_name}_warp.npz",
        matches=matches,
        gt_matches0=gt_matches0,
        gt_matches1=gt_matches1
    )

    # Запись в список пар
    pair_list.append(f"{base_name} {base_name}_warp")

# Сохранение pairs.txt
with open(output_folder / "pairs.txt", "w") as f:
    f.write("\n".join(pair_list))

print("✅ Готово! Сгенерировано:", len(pair_list), "пар.")
