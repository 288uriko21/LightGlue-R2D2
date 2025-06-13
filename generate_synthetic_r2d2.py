import os
import cv2
import numpy as np
import torch
import h5py
from pathlib import Path
from lightglue import R2D2
from lightglue.utils import load_image
from tqdm import tqdm

def get_rotation_homography(angle_deg, image_shape):
    """Генерация гомографии поворота вокруг центра изображения"""
    h, w = image_shape
    center = (w / 2, h / 2)
    R = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    # Преобразуем в 3x3 матрицу
    H = np.vstack([R, [0, 0, 1]])
    return H

def process_one_image_with_rotations(extractor, img_path, output_dir, image_size, angles, device):
    feats_dir = output_dir / "feats"
    matches_dir = output_dir / "matches"
    feats_dir.mkdir(exist_ok=True)
    matches_dir.mkdir(exist_ok=True)

    image = load_image(img_path, resize=image_size)
    image_np = image.permute(1, 2, 0).cpu().numpy()

    # Определяем начальный индекс фич
    current_count = len(list(feats_dir.glob("feat*.h5")))
    idx_original = current_count + 1
    name_original = f"feat{idx_original}"

    # Сохраняем оригинал
    tmp_path = output_dir / "tmp_img.png"
    cv2.imwrite(str(tmp_path), (image_np * 255).astype(np.uint8))
    feats0 = extractor.extract(tmp_path)
    tmp_path.unlink()

    kpts0 = feats0['keypoints'][0].cpu().numpy()
    desc0 = feats0['descriptors'][0].cpu().numpy()
    scores0 = feats0['keypoint_scores'][0].cpu().numpy()
    save_feat_h5(feats_dir / f"{name_original}.h5", kpts0, desc0, scores0, image_np.shape[:2])

    # Обрабатываем повороты
    for i, angle in enumerate(angles):
        H = get_rotation_homography(angle, image_np.shape[:2])
        warped = cv2.warpPerspective(image_np, H, image_size[::-1], flags=cv2.INTER_LINEAR)

        tmp_warp_path = output_dir / "tmp_warp.png"
        cv2.imwrite(str(tmp_warp_path), (warped * 255).astype(np.uint8))
        feats1 = extractor.extract(tmp_warp_path)
        tmp_warp_path.unlink()

        kpts1 = feats1['keypoints'][0].cpu().numpy()
        desc1 = feats1['descriptors'][0].cpu().numpy()
        scores1 = feats1['keypoint_scores'][0].cpu().numpy()

        idx_rot = idx_original + i + 1
        name_rot = f"feat{idx_rot}"
        save_feat_h5(feats_dir / f"{name_rot}.h5", kpts1, desc1, scores1, image_np.shape[:2])

        # Матчи и пары
        data = create_matches_data(kpts0, kpts1, H)
        save_gt_h5(matches_dir / f"{name_original}___{name_rot}.h5", data)

        H_flat = H.flatten()
        H_str = ' '.join(f"{v:.8f}" for v in H_flat)
        with open(output_dir / "pairs.txt", "a") as f:
            f.write(f"{name_original} {name_rot} {H_str}\n")




import numpy as np
import cv2

# def sample_homography(shape, perturbation=0.1):
#     """Генерация случайной гомографии"""
#     h, w = shape
#     margin = 0.1
#     pts1 = np.array([
#         [margin, margin],
#         [1 - margin, margin],
#         [1 - margin, 1 - margin],
#         [margin, 1 - margin]
#     ])
#     pts2 = pts1 + (np.random.rand(4, 2) - 0.5) * 2 * perturbation
#     pts1 *= np.array([[w, h]])
#     pts2 *= np.array([[w, h]])
#     H, _ = cv2.findHomography(pts1, pts2)
#     return H

def sample_homography(image_shape, perspective=True, rotation=True, scaling=True):
    h, w = image_shape
    center = np.array([w / 2, h / 2])

    # Стартовая — единичная матрица
    H = np.eye(3)

    if perspective:
        # Добавим случайную перспективную трансформацию
        pts1 = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)

        margin = 0.1  # насколько далеко можно сместить точки
        perturb = lambda x: x + np.random.uniform(-margin, margin) * np.array([w, h])
        pts2 = np.array([perturb(p) for p in pts1], dtype=np.float32)

        H_persp = cv2.getPerspectiveTransform(pts1, pts2)
        H = H_persp @ H

    if rotation:
        angle = np.random.uniform(-10, 10)  # в градусах
        R = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
        R = np.vstack([R, [0, 0, 1]])  # 2x3 -> 3x3
        H = R @ H

    if scaling:
        scale = np.random.uniform(1, 1.01)
        S = np.array([
            [scale, 0, (1 - scale) * center[0]],
            [0, scale, (1 - scale) * center[1]],
            [0, 0, 1]
        ])
        H = S @ H

    return H



def create_matches_data(kpts0, kpts1, H, dist_thresh=3):
    """Создание ground truth соответствий на основе гомографии"""
    kpts1_proj = cv2.perspectiveTransform(kpts0[None], H)[0]
    dists = np.linalg.norm(kpts1[None] - kpts1_proj[:, None], axis=-1)
    idx1 = dists.argmin(-1)
    dist = dists[np.arange(len(idx1)), idx1]
    valid = dist < dist_thresh

    # Формируем данные в формате для LightGlue
    n0, n1 = len(kpts0), len(kpts1)
    gt_assignment = np.zeros((n0 + 1, n1 + 1))  # +1 для dustbins
    gt_matches0 = -np.ones(n0, dtype=int)
    gt_matches1 = -np.ones(n1, dtype=int)

    # Заполняем правильные соответствия
    for idx0, idx1 in enumerate(idx1):
        if valid[idx0]:
            gt_assignment[idx0, idx1] = 1
            gt_matches0[idx0] = idx1
            gt_matches1[idx1] = idx0

    # Заполняем dustbins для несоответствующих точек
    invalid0 = np.where(gt_matches0 == -1)[0]
    invalid1 = np.where(gt_matches1 == -1)[0]
    gt_assignment[invalid0, -1] = 1
    gt_assignment[-1, invalid1] = 1

    gt_assignment = gt_assignment[:-1, :-1]  # обрезаем dustbins

    return {
        'keypoints0': kpts0.astype(np.float32),
        'keypoints1': kpts1.astype(np.float32),
        'gt_assignment': gt_assignment.astype(np.float32),
        'gt_matches0': gt_matches0.astype(np.int64),
        'gt_matches1': gt_matches1.astype(np.int64),
    }


def save_h5(data, path):
    """Сохранение данных в HDF5 файл"""
    with h5py.File(path, 'w') as f:
        for key, val in data.items():
            f.create_dataset(key, data=val)


def save_gt_h5(path, data):
    with h5py.File(path, 'w') as f:
        f.create_dataset("gt_matches0", data=data['gt_matches0'])
        f.create_dataset("gt_matches1", data=data['gt_matches1'])
        f.create_dataset("gt_assignment", data=data['gt_assignment'])

def save_feat_h5(path, keypoints, descriptors, scores, image_size):
    with h5py.File(path, 'w') as f:
        f.create_dataset("keypoints0", data=keypoints.astype(np.float32))
        f.create_dataset("descriptors0", data=descriptors.astype(np.float32))
        f.create_dataset("scores0", data=scores.astype(np.float32))
        f.create_dataset("image_size", data=np.array(image_size, dtype=np.float32))



def process_rotations_for_image(extractor, img_path, output_dir, image_size, angles, device):
    image = load_image(img_path, resize=image_size)
    image_np = image.permute(1, 2, 0).cpu().numpy()

    # Считаем, сколько уже feat-файлов
    feats_dir = output_dir / "feats"
    feats_dir.mkdir(exist_ok=True)
    current_feat_count = len(list(feats_dir.glob("feat*.h5")))

    # Имя для оригинала
    idx_src = current_feat_count + 1
    name_src = f"feat{idx_src}"

    # Сохраняем исходное изображение как временный файл
    tmp_img_path = output_dir / "tmp_src.png"
    cv2.imwrite(str(tmp_img_path), (image_np * 255).astype(np.uint8))
    feats_src = extractor.extract(tmp_img_path)
    tmp_img_path.unlink()

    kpts_src = feats_src['keypoints'][0].cpu().numpy()
    desc_src = feats_src['descriptors'][0].cpu().numpy()
    scores_src = feats_src['keypoint_scores'][0].cpu().numpy()

    # Сохраняем исходные фичи
    save_feat_h5(feats_dir / f"{name_src}.h5", kpts_src, desc_src, scores_src, image_np.shape[:2])

    # Обрабатываем повороты
    matches_dir = output_dir / "matches"
    matches_dir.mkdir(exist_ok=True)

    for i, angle in enumerate(angles):
        H = get_rotation_homography(angle, image_np.shape[:2])
        warped = cv2.warpPerspective(image_np, H, image_size[::-1], flags=cv2.INTER_LINEAR)

        tmp_warp_path = output_dir / "tmp_warp.png"
        cv2.imwrite(str(tmp_warp_path), cv2.cvtColor((warped * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        feats_warp = extractor.extract(tmp_warp_path)
        # tmp_warp_path.unlink()
        img = cv2.imread(tmp_warp_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(tmp_warp_path, img)

        kpts_warp = feats_warp['keypoints'][0].cpu().numpy()
        desc_warp = feats_warp['descriptors'][0].cpu().numpy()
        scores_warp = feats_warp['keypoint_scores'][0].cpu().numpy()

        idx_warp = idx_src + i + 1  # featX+1, featX+2, ...
        name_warp = f"feat{idx_warp}"

        # Сохраняем повернутые фичи
        save_feat_h5(feats_dir / f"{name_warp}.h5", kpts_warp, desc_warp, scores_warp, image_np.shape[:2])

        # Сохраняем gt
        data = create_matches_data(kpts_src, kpts_warp, H)
        save_gt_h5(matches_dir / f"{name_src}___{name_warp}.h5", data)

        # pairs.txt
        H_flat = H.flatten()
        H_str = ' '.join(f"{v:.8f}" for v in H_flat)
        with open(output_dir / "pairs.txt", "a") as f:
            f.write(f"{name_src} {name_warp} {H_str}\n")

def process_image_pair(extractor, img_path, output_dir, base_name, image_size, device):
    image = load_image(img_path, resize=image_size)
    image_np = image.permute(1, 2, 0).cpu().numpy()

    H = sample_homography(image_np.shape[:2])
    warped = cv2.warpPerspective(image_np, H, image_size[::-1], flags=cv2.INTER_LINEAR)

    tmp_img_path = output_dir / f"{base_name}_tmp.png"
    tmp_warp_path = output_dir / f"{base_name}_warp_tmp.png"
    cv2.imwrite(str(tmp_img_path), (image_np * 255).astype(np.uint8))
    cv2.imwrite(str(tmp_warp_path), (warped * 255).astype(np.uint8))

    img = cv2.imread(tmp_warp_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(tmp_warp_path, img)

    feats0 = extractor.extract(tmp_img_path)
    feats1 = extractor.extract(tmp_warp_path)

    # tmp_img_path.unlink()
    # tmp_warp_path.unlink()

    kpts0 = feats0['keypoints'][0].cpu().numpy()
    kpts1 = feats1['keypoints'][0].cpu().numpy()
    desc0 = feats0['descriptors'][0].cpu().numpy()
    desc1 = feats1['descriptors'][0].cpu().numpy()
    scores0 = feats0['keypoint_scores'][0].cpu().numpy()
    scores1 = feats1['keypoint_scores'][0].cpu().numpy()

    H = sample_homography(image_np.shape[:2])

    save_feat_h5(output_dir / "feats" / f"{base_name}.h5", kpts0, desc0, scores0, image_np.shape[:2])
    save_feat_h5(output_dir / "feats" / f"{base_name}_warp.h5", kpts1, desc1, scores1, image_np.shape[:2])

    data = create_matches_data(kpts0, kpts1, H)
    pair_name = f"{base_name}___{base_name}_warp"
    (output_dir / "matches").mkdir(exist_ok=True)
    save_gt_h5(output_dir / "matches" / f"{pair_name}.h5", data)

    # Сохраняем гомографию в виде строки из 9 чисел
    H_flat = H.flatten()
    H_str = ' '.join(f"{v:.8f}" for v in H_flat)

    # Обновляем pairs.txt
    with open(output_dir / "pairs.txt", "a") as f:
        f.write(f"{base_name} {base_name}_warp {H_str}\n")

    return pair_name



def main():
    # Параметры
    input_folder = Path("30 images")  # Папка с исходными изображениями
    output_folder = Path("synthetic_r2d2")  # Выходная папка
    max_keypoints = 1024  # Максимальное количество ключевых точек
    image_size = (480, 640)  # Размер изображений

    # Инициализация
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / "feats").mkdir(exist_ok=True)

    # Очистка файла пар, если он существует
    if (output_folder / "pairs.txt").exists():
        (output_folder / "pairs.txt").unlink()

    # Инициализация R2D2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = R2D2(max_num_keypoints=max_keypoints)

    # Обработка изображений
    images = sorted([p for p in input_folder.glob("*")
                     if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])

    for i, img_path in enumerate(tqdm(images, desc="Generating pairs")):
        i = i
        base_name = f"image_{i:04d}"
        try:
            process_image_pair(
                extractor, img_path, output_folder,
                base_name, image_size, device
            )
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # for i, img_path in enumerate(sorted(input_folder.glob("*.jpg"))):
    #
    #     angles = [10, 30, 45, 60, 90]
    #     process_one_image_with_rotations(extractor, img_path, output_folder, image_size, angles, device)
    #
    # print(f" Done! Generated {len(images)} pairs in {output_folder}")


if __name__ == "__main__":
    main()