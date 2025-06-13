from pathlib import Path

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ZippyPoint, R2D2
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch

torch.set_grad_enabled(False)
images = Path("assets")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

# extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
# matcher = LightGlue(features="superpoint").eval().to(device)
# torch.cuda.is_available()





extractor = SuperPoint(max_num_keypoints=400)  # load the extractor
# extractor = DISK(max_num_keypoints=500).eval().to(device)
# matcher = LightGlue(features="superpoint").eval().to(device)

# checkpoint = torch.load("checkpoint_best.tar", map_location="cpu")
# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", checkpoint.keys())

matcher = LightGlue(features='superpoint')  # важно: input_dim=128
# state_dict = torch.load("checkpoint_best_180.tar", map_location="cpu")['model']
# matcher.load_state_dict(state_dict, strict=False)

# state_dict = matcher.state_dict()





torch.cuda.is_available()


image0 = load_image(images / "spas.jpg")
image1 = load_image(images / "spas2.jpg")

feats0 = extractor.extract(image0.to(device))
feats1 = extractor.extract(image1.to(device))

# feats0 = extractor.extract(images/"spas.jpg")
# feats1 = extractor.extract(images/"spas2.jpg")

# print(feats0)
# print(feats0['keypoints'].shape)
# print(feats0['descriptors'].shape)
# print(feats0['image_size'].shape)
# print(feats0['keypoint_scores'].shape)

matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [
    rbd(x) for x in [feats0, feats1, matches01]
]  # remove batch dimension
#
kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

# print(matches)
# print(kpts0)
# print(kpts1)

axes = viz2d.plot_images([image0, image1], adaptive=False)

viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
#viz2d.add_text(0, f'Stop
# after {matches01["stop"]} layers', fs=20)
viz2d.save_plot("results/1")

kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
viz2d.plot_images([image0, image1])
# viz2d.plot_matches(m_kpts0, m_kpts1, color="crimson", lw=0.4)
axes = viz2d.plot_images([image0, image1], adaptive=False)
viz2d.plot_keypoints([kpts0, kpts1], colors=['b', 'r'], ps=10)
viz2d.save_plot("results/2")
