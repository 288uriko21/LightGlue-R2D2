import os, pdb
from PIL import Image
import numpy as np
import torch
import cv2

from r2d2.tools import common
from r2d2.tools.dataloader import norm_RGB
from r2d2.nets.patchnet import *


def load_network(model_fn):
    checkpoint = torch.load(model_fn, map_location=torch.device('cpu'))
    print("\n>> Creating net = " + checkpoint['net'])
    net = eval(checkpoint['net'])
    nb_of_weights = common.model_size(net)
    print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return net.eval()


class NonMaxSuppression(torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr

    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability >= self.rel_thr)

        return maxima.nonzero().t()[2:4]

class R2D2:
    def __init__(self, max_num_keypoints=5000):
        self.max_keypoints = max_num_keypoints
        self.resize = [1024, 680]
        self.nms_window = 3
        self.keypoint_threshold = 0.0001
        self.ratio_threshold = 0.95

    def extract(self, img) -> dict:


        def extract_multiscale(net, img, detector, scale_f=2 ** 0.25,
                               min_scale=0.0, max_scale=1,
                               min_size=256, max_size=1024,
                               verbose=False):
            old_bm = torch.backends.cudnn.benchmark
            torch.backends.cudnn.benchmark = False  # speedup

            # extract keypoints at multiple scales
            B, three, H, W = img.shape
            assert B == 1 and three == 3, "should be a batch with a single RGB image"

            assert max_scale <= 1
            s = 1.0  # current scale factor

            X, Y, S, C, Q, D = [], [], [], [], [], []
            while s + 0.001 >= max(min_scale, min_size / max(H, W)):
                if s - 0.001 <= min(max_scale, max_size / max(H, W)):
                    nh, nw = img.shape[2:]
                    if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
                    # extract descriptors
                    with torch.no_grad():
                        res = net(imgs=[img])

                    # get output and reliability map
                    descriptors = res['descriptors'][0]
                    reliability = res['reliability'][0]
                    repeatability = res['repeatability'][0]

                    # normalize the reliability for nms
                    # extract maxima and descs
                    y, x = detector(**res)  # nms
                    c = reliability[0, 0, y, x]
                    q = repeatability[0, 0, y, x]
                    d = descriptors[0, :, y, x].t()
                    n = d.shape[0]

                    # accumulate multiple scales
                    X.append(x.float() * W / nw)
                    Y.append(y.float() * H / nh)
                    S.append((32 / s) * torch.ones(n, dtype=torch.float32, device=d.device))
                    C.append(c)
                    Q.append(q)
                    D.append(d)
                s /= scale_f

                # down-scale the image for next iteration
                nh, nw = round(H * s), round(W * s)
                img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)

            # restore value
            torch.backends.cudnn.benchmark = old_bm

            Y = torch.cat(Y)
            X = torch.cat(X)
            S = torch.cat(S)  # scale
            scores = torch.cat(C) * torch.cat(Q)  # scores = reliability * repeatability
            XYS = torch.stack([X, Y], dim=-1)
            D = torch.cat(D)
            return XYS, D, scores

        def extract_keypoints(args):
            iscuda = common.torch_set_gpu(args.gpu)

            # load the network...
            net = load_network(args.model)
            if iscuda: net = net.cuda()

            # create the non-maxima detector
            detector = NonMaxSuppression(
                rel_thr=args.reliability_thr,
                rep_thr=args.repeatability_thr)

            while args.images:
                img_path = args.images.pop(0)

                if str(img_path).endswith('.txt'):
                    args.images = open(img_path).read().splitlines() + args.images
                    continue

                print(f"\nExtracting features for {img_path}")
                img = Image.open(img_path).convert('RGB')
                #img = img.resize((1024, 1024), Image.Resampling.LANCZOS)   ##################  хз, сама написала, если что

                W, H = img.size
                img = norm_RGB(img)[None]
                if iscuda: img = img.cuda()

                # extract keypoints/descriptors for a single image
                xys, desc, scores = extract_multiscale(net, img, detector,
                                                       scale_f=args.scale_f,
                                                       min_scale=args.min_scale,
                                                       max_scale=args.max_scale,
                                                       min_size=args.min_size,
                                                       max_size=args.max_size,
                                                       verbose=True)

                xys = xys.cpu().numpy()
                desc = desc.cpu().numpy()
                scores = scores.cpu().numpy()
                idxs = scores.argsort()[-args.top_k or None:]

                #outpath = img_path + '.' + args.tag
                #print(f"Saving {len(idxs)} keypoints to {outpath}")
                #np.savez(open(outpath, 'wb'),
                         # imsize=(W, H),
                         # keypoints=xys[idxs],
                         # descriptors=desc[idxs],
                         # scores=scores[idxs])
                         #
                # Преобразуем данные в тензоры PyTorch
                size_tensor = torch.tensor([W, H], dtype=torch.int32).unsqueeze(0)  # Добавляем новую ось
                keypointsTorch = torch.from_numpy(xys[idxs]).unsqueeze(0)  # Добавляем ось
                descriptorsTorch = torch.from_numpy(desc[idxs]).unsqueeze(0)  # Добавляем ось
                scoresTorch = torch.from_numpy(scores[idxs]).unsqueeze(0)  # Добавляем ось


                # Формируем словарь
                points = {
                    "keypoints": keypointsTorch,
                    "image_size": size_tensor,
                    "descriptors": descriptorsTorch,
                    "keypoint_scores": scoresTorch
                }
                # print("points", keypointsTorch)
        # print("scores", scores)
        # print("desk", descriptors)

            return points

        import argparse

        # Имитируем переданные аргументы
        args = argparse.Namespace(
            model="r2d2/models/r2d2_WASF_N8_big.pt",
            images=[img],
            tag="r2d2",
            top_k=self.max_keypoints,
            scale_f=1.189207115002721,
            min_size=256,
            max_size=1024,
            min_scale=0,
            max_scale=1,
            reliability_thr=0.7,
            repeatability_thr=0.7,
            gpu=[-1]  # Используем GPU 0
        )

        # Доступ к аргументам
        #print(args.model)  # "path/to/model"
        #print(args.images)  # ["image1.jpg", "image2.jpg"]

        return extract_keypoints(args)
