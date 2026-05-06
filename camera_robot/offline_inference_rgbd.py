#!/usr/bin/env python3
"""
offline_inference_rgbd.py

对采集好的 RGB + Depth PNG 样本做离线推理，可视化结果保存到一个新目录。
默认使用：
  - RGB:  sample_xxxx_rgb.png
  - Depth: sample_xxxx_depth_vis.png （16bit PNG, 0~65535 对应 0~8m）

用法示例：

  python3 offline_inference_rgbd.py \
      --data_dir /home/skj/my_dataset \
      --model_path /home/skj/camera_robot_ws/src/camera_robot/camera_robot/resource/weights/best_model_phi0.pth \
      --use_depth \
      --device cuda
"""

import os
import glob
import argparse

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# ========= 从本包内 efficientpose_lib 导入 =========
from efficientpose_lib.models.efficientpose import EfficientPose
from efficientpose_lib.config import Config


# ---------- rot6d → 旋转矩阵 ----------
def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    a1 = rot6d[:3]
    a2 = rot6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def rotation_to_euler_deg(R):
    r = Rotation.from_matrix(R)
    rpy = r.as_euler("zyx", degrees=True)
    return rpy[2], rpy[1], rpy[0]  # roll, pitch, yaw


# ---------- 深度 PNG 读取 ----------
def load_depth_from_png(png_path: str, max_m: float = 8.0) -> np.ndarray:
    """
    读取 16 位深度 PNG（0~65535 对应 0~max_m 米），返回 float32 米单位。
    """
    depth_u16 = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if depth_u16 is None:
        raise FileNotFoundError(f"cannot read depth png: {png_path}")
    if depth_u16.dtype != np.uint16:
        raise ValueError(f"depth png must be uint16, got {depth_u16.dtype}")
    depth = depth_u16.astype(np.float32) / 65535.0 * max_m
    return depth


# ---------- 预处理 ----------
def preprocess_rgb(rgb_bgr, image_size, mean, std):
    h, w = rgb_bgr.shape[:2]
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size))
    t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    t = (t - mean) / std
    return t.unsqueeze(0), (h, w)


def preprocess_rgbd(rgb_bgr, depth_m, image_size, mean, std):
    h, w = rgb_bgr.shape[:2]
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    rgb_r = cv2.resize(rgb, (image_size, image_size))

    depth = depth_m.astype(np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth[(depth < 0.1) | (depth > 8.0)] = 0.0

    depth_r = cv2.resize(depth, (image_size, image_size),
                         interpolation=cv2.INTER_NEAREST)
    depth_blur = cv2.GaussianBlur(depth_r, (3, 3), 0)
    depth_r[depth_r == 0] = depth_blur[depth_r == 0]

    depth_norm = depth_r / 8.0
    depth_norm[depth_r == 0] = 0.0

    t_rgb = torch.from_numpy(rgb_r).permute(2, 0, 1).float() / 255.0
    t_rgb = (t_rgb - mean) / std
    t_d = torch.from_numpy(depth_norm).unsqueeze(0).float()

    t = torch.cat([t_rgb, t_d], dim=0)
    return t.unsqueeze(0), (h, w)


# ---------- 可视化 ----------
def draw_result(img_bgr, bbox, rotation, translation, conf):
    x1, y1, x2, y2 = bbox
    roll, pitch, yaw = rotation_to_euler_deg(rotation)
    dist = float(np.linalg.norm(translation))
    tx, ty, tz = translation
    h, w = img_bgr.shape[:2]

    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = f"{dist:.2f}m | conf={conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    lx = max(x1, 0)
    ly = max(y1 - th - 8, th + 4)
    cv2.rectangle(img_bgr,
                  (lx, ly - th - 4), (lx + tw + 6, ly + 2),
                  (0, 200, 0), -1)
    cv2.putText(img_bgr, label, (lx + 3, ly - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    txt2 = (f"Tx:{tx:+.3f} Ty:{ty:+.3f} Tz:{tz:+.3f} | "
            f"R:{roll:+.1f} P:{pitch:+.1f} Y:{yaw:+.1f}")
    cv2.putText(img_bgr, txt2, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return img_bgr


# ---------- 主逻辑 ----------
def run_inference_on_folder(
    data_dir,
    model_path,
    phi=0,
    use_depth=True,
    out_dir=None,
    device="cuda",
    conf_thresh=0.3,
    depth_png_suffix="_depth_vis.png",
    depth_max_m=8.0,
):
    """
    depth_png_suffix: 深度PNG文件的后缀，例如 '_depth_vis.png'
    """
    if out_dir is None:
        out_dir = os.path.join(data_dir, "inference_vis")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(device)
    cfg = Config()
    cfg.phi = phi
    image_size = cfg.compound_coef[phi]["resolution"]

    print(f"[INFO] loading model from: {model_path}")
    model = EfficientPose(phi=phi, num_classes=1)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    rgb_files = sorted(glob.glob(os.path.join(data_dir, "*_rgb.png")))
    if not rgb_files:
        print(f"[WARN] no *_rgb.png found in {data_dir}")
        return

    print(f"[INFO] found {len(rgb_files)} samples in {data_dir}")

    for rgb_path in rgb_files:
        prefix = rgb_path.replace("_rgb.png", "")
        depth_png = prefix + depth_png_suffix
        base_name = os.path.basename(prefix)

        if use_depth and (not os.path.exists(depth_png)):
            print(f"[WARN] depth png missing for {base_name}, skip")
            continue

        bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] failed to read {rgb_path}, skip")
            continue

        if use_depth:
            depth = load_depth_from_png(depth_png, max_m=depth_max_m)
            t, (orig_h, orig_w) = preprocess_rgbd(
                bgr, depth, image_size, mean, std)
        else:
            t, (orig_h, orig_w) = preprocess_rgb(
                bgr, image_size, mean, std)

        t = t.to(device)

        with torch.no_grad():
            outputs = model(t)

        class_scores = outputs["class"][0, :, 0].cpu().numpy()
        best_idx = int(np.argmax(class_scores))
        best_conf = float(class_scores[best_idx])

        if best_conf < conf_thresh:
            print(f"[INFO] {base_name}: no detection (conf={best_conf:.2f})")
            vis = bgr.copy()
        else:
            rot6d = outputs["rotation"][0, best_idx].cpu().numpy()
            R = rot6d_to_matrix(rot6d)
            t_vec = outputs["translation"][0, best_idx].cpu().numpy()

            raw_bbox = outputs["bbox"][0, best_idx].cpu().numpy()
            x1 = int(np.clip(raw_bbox[0] * orig_w, 0, orig_w - 1))
            y1 = int(np.clip(raw_bbox[1] * orig_h, 0, orig_h - 1))
            x2 = int(np.clip(raw_bbox[2] * orig_w, 0, orig_w - 1))
            y2 = int(np.clip(raw_bbox[3] * orig_h, 0, orig_h - 1))

            vis = draw_result(
                bgr.copy(), (x1, y1, x2, y2), R, t_vec, best_conf)
            print(
                f"[INFO] {base_name}: conf={best_conf:.3f} "
                f"Tx={t_vec[0]:+.3f} Ty={t_vec[1]:+.3f} Tz={t_vec[2]:+.3f} "
                f"dist={np.linalg.norm(t_vec):.3f} m"
            )

        out_path = os.path.join(out_dir, base_name + "_vis.png")
        cv2.imwrite(out_path, vis)

    print(f"[INFO] done. results saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/skj/my_dataset",
        help="目录，里面是 sample_xxxx_rgb.png / sample_xxxx_depth_vis.png",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/skj/camera_robot_ws/src/camera_robot/"
                "camera_robot/resource/weights/best_model_phi0.pth",
    )
    parser.add_argument("--phi", type=int, default=0)
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conf_thresh", type=float, default=0.3)
    parser.add_argument(
        "--depth_png_suffix",
        type=str,
        default="_depth_vis.png",
        help="深度PNG文件后缀（默认 sample_xxxx_depth_vis.png）",
    )
    parser.add_argument(
        "--depth_max_m",
        type=float,
        default=8.0,
        help="保存PNG时对应的最大深度（米）",
    )
    args = parser.parse_args()

    run_inference_on_folder(
        data_dir=args.data_dir,
        model_path=args.model_path,
        phi=args.phi,
        use_depth=args.use_depth,
        device=args.device,
        conf_thresh=args.conf_thresh,
        depth_png_suffix=args.depth_png_suffix,
        depth_max_m=args.depth_max_m,
    )


if __name__ == "__main__":
    main()