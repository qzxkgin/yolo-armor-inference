#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""装甲板检测与四角点可视化推理脚本（YOLO26n-pose 权重 best.pt）。

功能
----
1. 加载课程统一下发的 YOLO26n 装甲板四角点模型 ``best.pt``；
2. 读取自行拍摄的装甲板视频，逐帧执行推理；
3. 绘制检测框、类别名、置信度，以及**模型预测的 4 个关键点**
   （关键点直接取自 ``result.keypoints``，不使用检测框的四个角代替）；
4. 保存与原始视频内容一一对应、可正常播放的结果视频，
   同时输出统计信息与若干张代表性结果截图。

关键点可视化说明
----------------
每个目标画出 4 个关键点，并按预测顺序连成四边形，便于看出关键点
属于哪一个目标；四个角点使用不同颜色并标注序号 0~3。

运行示例
--------
::

    # 使用默认路径（weights/best.pt + assets/armor_raw.mp4）
    python infer.py

    # 显式指定权重、输入视频与输出位置
    python infer.py --weights weights/best.pt --source assets/armor_raw.mp4 \\
        --output results/armor_result.mp4 --device 0 --imgsz 640 --conf 0.25

    # 没有 GPU 时用 CPU 推理
    python infer.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------------------------------- #
# 可视化配色：蓝方 B1~B3 使用蓝色系，红方 R1~R3 使用红色系（OpenCV 为 BGR 顺序）
# --------------------------------------------------------------------------- #
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "B1": (255, 64, 0),
    "B2": (255, 144, 0),
    "B3": (255, 208, 0),
    "R1": (0, 64, 255),
    "R2": (0, 144, 255),
    "R3": (0, 208, 255),
}
FALLBACK_COLOR = (0, 255, 0)

# 4 个关键点各自的颜色，用于区分角点顺序（序号由模型定义）
KEYPOINT_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]


# --------------------------------------------------------------------------- #
# 视频写出：优先用 ffmpeg 编码 H.264（体积小、兼容性好），失败则回退 OpenCV
# --------------------------------------------------------------------------- #
class VideoSink:
    """结果视频写出器。

    优先通过管道调用 ffmpeg 编码 H.264（yuv420p，可被浏览器与常见播放器
    直接播放）；若环境中没有 ffmpeg，则回退到 OpenCV 的 ``mp4v`` 编码。
    """

    def __init__(self, path: Path, fps: float, size: tuple[int, int], crf: int = 23):
        self.path = Path(path)
        self.size = size  # (width, height)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.proc: subprocess.Popen | None = None
        self.writer: cv2.VideoWriter | None = None
        self.backend = ""

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            cmd = [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{size[0]}x{size[1]}", "-r", f"{fps:.6f}",
                "-i", "-", "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(self.path),
            ]
            try:
                self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                self.backend = "ffmpeg/libx264"
                return
            except OSError:
                self.proc = None

        # 回退方案：OpenCV 自带编码器
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.path), fourcc, fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"无法创建输出视频：{self.path}")
        self.backend = "opencv/mp4v"

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        if self.proc is not None:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.writer.write(frame)

    def release(self) -> None:
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
        if self.writer is not None:
            self.writer.release()


# --------------------------------------------------------------------------- #
# 绘制
# --------------------------------------------------------------------------- #
def draw_result(frame: np.ndarray, result, scale: float) -> int:
    """在画面中绘制检测框、类别、置信度与 4 个关键点，返回本帧目标数。"""
    boxes = result.boxes
    keypoints = result.keypoints
    if boxes is None or len(boxes) == 0:
        return 0

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    names = result.names

    # 关键点坐标直接来自模型预测结果，形状为 (N, 4, 2)
    kxy = None
    kconf = None
    if keypoints is not None and keypoints.xy is not None and len(keypoints.xy) == len(xyxy):
        kxy = keypoints.xy.cpu().numpy()
        try:
            kconf = None if keypoints.conf is None else keypoints.conf.cpu().numpy()
        except AttributeError:
            kconf = None

    lw = max(2, int(round(2 * scale)))            # 检测框线宽
    kpt_r = max(3, int(round(5 * scale)))         # 关键点半径
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.6 * scale                              # 关键点序号字号
    fs_label = 0.7 * scale                        # 类别标签字号
    lw_text = max(1, int(round(1.5 * scale)))

    for i in range(len(xyxy)):
        name = str(names.get(int(cls[i]), cls[i]))
        color = CLASS_COLORS.get(name, FALLBACK_COLOR)
        x1, y1, x2, y2 = (int(round(v)) for v in xyxy[i])

        # (1) 关键点：先按预测顺序连成四边形，再画点与序号
        if kxy is not None:
            pts = kxy[i]
            poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [poly], True, color, max(1, lw - 1), cv2.LINE_AA)
            for k in range(len(pts)):
                px, py = int(round(float(pts[k][0]))), int(round(float(pts[k][1])))
                kc = KEYPOINT_COLORS[k % len(KEYPOINT_COLORS)]
                cv2.circle(frame, (px, py), kpt_r, kc, -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), kpt_r, (255, 255, 255), max(1, int(scale)), cv2.LINE_AA)
                org = (px + int(7 * scale), py - int(7 * scale))
                cv2.putText(frame, str(k), org, font, fs, (0, 0, 0), int(3 * scale), cv2.LINE_AA)
                cv2.putText(frame, str(k), org, font, fs, kc, lw_text, cv2.LINE_AA)

        # (2) 检测框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, lw, cv2.LINE_AA)

        # (3) 类别 + 置信度标签
        label = f"{name} {conf[i]:.2f}"
        (tw, th), base = cv2.getTextSize(label, font, fs_label, lw_text)
        pad = int(5 * scale)
        if y1 - th - 2 * pad > 0:
            top = y1 - th - 2 * pad
        else:
            top = min(y2 + pad, frame.shape[0] - th - 2 * pad)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 2 * pad, top + th + 2 * pad), color, -1)
        cv2.putText(frame, label, (x1 + pad, top + th + pad // 2), font, fs_label,
                    (255, 255, 255), lw_text, cv2.LINE_AA)

    return len(xyxy)


# --------------------------------------------------------------------------- #
# 命令行参数
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO26n 装甲板四角点模型推理与可视化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", default="weights/best.pt",
                        help="模型权重路径（best.pt）")
    parser.add_argument("--source", default="assets/armor_raw.mp4",
                        help="输入视频路径（自行拍摄的原始视频）")
    parser.add_argument("--output", default="results/armor_result.mp4",
                        help="结果视频保存路径")
    parser.add_argument("--stats", default="results/inference_stats.json",
                        help="推理统计信息保存路径")
    parser.add_argument("--frames-dir", default="docs",
                        help="代表性结果截图保存目录（设为空字符串则不保存）")
    parser.add_argument("--save-frames", type=int, default=6,
                        help="保存代表性截图的数量，0 表示不保存")
    parser.add_argument("--device", default="auto",
                        help="推理设备：auto / 0 / cpu")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理输入尺寸，须为 32 的整数倍；默认 640，与模型训练尺寸一致")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    parser.add_argument("--max-det", type=int, default=100, help="单帧最大目标数")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="只处理前 N 帧（0 表示处理全部）")
    parser.add_argument("--crf", type=int, default=23,
                        help="ffmpeg 编码质量（数值越小越清晰、文件越大）")
    return parser.parse_args(argv)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    weights = Path(args.weights)
    source = Path(args.source)
    if not weights.is_file():
        print(f"[错误] 找不到模型权重：{weights}\n"
              f"       请按 README 说明将 best.pt 放到该位置。", file=sys.stderr)
        return 2
    if not source.is_file():
        print(f"[错误] 找不到输入视频：{source}", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    print(f"[信息] 权重      : {weights}")
    print(f"[信息] 输入视频  : {source}")
    print(f"[信息] 推理设备  : {device}")
    print(f"[信息] imgsz/conf: {args.imgsz} / {args.conf}")

    model = YOLO(str(weights))
    task = getattr(model, "task", "unknown")
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    kpt_shape = None
    try:
        kpt_shape = model.model.yaml.get("kpt_shape")
    except Exception:
        pass
    print(f"[信息] 模型任务  : {task}，类别 {list(names.values())}，关键点形状 {kpt_shape}")
    if task != "pose" and kpt_shape is None:
        print("[警告] 该权重似乎不是关键点（pose）模型，可能无法输出 4 个关键点。", file=sys.stderr)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"[错误] 无法打开视频：{source}", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames > 0:
        total = min(total, args.max_frames)
    print(f"[信息] 视频信息  : {width}x{height} @ {fps:.2f} fps，共 {total} 帧 "
          f"(约 {total / fps:.2f} 秒)")

    sink = VideoSink(Path(args.output), fps, (width, height), crf=args.crf)
    print(f"[信息] 结果视频  : {Path(args.output).resolve()}（编码 {sink.backend}）")

    scale = max(1.0, min(width, height) / 1080.0)

    class_count: dict[str, int] = defaultdict(int)
    class_conf_sum: dict[str, float] = defaultdict(float)
    frames_with_det = 0
    total_boxes = 0
    kpt_conf_sum = 0.0
    kpt_conf_num = 0
    top_frames: list[tuple[float, int, np.ndarray]] = []

    t0 = time.perf_counter()
    idx = 0
    while idx < total:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=device,
            verbose=False,
        )[0]

        n = draw_result(frame, result, scale)

        if n > 0:
            frames_with_det += 1
            total_boxes += n
            confs = result.boxes.conf.cpu().numpy()
            for c, p in zip(result.boxes.cls.cpu().numpy().astype(int), confs):
                nm = str(names.get(int(c), c))
                class_count[nm] += 1
                class_conf_sum[nm] += float(p)
            if result.keypoints is not None:
                try:
                    kc = result.keypoints.conf
                    if kc is not None:
                        kpt_conf_sum += float(kc.cpu().numpy().sum())
                        kpt_conf_num += int(kc.numel())
                except AttributeError:
                    pass

            if args.save_frames > 0 and args.frames_dir:
                score = n + float(np.mean(confs))
                small = cv2.resize(frame, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
                top_frames.append((score, idx, small))
                top_frames.sort(key=lambda item: -item[0])
                del top_frames[args.save_frames:]

        sink.write(frame)
        idx += 1
        if idx % 50 == 0 or idx == total:
            print(f"\r[进度] {idx}/{total} 帧，检出目标 {total_boxes} 个，"
                  f"用时 {time.perf_counter() - t0:.1f}s", end="", flush=True)

    print()
    cap.release()
    sink.release()
    elapsed = time.perf_counter() - t0

    # ---------------- 保存代表性截图 ----------------
    saved_shots: list[str] = []
    if top_frames and args.frames_dir:
        out_dir = Path(args.frames_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for rank, (score, frame_idx, img) in enumerate(top_frames, start=1):
            shot = out_dir / f"sample_{rank:02d}_frame{frame_idx:04d}.jpg"
            cv2.imwrite(str(shot), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved_shots.append(str(shot))
        print(f"[信息] 已保存 {len(saved_shots)} 张代表性截图到 {out_dir}/")

    # ---------------- 统计信息 ----------------
    stats = {
        "weights": str(weights),
        "source": str(source),
        "output_video": str(Path(args.output).resolve()),
        "device": device,
        "imgsz": args.imgsz,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "video": {
            "width": width,
            "height": height,
            "fps": round(fps, 3),
            "total_frames": total,
            "duration_sec": round(total / fps, 3),
        },
        "result": {
            "frames_processed": idx,
            "frames_with_detection": frames_with_det,
            "detection_frame_rate": round(frames_with_det / idx, 4) if idx else 0.0,
            "total_boxes": total_boxes,
            "mean_confidence": round(
                sum(class_conf_sum.values()) / total_boxes, 4) if total_boxes else 0.0,
            "mean_keypoint_confidence": round(kpt_conf_sum / kpt_conf_num, 4) if kpt_conf_num else None,
            "per_class": {
                nm: {
                    "count": class_count[nm],
                    "mean_confidence": round(class_conf_sum[nm] / class_count[nm], 4),
                }
                for nm in sorted(class_count)
            },
            "elapsed_sec": round(elapsed, 2),
            "inference_fps": round(idx / elapsed, 2) if elapsed else 0.0,
        },
        "sample_frames": saved_shots,
    }

    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n================ 推理结果汇总 ================")
    print(f"处理帧数        : {idx}")
    print(f"含目标帧数      : {frames_with_det}（{frames_with_det / idx * 100:.1f}%）" if idx else "")
    print(f"目标总数        : {total_boxes}")
    for nm in sorted(class_count):
        print(f"  {nm}: {class_count[nm]} 个，平均置信度 {class_conf_sum[nm] / class_count[nm]:.3f}")
    if kpt_conf_num:
        print(f"关键点平均置信度: {kpt_conf_sum / kpt_conf_num:.3f}")
    print(f"总耗时          : {elapsed:.2f} s（平均 {idx / elapsed:.2f} FPS）")
    print(f"结果视频        : {Path(args.output).resolve()}")
    print(f"统计文件        : {stats_path.resolve()}")
    print("=============================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
