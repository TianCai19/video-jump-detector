#!/usr/bin/env python3
"""
视频跳转检测器 - 带实时进度显示
使用"去抖 + 帧差能量 + 异常值检测"快速找出视频中的跳转点
"""

import cv2
import numpy as np
import json
import os
import time
from pathlib import Path
from tqdm import tqdm

def robust_zscore(x, eps=1e-9):
    """
    计算鲁棒Z分数（使用MAD）
    """
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    z = 0.6745 * (x - med) / mad
    return z, med, mad

def detect_jumps(video_path, z_th=6.0, min_gap=8, use_align=True, max_frames=None):
    """
    检测视频中的跳转点

    Args:
        video_path: 视频文件路径
        z_th: Z分数阈值（默认6.0）
        min_gap: 最小间隔帧数（防止连续检测）
        use_align: 是否使用全局对齐（抵消抖动）
        max_frames: 最大处理帧数（None表示处理全部）

    Returns:
        result: 跳转点列表，每个包含frame, time_sec, score, z
        scores: 所有帧的变化分数
        z: 所有帧的Z分数
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # 获取总帧数用于进度条
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames and max_frames < total_frames:
        total_frames = max_frames

    print(f"视频信息: FPS={fps:.2f}, 总帧数={total_frames}")

    # ORB特征检测器
    orb = cv2.ORB_create(800)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    scores = []
    prev_gray = None
    prev_kp, prev_des = None, None

    idx = 0

    # 创建进度条
    pbar = tqdm(total=total_frames, desc="🔍 检测跳转点", unit="帧", ncols=100)

    start_time = time.time()
    last_update = start_time

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if max_frames and idx > max_frames:
            break

        # 转换为灰度图并模糊
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_gray is None:
            prev_gray = gray
            prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)
            scores.append(0.0)
            pbar.update(1)
            continue

        cur = gray

        # 1) 全局对齐（抵消抖动）
        if use_align and prev_des is not None:
            kp, des = orb.detectAndCompute(cur, None)
            if des is not None and len(des) > 20 and prev_des is not None and len(prev_des) > 20:
                matches = bf.match(prev_des, des)
                matches = sorted(matches, key=lambda m: m.distance)[:120]

                if len(matches) >= 12:
                    pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                    pts_cur  = np.float32([kp[m.trainIdx].pt for m in matches])

                    # 仿射变换
                    M, inliers = cv2.estimateAffinePartial2D(
                        pts_cur, pts_prev, method=cv2.RANSAC, ransacReprojThreshold=3.0
                    )
                    if M is not None:
                        cur = cv2.warpAffine(cur, M, (cur.shape[1], cur.shape[0]),
                                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            prev_kp, prev_des = kp, des

        # 2) 帧差分数：变化像素占比
        diff = cv2.absdiff(cur, prev_gray)
        _, bw = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)  # 阈值可调
        score = float(bw.mean())  # 0~255
        scores.append(score)

        prev_gray = cur

        # 更新进度条
        pbar.update(1)

        # 每秒更新一次详细信息
        current_time = time.time()
        if current_time - last_update >= 1.0:
            elapsed = current_time - start_time
            fps_current = idx / elapsed if elapsed > 0 else 0
            eta = (total_frames - idx) / fps_current if fps_current > 0 else 0
            percent = (idx / total_frames) * 100

            pbar.set_postfix({
                'FPS': f"{fps_current:.1f}",
                'ETA': f"{eta:.1f}s",
                '进度': f"{percent:.1f}%",
                '帧': f"{idx}/{total_frames}"
            })
            last_update = current_time

    pbar.close()
    cap.release()

    # 3) 鲁棒异常检测
    print("\n📊 分析变化分数...")
    z, med, mad = robust_zscore(scores)
    candidates = np.where(z > z_th)[0].tolist()

    # 4) 去重：只保留相隔min_gap帧以上的点
    jumps = []
    last = -10**9
    for t in candidates:
        if t - last >= min_gap:
            jumps.append(t)
            last = t

    # 输出：帧号 + 时间点（秒）
    result = [{"frame": t, "time_sec": t / fps, "score": scores[t], "z": float(z[t])} for t in jumps]
    return result, scores, z

def save_keyframes(video_path, jump_frames, output_dir="keyframes"):
    """
    保存关键帧图像
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)

    print(f"\n💾 保存关键帧图像到目录: {output_dir}")

    # 创建进度条
    pbar = tqdm(total=len(jump_frames), desc="📸 保存关键帧", unit="张", ncols=80)

    for i, jump in enumerate(jump_frames):
        frame_idx = jump['frame']
        time_sec = jump['time_sec']

        # 跳转到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            # 生成文件名：jump_序号_时间戳_帧号.jpg
            filename = f"jump_{i+1:03d}_{time_sec:.2f}s_frame{frame_idx:06d}.jpg"
            filepath = os.path.join(output_dir, filename)

            # 保存图像
            cv2.imwrite(filepath, frame)
            pbar.set_postfix({'文件': filename[:30]})

        pbar.update(1)

    pbar.close()
    cap.release()

def main():
    video_path = "1ad0f046c0dd45f09797593a9db7a294.mp4"

    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"❌ 错误: 视频文件不存在: {video_path}")
        return

    print("="*70)
    print("🎬 视频跳转检测器 - 带实时进度显示")
    print("="*70)
    print(f"📹 处理视频: {video_path}")

    # 获取视频信息
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    print(f"📊 视频信息: FPS={fps:.2f}, 总帧数={total_frames}, 时长={duration:.2f}秒")

    # 计算要处理的帧数（取前1分钟）
    max_frames = int(fps * 60)  # 前60秒
    if max_frames > total_frames:
        max_frames = total_frames

    print(f"⏱️  处理前 {max_frames} 帧 (前 {max_frames/fps:.1f} 秒)")

    # 执行跳转检测
    print("\n🚀 开始检测跳转点...")
    start_time = time.time()

    result, scores, z = detect_jumps(
        video_path,
        z_th=6.0,      # Z分数阈值
        min_gap=8,     # 最小间隔8帧
        use_align=True,  # 启用全局对齐
        max_frames=max_frames
    )

    elapsed_time = time.time() - start_time
    print(f"\n⏱️  检测完成! 耗时 {elapsed_time:.2f} 秒")

    print(f"\n🎯 检测结果: 发现 {len(result)} 个跳转点:")
    print("-" * 70)

    if result:
        for i, jump in enumerate(result):
            print(f"{i+1:2d}. 帧{jump['frame']:6d} | "
                  f"⏰ 时间{jump['time_sec']:6.2f}s | "
                  f"📈 分数{jump['score']:6.2f} | "
                  f"🔢 Z值{jump['z']:6.2f}")
    else:
        print("❌ 未检测到任何跳转点")

    # 保存JSON结果
    output_data = {
        "video_file": video_path,
        "video_info": {
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": duration,
            "processed_frames": max_frames
        },
        "detection_params": {
            "z_threshold": 6.0,
            "min_gap_frames": 8,
            "use_alignment": True
        },
        "jumps": result,
        "statistics": {
            "total_jumps": len(result),
            "avg_score": float(np.mean([j['score'] for j in result])) if result else 0,
            "avg_z": float(np.mean([j['z'] for j in result])) if result else 0,
            "processing_time_sec": elapsed_time
        }
    }

    json_path = "jump_detection_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ JSON结果已保存到: {json_path}")

    # 保存关键帧图像
    if result:
        save_keyframes(video_path, result, "keyframes")
        print("\n✅ 所有关键帧已保存完成!")
    else:
        print("\n⚠️  未检测到跳转点，跳过关键帧保存")

    print("\n" + "="*70)
    print("✨ 处理完成!")
    print("="*70)

if __name__ == "__main__":
    main()
