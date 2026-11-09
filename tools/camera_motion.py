#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
H3 运镜迁移 · 参考片运镜测量器  v4

把一段参考视频的镜头运动，测成可复现参数 + H3 官方相机词表的英文句式。
这是「学优秀运镜 → 沉淀成可复用 prompt」闭环里的测量环节。

只在打包 Python 里跑：
  D:\\AI\\A-ComfyUI-GF\\ComfyUI_windows_portable\\python_embeded\\python.exe

用法：
  py camera_motion.py --video ref.mp4 --out DIR [--sheet] [--emit-prompt]
  py camera_motion.py --video ref.mp4 --check-prompt capsule.md
  py camera_motion.py --selftest          # 合成数据自检，改代码后必跑

================================ 演进史（都是为了修错，别退回去）================================
v1 稠密光流 Farneback + 最小二乘拟合相似变换。
   错 1：滚转有 0.04°/帧系统偏置 → 124 帧攒出「滚了 4.8°」假阳性。
   错 2：合成 3.0° 只测出 1.27°（低估 2.4×），2% 缩放还漏成 0.49° 旋转。
v2 加了符号一致性 + t 检验压住假阳性。但低估没修，真滚转也偏小。
v3 换稀疏角点 LK + RANSAC 拟合。合成自检精度到了旋转 0.012°、缩放 0.0000。
   但**真片上仍然错**：L2T_S2 那条，它的 prompt 明确只写了
   "dollies in slowly, tilting up slightly"，实测却报 +44% 缩放、14% 横移、1.4° 滚转。
v4（本版）找到根因：**人物在画面里长大，把全局拟合带跑了**。
   同一对首尾帧，分区测量结果天差地别：
       全画面 scale=0.94 rot=-4.28°      ← 被主体污染，连方向都是反的
       上带(石拱) 1.098 / -0.21°          ← 背景，稳定
       左柱       1.118 / -0.21°
       右柱       1.101 / -0.16°
       下带(地面) 0.368 / -10.82°         ← 运动模糊，直接崩
   → 结论：**必须只测背景**。v4 取画面四周若干背景区分别测，
     再对结果做 MAD 剔除 + 中位数聚合；区域间不一致就明确报低置信度，而不是硬给个数。

================================ 残留的物理歧义（不是 bug，别硬修）================================
· Pan 与 Truck、Tilt 与 Pedestal 在 2D 里都是「内容整体平移」，物理不可分。
  脚本同时给两个候选标签，按原片二选一。
· 主体占满画面、背景不可见时（特写），测出来的是主体运动，不是相机运动。
  这种镜头 v4 会报 「背景不足，不可判」，宁可不给结论。
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

try:  # Windows 管道默认 GBK，中文报告会炸
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------- 分级阈值
# H3 官方词表只有 small/large amplitude、slow/fast speed 两档修饰，下面是量化门限。
# 单位一律「画面宽/高的比例」，与分辨率无关，可跨片比较。
AMP_STATIC = 0.045
AMP_LARGE = 0.22
SPD_SLOW = 0.05
SPD_FAST = 0.18
ZOOM_EPS = 0.035
ROT_EPS_DEG = 2.5
PARALLAX_EPS = 0.45
CUT_FLOOR = 14.0
CUT_PERSIST = 0.55

# 背景区定义：画面四周的带子。刻意避开中心（主体通常在中景/近景）。
BG_REGIONS = {
    "上带": (0.00, 0.00, 1.00, 0.20),
    "下带": (0.00, 0.80, 1.00, 1.00),
    "左带": (0.00, 0.15, 0.18, 0.85),
    "右带": (0.82, 0.15, 1.00, 0.85),
}
MIN_PTS_PER_REGION = 18
MIN_REGIONS_OK = 2
AGREE_SCALE_TOL = 0.05      # 各背景区缩放的中位绝对偏差超过 5% 视为不一致
AGREE_ROT_TOL_DEG = 1.0     # 各背景区滚转的中位绝对偏差超过 1° 视为不一致


# ---------------------------------------------------------------- 解码
def load_gray_frames(path, maxw=480):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("打不开视频: %s" % path)
    sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if maxw and fr.shape[1] > maxw:
            sc = maxw / float(fr.shape[1])
            fr = cv2.resize(fr, (maxw, max(2, int(round(fr.shape[0] * sc)))),
                            interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        raise SystemExit("视频解不出任何帧: %s" % path)
    return frames, fps, (sw, sh)


def _norm_small(g, size=(64, 36)):
    s = cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    return (s - s.mean()) / (s.std() + 1e-6)


def struct_diff(a, b):
    return float(np.abs(_norm_small(a) - _norm_small(b)).mean()) * 100.0


# ---------------------------------------------------------------- 切镜（带持续性校验）
def detect_cuts(frames):
    n = len(frames)
    diffs = [0.0] * n
    for i in range(1, n):
        diffs[i] = struct_diff(frames[i - 1], frames[i])
    if n < 5:
        return [], diffs

    body = diffs[1:]
    med = float(np.median(body))
    thr = max(CUT_FLOOR, med * 3.2)
    cands = [i for i in range(1, n) if diffs[i] > thr]
    if not cands:
        return [], diffs

    # ---- 先剔除「运动密集段」，再做持续性校验 ----
    # 顺序很重要（踩过）：如果先做持续性校验（间隔 >=10 帧去重），
    # 甩镜那一串候选会被压成稀疏的 3 个点，密度反而看不出来，过滤就失效了。
    # 实测原始候选的密度（±10 帧窗口，不含自己）：
    #   G_5005（甩镜）  候选 28 个连成一片，每个点密度 15~20
    #   00153（真硬切）f82 簇 4 个、f98 簇 3 个，两点相距 16 帧
    # 所以窗口要**小于两簇间距**（16），否则真假切点会被连成一片。
    WINDOW = 10
    DENSE = 4      # ±10 帧内超过这么多个同伴 = 一段持续运动

    def dense(c):
        return sum(1 for o in cands if abs(o - c) <= WINDOW) - 1

    quiet = [c for c in cands if dense(c) <= DENSE]
    if not quiet:
        # 整段都在动（纯甩镜），没有任何一处安静 → 没有任何切镜
        return [], diffs

    cuts = []
    for c in quiet:
        probe = min(c + 10, n - 1)
        if probe <= c:
            keep = True
        else:
            persist = struct_diff(frames[c - 1], frames[probe])
            keep = persist >= CUT_PERSIST * diffs[c] or persist >= thr
        if keep and (not cuts or c - cuts[-1] >= 10):
            cuts.append(c)
    return cuts, diffs


# ---------------------------------------------------------------- 单区域相似变换
def fit_region(a, b, box, maxpts=700):
    """
    在指定矩形区域内**检出**角点，但在整帧上跟踪。

    ⚠️ 这一步踩过坑，别改回去：最初是「区域裁剪 + 区域内跟踪」，
    结果大幅平移时前缘点跑出区域被裁掉，只剩边界附近的点，测出的位移严重偏小
    （下移 45px 只测出 12.8px）。区域只该用来**选谁参与拟合**，
    不该限制跟踪的范围。
    """
    h, w = a.shape
    x0 = int(round(box[0] * w)); x1 = int(round(box[2] * w))
    y0 = int(round(box[1] * h)); y1 = int(round(box[3] * h))
    if (x1 - x0) < 24 or (y1 - y0) < 24:
        return None
    mask = np.zeros((h, w), np.uint8)
    mask[y0:y1, x0:x1] = 255
    pts = cv2.goodFeaturesToTrack(a, maxCorners=maxpts, qualityLevel=0.01,
                                  minDistance=7, blockSize=7, mask=mask)
    if pts is None or len(pts) < MIN_PTS_PER_REGION:
        return None
    # 整帧跟踪
    trk, st, err = cv2.calcOpticalFlowPyrLK(
        a, b, pts, None, winSize=(31, 31), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
    if trk is None or st is None:
        return None
    p1 = trk.reshape(-1, 2)
    ok = (st.ravel() == 1) & (err.ravel() < 30.0)
    ok &= (p1[:, 0] > 1) & (p1[:, 0] < w - 2) & (p1[:, 1] > 1) & (p1[:, 1] < h - 2)
    if ok.sum() < MIN_PTS_PER_REGION:
        return None
    p0, p1 = pts.reshape(-1, 2)[ok], p1[ok]
    M, inl = cv2.estimateAffinePartial2D(
        p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.0,
        maxIters=5000, confidence=0.999)
    if M is None:
        return None
    aa, bb = float(M[0, 0]), float(M[1, 0])
    scale = math.hypot(aa, bb)
    rot = math.degrees(math.atan2(bb, aa))     # 屏幕顺时针为正

    # ⚠️ 别直接把 M[:,2] 当平移用（踩过）：
    # 绕画面中心缩放 s 倍时，仿射矩阵的平移分量是 c - s·c ≠ 0，
    # 于是一个纯推近会被错报成「同时还在横移」。必须把平移取在**画面中心**，
    # 即中心点的实际位移 (M·c + t) - c，这才是相机真正移动了多少。
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    cdx = (aa * cx + M[0, 1] * cy + float(M[0, 2])) - cx
    cdy = (bb * cx + M[1, 1] * cy + float(M[1, 2])) - cy

    pred = (M[:, :2] @ p0.T).T + M[:, 2]
    resid = np.linalg.norm(pred - p1, axis=1)
    disp = np.linalg.norm(p1 - p0, axis=1)
    return {
        "scale": scale, "rot": rot,
        "tx": float(cdx), "ty": float(cdy),
        "pts": int(ok.sum()),
        "parallax": float(np.median(resid)) / (float(np.median(disp)) + 1e-6),
    }


def aggregate_regions(a, b):
    """
    逐背景区测量 → MAD 剔除离群区 → 中位数。
    返回聚合结果 + 每个区的原始读数（报告里全列出来，便于人核对）。
    """
    raw = {}
    for name, box in BG_REGIONS.items():
        r = fit_region(a, b, box)
        if r is not None:
            raw[name] = r
    if len(raw) < MIN_REGIONS_OK:
        return {"ok": False, "reason": "背景可用区域不足（<2）", "raw": raw}

    names = list(raw.keys())
    scales = np.array([raw[n]["scale"] for n in names])
    rots = np.array([raw[n]["rot"] for n in names])

    def mad_keep(v):
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) + 1e-9
        return np.abs(v - med) <= 3.0 * 1.4826 * mad

    keep = mad_keep(scales) & mad_keep(rots)
    if keep.sum() < MIN_REGIONS_OK:
        keep = np.ones(len(names), bool)   # 全不一致时按全部取中位数，但下面会报低置信

    sel = [names[i] for i in range(len(names)) if keep[i]]
    scale = float(np.median([raw[n]["scale"] for n in sel]))
    rot = float(np.median([raw[n]["rot"] for n in sel]))
    tx = float(np.median([raw[n]["tx"] for n in sel]))
    ty = float(np.median([raw[n]["ty"] for n in sel]))
    par = float(np.median([raw[n]["parallax"] for n in sel]))

    spread_scale = float(scales.max() - scales.min())
    spread_rot = float(rots.max() - rots.min())
    # ⚠️ 用 MAD（稳健离散度）判一致性，不要用 max-min（踩过）：
    # 真实片子各背景区本来就有视差差异（上带 1.024 / 下带 1.093 / 右带 1.038 都合理），
    # max-min 会被单个坏区拉爆，把正常镜头判成「低置信」。
    mad_scale = float(np.median(np.abs(scales - np.median(scales))))
    mad_rot = float(np.median(np.abs(rots - np.median(rots))))
    agree = mad_scale <= AGREE_SCALE_TOL and mad_rot <= AGREE_ROT_TOL_DEG
    return {
        "ok": True, "scale": scale, "rot": rot, "tx": tx, "ty": ty,
        "parallax": par, "regions_used": sel, "regions_all": names,
        "spread_scale": spread_scale, "spread_rot": spread_rot,
        "mad_scale": mad_scale, "mad_rot": mad_rot,
        "agree": agree, "raw": raw,
        "confidence": ("高" if agree and len(sel) >= 3 else
                       "中" if agree else "低（各区不一致，读数仅供参考）"),
    }


# ---------------------------------------------------------------- 清晰度 / 运动模糊
# 为什么需要这个模块（踩过的坑，别删）：
#   早期版本只用「几何位移」判所有运镜，结果把 whip pan 判成「低置信度/无效」。
#   实际上甩镜的特征不是位移，而是**画面糊成横向光带**——中间帧根本拟合不出运动。
#   工具报「低置信度」恰恰是它生效的签名，不是失败。
#   => 几何运动与画面模糊是**两类独立信号**，必须分开测。
BLUR_DIP_RATIO = 0.45     # 单帧「糊」的门限（相对参考清晰度）
BLUR_AREA_RATIO = 0.15    # 低于门限的帧占比超过这个值，才算「持续糊过」
BLUR_MIN_FRAMES = 6
MOTION_BLUR_LABELS = ("Whip Pan", "Crash Zoom", "Shake")


def sharpness_curve(frames, s, e, maxw=None):
    """逐帧 Laplacian 方差（越大越清晰）。这是运动模糊最便宜的代理指标。"""
    out = []
    for i in range(s, e + 1):
        g = frames[i]
        if maxw and g.shape[1] > maxw:
            sc = maxw / float(g.shape[1])
            g = cv2.resize(g, (maxw, max(2, int(round(g.shape[0] * sc)))),
                           interpolation=cv2.INTER_AREA)
        out.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    return out


def analyze_blur(sharp, fps):
    """
    检测强运动模糊（甩镜 / 急变焦 / 强抖动的签名）。

    ⚠️ 这里的判据是实测改出来的，别按直觉改回去：
    最初假设「清晰→糊→恢复」的 V 形，结果 G_5005（已抽帧确认是完整甩镜）
    实测是**一直糊到结尾**——甩完之后画面停在运动模糊上，并没有恢复清晰。
    真实曲线（相对清晰度）：
        f0 1.92 → f40 0.56 → f56 0.17（谷底）→ f120 0.42
    所以判据必须是「糊过」而不是「恢复」。

    对比参考：静止画面几乎全程 1.00（A_1001 波动 <±3%）。

    返回：
      has_blur     是否出现可信的强模糊段
      dip_ratio    最低清晰度 / 参考清晰度（越小越糊）
      blur_ratio   尾段清晰度 / 参考（判断有没有恢复）
      profile      sharp-dip-recover / sharp-dip-hold / none
    """
    n = len(sharp)
    if n < BLUR_MIN_FRAMES + 4:
        return {"has_blur": False, "reason": "too_short", "n": n}

    # 参考清晰度取 **90 分位**（最清晰的那批帧），不是首尾中位。
    # ⚠️ 这里踩过坑：最初用「首尾各 20% 的中位」当基准，
    #    但甩镜往往甩完就停在糊画面上（G_5005 尾段只有 0.42），
    #    糊的尾段把基准拉低，于是整段看起来都"正常"——检测直接失效。
    #    用高分位当基准，才等于「这段视频原本能有多清晰」。
    arr = np.asarray(sharp, dtype=np.float64)
    ref = float(np.percentile(arr, 90)) or 1e-6
    rel = arr / ref

    # 主判据用「低于门限的帧占比」，不用「连续段长度」。
    # 实测：连续段会被谷底的微小抖动切断（G_5005 只连成 2 帧），
    # 但整段里 66% 的帧其实都是糊的——按占比判才抓得住。
    blur_mask = rel < BLUR_DIP_RATIO
    area = float(blur_mask.mean())
    lo = int(np.argmin(rel))
    dip = float(rel[lo])
    tail = float(np.median(rel[-max(2, int(n * 0.2)):]))

    has = area >= BLUR_AREA_RATIO and int(blur_mask.sum()) >= BLUR_MIN_FRAMES

    # 形状：甩完是否恢复清晰
    if not has:
        profile = "none"
    elif tail >= 0.75:
        profile = "sharp-dip-recover"
    else:
        profile = "sharp-dip-hold"

    # 糊得最久的那一段（仅用于报告定位）
    idx = np.where(blur_mask)[0]
    seg_start = int(idx[0]) if len(idx) else None
    seg_end = int(idx[-1]) if len(idx) else None

    return {
        "has_blur": bool(has),
        "n": n,
        "blur_area": round(area, 3),
        "blur_frames": int(blur_mask.sum()),
        "blur_start": seg_start, "blur_end": seg_end,
        "blur_seconds": round(area * n / fps, 2) if has else 0.0,
        "dip_ratio": round(dip, 3),
        "dip_at": lo,
        "tail_ratio": round(tail, 3),
        "ref_sharpness": round(ref, 1),
        "profile": profile,
        "curve": [round(float(x), 2) for x in rel[:: max(1, n // 12)][:12]],
    }


# ---------------------------------------------------------------- 主测量
def analyze(path, maxw=480, windows=3):
    """
    对每个镜头，在 windows 个时间窗口上做「窗口首帧 vs 尾帧」的背景测量，
    再把各窗口的缩放相乘、旋转相加、位移相加（窗口数少，累计误差可控），
    每次窗口测量内部都用背景区聚合。
    """
    frames, fps, (sw, sh) = load_gray_frames(path, maxw=maxw)
    n = len(frames)
    h, w = frames[0].shape
    cuts, diffs = detect_cuts(frames)

    bounds = [0] + cuts + [n]
    shots = []
    for k in range(len(bounds) - 1):
        s, e = bounds[k], bounds[k + 1] - 1
        if e <= s:
            continue
        shot = measure_shot(frames, s, e, w, h, fps, k + 1, windows)
        # 清晰度分析：与几何运动独立，单列
        sharp = sharpness_curve(frames, s, e)
        shot["blur"] = analyze_blur(sharp, fps)
        shots.append(shot)

    return {
        "video": os.path.abspath(path),
        "src_width": sw, "src_height": sh, "work_size": [w, h],
        "fps": round(fps, 3), "frames": n, "duration": round(n / fps, 3),
        "cut_frames": cuts, "shots": shots,
    }


def measure_shot(frames, s, e, w, h, fps, index, windows=3):
    base = {"index": index, "start_frame": s, "end_frame": e,
            "start_time": round(s / fps, 3), "end_time": round(e / fps, 3),
            "n_frames": e - s + 1}
    nf = e - s
    if nf < 4:
        base.update({"labels": ["Static Shot"], "amplitude": 0.0,
                     "amplitude_phrase": "", "speed_phrase": "", "confidence": "高",
                     "h3": "The camera holds a static shot."})
        return base

    # 分窗
    wins = max(1, min(windows, nf // 6))
    edges = [s + int(round(i * nf / wins)) for i in range(wins + 1)]
    per_win, scales, rots, txs, tys, pars, confs = [], [], [], [], [], [], []
    for i in range(wins):
        a_i, b_i = edges[i], min(edges[i + 1], e)
        if b_i - a_i < 4:
            continue
        agg = aggregate_regions(frames[a_i], frames[b_i])
        per_win.append({"from": a_i, "to": b_i, "ok": agg["ok"],
                        "conf": agg.get("confidence", "—"), "agg": agg})
        if not agg["ok"]:
            continue
        scales.append(agg["scale"]); rots.append(agg["rot"])
        txs.append(agg["tx"]); tys.append(agg["ty"]); pars.append(agg["parallax"])
        confs.append(agg["confidence"])

    if not scales:
        base.update({"labels": ["Static Shot"], "amplitude": 0.0,
                     "amplitude_phrase": "", "speed_phrase": "",
                     "confidence": "不可判（背景不足）", "windows": per_win,
                     "h3": "The camera motion could not be determined (insufficient background)."})
        return base

    zoom = float(np.prod(scales))
    rot_total = float(np.sum(rots))
    norm_x = float(np.sum(txs)) / w
    norm_y = float(np.sum(tys)) / h
    parallax = float(np.median(pars))
    dur = nf / fps
    # 速度：总位移量 / 时长
    total_motion = math.hypot(norm_x, norm_y) + abs(math.log(max(zoom, 1e-3))) * 0.5
    spd = total_motion / dur if dur > 0 else 0.0
    amp = max(abs(norm_x), abs(norm_y), abs(zoom - 1.0) * 0.5)
    dolly = parallax >= PARALLAX_EPS

    conf = "高" if all(c.startswith("高") for c in confs) else (
        "中" if all(c.startswith(("高", "中")) for c in confs) else "低")

    labels, en = [], []
    # 符号约定（v3 修过一次，别搞反）：
    #   横向：内容右移 = 相机向左摇/左移（x 轴不翻转）
    #   纵向：内容下移 = 相机上仰/上升（图像 y 向下，所以反一下）
    if abs(norm_x) >= AMP_STATIC:
        side = "left" if norm_x > 0 else "right"
        labels += ["Pan " + side.title(), "Truck " + side.title()]
        en.append("pans %s, sweeping the frame horizontally by about %.0f%% of the frame width"
                  % (side, abs(norm_x) * 100))
    if abs(norm_y) >= AMP_STATIC:
        side = "up" if norm_y > 0 else "down"
        labels += ["Tilt " + side.title(), "Pedestal " + side.title()]
        en.append("tilts %s, moving the frame vertically by about %.0f%% of the frame height"
                  % (side, abs(norm_y) * 100))
    if abs(zoom - 1.0) >= ZOOM_EPS:
        if zoom > 1:
            verb = "pushes in" if dolly else "zooms in"
            labels += ["Push In", "Zoom In"] if dolly else ["Zoom In"]
        else:
            verb = "pulls out" if dolly else "zooms out"
            labels += ["Pull Out", "Zoom Out"] if dolly else ["Zoom Out"]
        en.append("%s, changing the framing by %.0f%% (to %.2fx)"
                  % (verb, abs(zoom - 1) * 100, zoom))
    if abs(rot_total) >= ROT_EPS_DEG:
        side = "clockwise" if rot_total > 0 else "counterclockwise"
        labels.append("Roll " + side.title())
        en.append("rolls %s by about %.1f degrees" % (side, abs(rot_total)))

    static = not en
    if static:
        labels = ["Static Shot"]
        h3 = "The camera holds a static shot while the action unfolds."
    else:
        amp_phrase = ("with large amplitude" if amp >= AMP_LARGE
                      else "" if amp < AMP_STATIC else "with small amplitude")
        spd_phrase = ("at slow speed" if spd <= SPD_SLOW
                      else "at fast speed" if spd >= SPD_FAST else "")
        mods = " ".join(x for x in (amp_phrase, spd_phrase) if x)
        h3 = "The camera %s%s." % (" and ".join(en), (", " + mods) if mods else "")

    if static:
        amp_phrase = spd_phrase = ""

    base.update({
        "norm_x": round(norm_x, 4), "norm_y": round(norm_y, 4),
        "zoom_total": round(zoom, 4), "rot_deg_total": round(rot_total, 2),
        "parallax_ratio": round(parallax, 3),
        "dolly_evidence": "有视差→机位在动（Push/Pull）" if dolly else "无视差→变焦或近似平面（Zoom）",
        "amplitude": round(amp, 4),
        "amplitude_phrase": amp_phrase or ("medium (default, unstated)" if not static else ""),
        "speed_norm_per_s": round(spd, 4),
        "speed_phrase": spd_phrase or ("normal (default, unstated)" if not static else ""),
        "labels": labels, "h3": h3, "confidence": conf,
        "windows": per_win,
    })
    return base


# ---------------------------------------------------------------- 抽帧联络表
def contact_sheet(path, out_png, cols=3, rows=3, thumb_w=360):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        cap.release()
        return None
    idx = [int(round(k * (n - 1) / (cols * rows - 1))) for k in range(cols * rows)]
    tiles = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            continue
        sc = thumb_w / float(fr.shape[1])
        fr = cv2.resize(fr, (thumb_w, max(2, int(fr.shape[0] * sc))),
                        interpolation=cv2.INTER_AREA)
        cv2.putText(fr, "f%d" % i, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(fr, "f%d" % i, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(fr)
    cap.release()
    if not tiles:
        return None
    th, tw = tiles[0].shape[:2]
    sheet = np.zeros((th * rows, tw * cols, 3), dtype=np.uint8)
    for k, t in enumerate(tiles[: cols * rows]):
        r, c = divmod(k, cols)
        sheet[r * th: r * th + t.shape[0], c * tw: c * tw + t.shape[1]] = t
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    cv2.imwrite(out_png, sheet)
    return out_png


# ---------------------------------------------------------------- 报告
def fmt_report(r):
    L = ["=" * 74, "运镜测量报告", "=" * 74,
         "视频      : %s" % r["video"],
         "画幅/帧率 : %dx%d  %.2ffps  %d 帧  %.2fs" % (
             r["src_width"], r["src_height"], r["fps"], r["frames"], r["duration"]),
         "切镜      : %s" % ("无（单镜到底）" if not r["cut_frames"]
                              else "第 %s 帧（共 %d 个切点）" % (r["cut_frames"], len(r["cut_frames"]))),
         ""]
    for s in r["shots"]:
        L.append("── 镜头 %d  [%.2fs → %.2fs]  %d 帧  置信度 %s" % (
            s["index"], s["start_time"], s["end_time"], s["n_frames"], s.get("confidence", "—")))
        if "norm_x" not in s or s.get("labels") == ["Static Shot"]:
            L.append("   %s" % s["h3"])
            L.append("")
            continue
        L.append("   平移   横向 %+.1f%% 画面宽（%s）   纵向 %+.1f%% 画面高（%s）" % (
            s["norm_x"] * 100, "内容右移=相机向左摇/左移" if s["norm_x"] > 0 else "内容左移=相机向右摇/右移",
            s["norm_y"] * 100, "内容下移=相机上仰/上升" if s["norm_y"] > 0 else "内容上移=相机下俯/下降"))
        L.append("   缩放   %+.1f%%（累计 %.3fx）   视差比 %.2f → %s" % (
            (s["zoom_total"] - 1) * 100, s["zoom_total"],
            s["parallax_ratio"], s.get("dolly_evidence", "")))
        L.append("   滚转   %+.2f°" % s["rot_deg_total"])
        L.append("   幅度   %.1f%% → %s       速度 %.1f%% 画面/秒 → %s" % (
            s["amplitude"] * 100, s["amplitude_phrase"],
            s["speed_norm_per_s"] * 100, s["speed_phrase"]))
        L.append("   判读   %s" % " + ".join(s["labels"]))
        L.append("   H3     %s" % s["h3"])
        # 背景区分项读数，便于人核对（这是 v4 的核心，别省）
        for wi, wd in enumerate(s.get("windows", [])):
            if not wd["ok"]:
                L.append("    窗口%d f%d→f%d  %s" % (wi + 1, wd["from"], wd["to"], wd["agg"]["reason"]))
                continue
            agg = wd["agg"]
            parts = ["%s×%.3f/%.1f°" % (n, agg["raw"][n]["scale"], agg["raw"][n]["rot"])
                     for n in agg["regions_used"]]
            L.append("    窗口%d f%d→f%d  用 %d 区，MAD 缩放%.3f 滚转%.2f° → %s" % (
                wi + 1, wd["from"], wd["to"], len(agg["regions_used"]),
                agg["mad_scale"], agg["mad_rot"], agg["confidence"]))
            L.append("              %s" % "  ".join(parts))
        L.append("")
    L.append("说明：只测画面四周的背景带，不测中心（中心是主体，会把相机运动带跑）。")
    L.append("      Pan/Truck、Tilt/Pedestal 在 2D 里物理不可分，给双候选，按原片二选一。")
    return "\n".join(L)


def emit_prompt_block(r):
    out = []
    if len(r["shots"]) == 1:
        return r["shots"][0]["h3"]
    for s in r["shots"]:
        body = s["h3"].replace("The camera ", "").rstrip(".")
        if body.startswith("holds a static"):
            out.append("[Shot %d] %s." % (s["index"], s["h3"]))
        else:
            out.append("[Shot %d] At %02d:%06.3f, the camera %s." % (
                s["index"], int(s["start_time"] // 60), s["start_time"] % 60, body))
    return "\n".join(out)


# ---------------------------------------------------------------- prompt 校验
CAM_TERMS = [
    ("push in", "Push In"), ("pull out", "Pull Out"),
    ("zoom in", "Zoom In"), ("zoom out", "Zoom Out"),
    ("pan left", "Pan Left"), ("pan right", "Pan Right"),
    ("truck left", "Truck Left"), ("truck right", "Truck Right"),
    ("tilt up", "Tilt Up"), ("tilt down", "Tilt Down"),
    ("pedestal up", "Pedestal Up"), ("pedestal down", "Pedestal Down"),
    ("arc shot", "Arc Shot"), ("tracking shot", "Tracking Shot"),
    ("static shot", "Static Shot"),
    ("shake slightly", "Shake Slightly"), ("shake strongly", "Shake Strongly"),
    ("pov", "POV"), ("roll clockwise", "Roll Clockwise"),
    ("roll counterclockwise", "Roll Counterclockwise"),
    # 常见自由写法（H3 也认，但不在官方词表里）
    ("dolly in", "Push In"), ("dolly out", "Pull Out"),
    ("tilt up", "Tilt Up"),
]
FAMILIES = [
    {"Push In", "Zoom In"}, {"Pull Out", "Zoom Out"},
    {"Pan Left", "Truck Left"}, {"Pan Right", "Truck Right"},
    {"Tilt Up", "Pedestal Up"}, {"Tilt Down", "Pedestal Down"},
]


def _same_family(a, b):
    return a == b or any(a in f and b in f for f in FAMILIES)


def check_prompt(r, text):
    low = text.lower()
    claimed = []
    for kw, tag in CAM_TERMS:
        if kw in low and tag not in claimed:
            claimed.append(tag)
    measured = set()
    for s in r["shots"]:
        for lb in s.get("labels", []):
            measured.add(lb)
    L = ["-" * 74, "prompt 运镜校验", "-" * 74,
         "prompt 声称: %s" % (", ".join(claimed) if claimed else "（没写任何运镜词）"),
         "实测判读  : %s" % (", ".join(sorted(measured)) or "—")]
    hit = [c for c in claimed if c in measured]
    fam = [c for c in claimed if c not in measured and any(_same_family(c, m) for m in measured)]
    miss = [c for c in claimed if c not in measured and not any(_same_family(c, m) for m in measured)]
    if hit:
        L.append("OK  直接对应 : %s" % ", ".join(hit))
    if fam:
        L.append("~   同族对应 : %s（方向对得上，措辞不同，H3 都认）" % ", ".join(fam))
    if miss:
        L.append("X   实测没有 : %s  ← 最大问号，先看抽帧再定" % ", ".join(miss))
    extra = [m for m in sorted(measured) if not any(_same_family(c, m) for c in claimed)]
    if extra:
        L.append("+   漏写     : %s（实测有、prompt 没提 → 这一路吃模型自由发挥）" % ", ".join(extra))
    if claimed and not hit and not fam:
        L.append("!! 完全对不上：要么 prompt 没生效，要么这段参考片里根本没这个运镜。")
    return "\n".join(L)


# ---------------------------------------------------------------- 自检
def _synthetic_scene(h=360, w=640, seed=7):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.uint8)
    for _ in range(300):
        x, y = int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20))
        cv2.circle(img, (x, y), int(rng.integers(3, 20)), int(rng.integers(40, 255)), -1)
    return cv2.GaussianBlur(img, (5, 5), 0)


def selftest():
    """合成已知变换，校验测量精度与符号方向。改拟合/聚合代码后必须跑。"""
    print("=" * 74)
    print("自检 1：单区域精度（合成已知变换）")
    print("=" * 74)
    base = _synthetic_scene()
    h, w = base.shape
    ctr = ((w - 1) / 2, (h - 1) / 2)

    cases = []
    for deg in (-3.0, -1.0, -0.5, 0.5, 1.0, 3.0):
        cases.append(("rot %+.1f°" % deg, cv2.getRotationMatrix2D(ctr, deg, 1.0)))
    for sc in (0.96, 0.99, 1.01, 1.04):
        cases.append(("zoom ×%.2f" % sc, cv2.getRotationMatrix2D(ctr, 0, sc)))
    for sh in (2.0, 8.0):
        cases.append(("shift %+.0fpx" % sh, np.float32([[1, 0, sh], [0, 1, 0]])))

    worst_rot = 0.0
    print("%-16s %11s %11s" % ("变换", "测出旋转", "测出缩放"))
    for name, M in cases:
        moved = cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_LINEAR)
        r = fit_region(base, moved, (0, 0, 1, 1))
        if r is None:
            print("%-16s  拟合失败" % name)
            worst_rot = 99.0
            continue
        if name.startswith("rot"):
            want = -float(name.split()[1].rstrip("°"))
            err = abs(r["rot"] - want)
            worst_rot = max(worst_rot, err)
            print("%-16s %+10.3f° %11.4f   误差 %.3f°" % (name, r["rot"], r["scale"], err))
        else:
            print("%-16s %+10.3f° %11.4f" % (name, r["rot"], r["scale"]))

    zerr = []
    for sc in (0.96, 0.99, 1.01, 1.04):
        moved = cv2.warpAffine(base, cv2.getRotationMatrix2D(ctr, 0, sc), (w, h))
        r = fit_region(base, moved, (0, 0, 1, 1))
        if r:
            zerr.append(abs(r["scale"] - sc))
    print("-" * 74)
    print("最大旋转误差 %.3f°（要求 <0.15°）  最大缩放误差 %.4f（要求 <0.005）" % (
        worst_rot, max(zerr) if zerr else 0))
    prec_ok = worst_rot < 0.15 and (not zerr or max(zerr) < 0.005)

    print()
    print("=" * 74)
    print("自检 2：方向符号（v3 修过一次，回归必测）")
    print("=" * 74)
    sign_ok = True
    # 位移必须超过 AMP_STATIC(4.5% 画面) 才会被判读，否则只会报 Static Shot。
    # 640 宽 → 4.5% = 29px；这里用 45px(7%) 保证稳稳越过门限。
    shift_px = 45
    for label, sh, want in (("内容右移", (shift_px, 0), "Pan Left"),
                            ("内容左移", (-shift_px, 0), "Pan Right"),
                            ("内容下移", (0, shift_px), "Tilt Up"),
                            ("内容上移", (0, -shift_px), "Tilt Down")):
        moved = cv2.warpAffine(base, np.float32([[1, 0, sh[0]], [0, 1, sh[1]]]),
                               (w, h), borderMode=cv2.BORDER_REFLECT)
        agg = aggregate_regions(base, moved)
        s = _quick_shot_from_agg(agg, w, h, 24.0, 1)
        got = (s.get("labels") or ["?"])[0]
        good = got == want
        sign_ok = sign_ok and good
        print("  %-8s %+3dpx → %-11s 期望 %-11s %s" % (
            label, sh[0] or sh[1], got, want, "OK" if good else "错！"))

    print()
    print("=" * 74)
    print("自检 3：主体独立运动 —— 污染确实存在，而背景法会**拒绝作答**")
    print("=" * 74)
    # 合成实验结论（别改判据，这是量出来的边界）：
    #   RANSAC 本身对「主体叠加运动」相当稳健——主体独立横移 80px 时，
    #   只要背景角点数量 >= 主体的 ~1/3，全画面法就仍能给出正解。
    #   真正会崩的是「主体角点压倒背景」（本例 12:1，典型如人物特写/细节满屏）：
    #       背景撒 60 点 → 全画面法 zoom=1.000 tx=+80px  ← 完全错了（真值 1.10 / 0）
    #   这时背景带因为每带角点不足，返回「不可用」而**不给错误答案**——
    #   这正是 v4 想要的行为：宁可说测不了，也不要给出被主体带跑的数。
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = (60 + 40 * np.sin(xx / 140.0) + 30 * np.cos(yy / 110.0)).clip(0, 255).astype(np.uint8)
    rng_bg = np.random.default_rng(23)
    for _ in range(60):   # 刻意让背景稀疏，制造「主体压倒背景」
        px, py = int(rng_bg.integers(6, w - 6)), int(rng_bg.integers(6, h - 6))
        cv2.circle(bg, (px, py), int(rng_bg.integers(2, 7)),
                   int(rng_bg.integers(20, 240)), -1)

    rng = np.random.default_rng(5)
    subj = rng.integers(0, 256, (200, 200)).astype(np.uint8)   # 满分辨率噪声 = 角点极密
    ya, xa = 80, 220
    frame_a = bg.copy()
    frame_a[ya:ya + 200, xa:xa + 200] = subj
    frame_b = cv2.warpAffine(bg, cv2.getRotationMatrix2D(ctr, 0, 1.10), (w, h))
    frame_b[ya:ya + 200, xa:xa + 200] = cv2.warpAffine(
        subj, np.float32([[1, 0, 80], [0, 1, 0]]), (200, 200))

    full = fit_region(frame_a, frame_b, (0, 0, 1, 1))
    agg = aggregate_regions(frame_a, frame_b)
    print("  真值：背景 zoom 1.10 且无平移；主体（角点极密）独立横移 80px")
    print("  全画面法 : zoom=%.3f  tx=%+.1f px   ← 被主体带跑，给出错误答案" % (full["scale"], full["tx"]))
    if agg["ok"]:
        print("  背景带法 : zoom=%.3f  tx=%+.1f px（%s）" % (agg["scale"], agg["tx"], agg["confidence"]))
    else:
        print("  背景带法 : 拒绝作答（%s）  ← 这是正确行为" % agg["reason"])

    full_wrong = abs(full["scale"] - 1.10) > 0.05 or abs(full["tx"]) > 20
    bg_safe = (not agg["ok"]) or (abs(agg["scale"] - 1.10) < 0.03 and abs(agg["tx"]) < 6)
    contam_ok = full_wrong and bg_safe

    print()
    print("=" * 74)
    print("自检 4：背景充足时不该拒绝（阴性对照，防止阈值调得过严）")
    print("=" * 74)
    bg2 = (60 + 40 * np.sin(xx / 140.0) + 30 * np.cos(yy / 110.0)).clip(0, 255).astype(np.uint8)
    rng_bg2 = np.random.default_rng(23)
    for _ in range(200):
        px, py = int(rng_bg2.integers(6, w - 6)), int(rng_bg2.integers(6, h - 6))
        cv2.circle(bg2, (px, py), int(rng_bg2.integers(2, 7)),
                   int(rng_bg2.integers(20, 240)), -1)
    fa2 = bg2.copy()
    fa2[ya:ya + 200, xa:xa + 200] = subj
    fb2 = cv2.warpAffine(bg2, cv2.getRotationMatrix2D(ctr, 0, 1.10), (w, h))
    fb2[ya:ya + 200, xa:xa + 200] = cv2.warpAffine(
        subj, np.float32([[1, 0, 80], [0, 1, 0]]), (200, 200))
    agg2 = aggregate_regions(fa2, fb2)
    if agg2["ok"]:
        print("  背景充足 : zoom=%.3f  tx=%+.1f px（%s，%d 区）  真值 1.10 / 0"
              % (agg2["scale"], agg2["tx"], agg2["confidence"], len(agg2["regions_used"])))
        neg_ok = abs(agg2["scale"] - 1.10) < 0.05 and abs(agg2["tx"]) < 10
    else:
        print("  背景充足 : 仍然拒绝（%s）  ← 阈值过严" % agg2["reason"])
        neg_ok = False
    print("  判定：%s" % ("通过 —— 背景够时正常作答" if neg_ok else "不通过"))
    print("  判定：%s" % ("通过 —— 全画面法被带跑到错误答案，背景法拒绝作答" if contam_ok
                          else "不通过"))

    ok = prec_ok and sign_ok and contam_ok and neg_ok

    print()
    print("=" * 74)
    print("自检 5：运动模糊检测 + 假切点过滤")
    print("=" * 74)
    # 合成一段「清晰 → 强运动模糊 → 保持糊」的序列（甩镜的真实形态）
    # 注意：模糊必须是**逐帧剧烈变化**的，才会像真实甩镜那样在帧差上产生一串候选。
    # 早先版本只是把同一个模糊结果渐变（每帧变化很小），候选点只有 2 个，
    # 密度判据抓不住——合成用例本身就失真了。
    rng2 = np.random.default_rng(31)
    tex = rng2.integers(0, 256, (360, 640)).astype(np.uint8)
    tex = cv2.GaussianBlur(tex, (3, 3), 0)
    seq = []
    for i in range(60):
        if i < 12:
            f = tex.copy()
        else:
            # 每帧用不同的横向位移 + 强度递增的模糊，模拟相机高速旋转
            dx = int(6 + (i - 12) * 9)
            k = 3 + int(30 * min(1.0, (i - 12) / 10.0))
            shifted = np.roll(tex, dx, axis=1)
            f = cv2.blur(shifted, (k, 1))
        seq.append(f)
    sharp = sharpness_curve(seq, 0, len(seq) - 1)
    b = analyze_blur(sharp, 24.0)
    blur_ok = bool(b["has_blur"])
    print("  合成甩镜段 : has_blur=%s  dip=%.3f  profile=%s  （期望 True）"
          % (b["has_blur"], b.get("dip_ratio", 0), b.get("profile", "")))
    print("  判定：%s" % ("通过" if blur_ok else "不通过 —— 强模糊没被检出"))

    # 静止段不该报模糊（阴性对照）
    flat = [tex.copy() for _ in range(60)]
    b2 = analyze_blur(sharpness_curve(flat, 0, 59), 24.0)
    neg2_ok = not b2["has_blur"]
    print("  静止段     : has_blur=%s  （期望 False）" % b2["has_blur"])
    print("  判定：%s" % ("通过" if neg2_ok else "不通过 —— 把清晰画面误报成模糊"))

    # 模糊过渡不该被切成多个镜头（这条最容易回归）
    cuts_fake, _ = detect_cuts(seq)
    cut_ok = len(cuts_fake) == 0
    print("  连续模糊段 : 检出切点=%s  （期望 []，不该把模糊当成硬切）" % cuts_fake)
    print("  判定：%s" % ("通过" if cut_ok else "不通过 —— 模糊被误判成切镜"))

    ok = ok and blur_ok and neg2_ok and cut_ok
    print()
    print("自检总结：%s" % ("全部通过" if ok else "有超差项 —— 不要拿测量结果下结论"))
    return 0 if ok else 1


def _quick_shot_from_agg(agg, w, h, fps, index):
    """自检用：把一次聚合结果直接过一遍判读逻辑。"""
    if not agg["ok"]:
        return {"labels": []}
    norm_x, norm_y = agg["tx"] / w, agg["ty"] / h
    zoom, rot = agg["scale"], agg["rot"]
    labels = []
    if abs(norm_x) >= AMP_STATIC:
        labels.append("Pan " + ("Left" if norm_x > 0 else "Right").title())
    if abs(norm_y) >= AMP_STATIC:
        labels.append("Tilt " + ("Up" if norm_y > 0 else "Down").title())
    if abs(zoom - 1.0) >= ZOOM_EPS:
        labels.append("Push In" if zoom > 1 else "Pull Out")
    if abs(rot) >= ROT_EPS_DEG:
        labels.append("Roll " + ("Clockwise" if rot > 0 else "Counterclockwise").title())
    return {"labels": labels or ["Static Shot"]}


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="参考片运镜测量（H3 运镜迁移用）")
    ap.add_argument("--video", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--check-prompt", default="")
    ap.add_argument("--emit-prompt", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--windows", type=int, default=3, help="每个镜头分几个时间窗口测（默认 3）")
    ap.add_argument("--max-width", type=int, default=480)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.video:
        ap.error("需要 --video（或用 --selftest）")

    r = analyze(args.video, maxw=args.max_width, windows=args.windows)
    report = fmt_report(r)
    print(report)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        jp = args.json or os.path.join(args.out, "motion.json")
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print("\n已写 json   : %s" % jp)
        print("已写报告    : %s" % os.path.join(args.out, "report.txt"))
        if args.emit_prompt:
            pp = os.path.join(args.out, "camera_block.txt")
            with open(pp, "w", encoding="utf-8") as f:
                f.write(emit_prompt_block(r) + "\n")
            print("已写运镜段落: %s" % pp)
        if args.sheet:
            sp = os.path.join(args.out, "sheet.png")
            if contact_sheet(args.video, sp):
                print("已写抽帧    : %s" % sp)

    if args.check_prompt:
        with open(args.check_prompt, "r", encoding="utf-8") as f:
            print("\n" + check_prompt(r, f.read()))


if __name__ == "__main__":
    main()
