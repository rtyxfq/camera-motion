#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P1 运镜标定批跑

设计原则（每条都对应一个已知坑）：
  1. 唯一变量 = 运镜句。其余 prompt 文本、seed、分辨率、时长、步数全部固定。
  2. 空镜场景，画面里没有会自己动的东西 —— 否则主体运动会污染相机测量。
     （教训：早前一条对照实验的 prompt 里有人物动作，导致运镜测不准，工具改了三版）
  3. 每条 5s / 124 帧 / 1344x768 / 4 步 turbo —— 单条约 2~3 分钟。
  4. 每组 3 个 seed —— 单条会被随机性骗。
  5. 显式声明 one continuous shot —— 否则 H3 会把长 prompt 自动切镜，
     一旦切镜，逐镜运动就被切点打断，测不出整段累计量。
     （教训：MiniMax_H3_00153 那条 10s 被切成 3 镜，prompt 里根本没要求切）

用法：
  python run_calibration.py --group A --dry-run     # 只打印将要提交的 prompt
  python run_calibration.py --group A               # 跑 A 组（3 条）
  python run_calibration.py --group all             # 跑全部
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from comfyapi import submit, wait, outputs_of, download  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "runs", "videos")
LOGS = os.path.join(ROOT, "runs")

FIRST_FRAME = "CALIB_moss17_16x9.png"

WIDTH, HEIGHT = 1344, 768
LENGTH = 124          # 124 帧 ≈ 5.17s @24fps
STEPS = 4             # turbo 4 步
SEEDS = [1001, 2002, 3003]

# ---------------------------------------------------------------- prompt 骨架
# 共享部分必须逐字一致。任何差异都会成为混杂变量。
SCENE = (
    "Live-action, cinematic, a wide shot frames a colossal weathered stone hand "
    "standing upright on a concrete platform above still shallow water. Layered "
    "metal scaffolding rises on both the left and right sides, receding toward a "
    "hazy city skyline on the horizon. Wet ground reflects the pale sky, and thin "
    "construction cranes are visible in the far distance."
)

SHARED_TAIL = (
    " This is one continuous shot with no cuts and no scene changes. "
    "The scene itself remains still, with no people, no vehicles and no moving objects."
)

BASE = (
    "For the target video, at 0.00 seconds into the target video, <Picture 1> "
    "(from [Shot 1]) is fully referenced.\n\n"
    "integrated_multimodal_description: [Shot 1] " + SCENE + "{CAMERA}" + SHARED_TAIL + "\n\n"
    "overall_soundscape: Quiet open-air ambience with faint wind and a distant "
    "construction hum. Still water laps softly against the platform edge.\n\n"
    "non_diegetic_music: N/A"
)

# ---------------------------------------------------------------- 对照组
# 每组只提供运镜句。CAMERA 为空时连空格都不留，确保无运镜词。
GROUPS = {
    # 不写任何运镜词 → 测 H3 默认行为（漂不漂）
    "A": {
        "desc": "不写运镜词（测默认）",
        "camera": "",
    },
    # 显式锁帧 → 测 Static Shot 有没有用
    "B": {
        "desc": "显式 Static Shot（测锁帧）",
        "camera": " The camera holds a static shot.",
    },
    # 基线：只写动词，不带幅度速度
    "C": {
        "desc": "Push In 无修饰（测基线）",
        "camera": " The camera pushes in.",
    },
    # 幅度+速度都写上 → 与 C 对比就是幅度/速度的贡献
    "D": {
        "desc": "Push In + large + slow（测幅度速度）",
        "camera": " The camera pushes in with large amplitude at slow speed.",
    },
    # ---- 第二轮：把 D 的两个变量拆开（2x2 析因设计）----
    #        无速度        + slow speed
    # 无幅度   C            E
    # +large   F            D
    # 只变速度 → 与 C 比，看 slow 单独的影响
    "E": {
        "desc": "Push In + slow（只测速度）",
        "camera": " The camera pushes in at slow speed.",
    },
    # 只变幅度 → 与 C 比，看 large 单独的影响
    "F": {
        "desc": "Push In + large（只测幅度）",
        "camera": " The camera pushes in with large amplitude.",
    },
    # ---- 第三轮：whip pan 验证 ----
    # 官方 12 类词表里没有 whip pan，但本机 3 个工作流在用它。
    # 这是"正在使用、可能一直无效"的最高风险项，必须先测。
    # G: 直写官方不认的词
    "G": {
        "desc": "whip pan 裸写（测官方未列词是否生效）",
        "camera": " The camera whip pans.",
    },
    # H: 拆成官方词 + 快速度，作为"如果 G 无效就改用这个"的备选
    #    宽度：横向位移预期大，便于和 G 对比
    "H": {
        "desc": "Pan left + fast（whip pan 的官方词替代方案）",
        "camera": " The camera pans left at fast speed.",
    },
    # ---- 第四轮：词表是否封闭 ----
    # 这条是 P0 里所有 ❌/⬜ 判定的上游问题。
    # 官方原文只说 "Available Expression"，从没说封闭集合。
    # 若 I 与 C（pushes in）效果相当 → 词表不封闭，同义词可用；
    # 若 I 明显更弱/无效 → 词表趋近封闭，必须换官方词。
    "I": {
        "desc": "dollies in（非官方同义词，对照 C 组）",
        "camera": " The camera dollies in.",
    },
}

# 补跑用的新种子（不与第一轮重复，避免同 seed 覆盖或复用同一随机流）
SEEDS_ROUND2 = [4004, 5005, 6006]


def build_prompt(camera_sentence):
    return BASE.replace("{CAMERA}", camera_sentence)


def seeds_for_group(g):
    """第一轮 A–D 用 SEEDS；E/F 及之后的新组用 SEEDS_ROUND2。

    补跑 C/D 时用 --seeds 显式传入新种子，避免和已有数据同 seed。
    """
    return SEEDS_ROUND2 if g in ("E", "F", "G", "H", "I") else SEEDS


def build_workflow(prompt, seed, group="X"):
    """4 步 turbo 版：fl2va 底模 + fl2v_turbo_8step LoRA + simple 调度。

    ⚠️ filename_prefix 必须带组号（踩过）：最初只写 seed，
    结果三个 seed × 四组共 12 个文件全叫 CAL_<seed>_0000N_.mp4，
    文件名看不出是哪一组，只能靠时间顺序猜。
    """
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "weight_dtype": "default"}},
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["unet", 0],
            "lora_name": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
            "strength_model": 1.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {
            "vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "avae": {"class_type": "VAELoader", "inputs": {
            "vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "img": {"class_type": "LoadImage", "inputs": {"image": FIRST_FRAME}},
        "h3": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["clip", 0],
            "vae": ["vae", 0],
            "prompt": prompt,
            "width": WIDTH, "height": HEIGHT, "length": LENGTH,
            "first_frame": ["img", 0]}},
        "sampler_sel": {"class_type": "KSamplerSelect", "inputs": {
            "sampler_name": "res_multistep"}},
        "sched": {"class_type": "BasicScheduler", "inputs": {
            "model": ["lora", 0], "scheduler": "simple",
            "steps": STEPS, "denoise": 1}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "guider": {"class_type": "BasicGuider", "inputs": {
            "model": ["lora", 0], "conditioning": ["h3", 0]}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["noise", 0], "guider": ["guider", 0],
            "sampler": ["sampler_sel", 0], "sigmas": ["sched", 0],
            "latent_image": ["h3", 1]}},
        "dec": {"class_type": "VAEDecode", "inputs": {
            "samples": ["sample", 0], "vae": ["vae", 0]}},
        "adec": {"class_type": "VAEDecodeAudio", "inputs": {
            "samples": ["sample", 0], "vae": ["avae", 0]}},
        "vid": {"class_type": "CreateVideo", "inputs": {
            "images": ["dec", 0], "fps": 24, "audio": ["adec", 0]}},
        "save": {"class_type": "SaveVideo", "inputs": {
            "video": ["vid", 0],
            "filename_prefix": "calib/CAL_%s_seed%d" % (group, seed),
            "format": "auto", "format.codec": "auto"}},
    }


def main():
    global SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", help="A~F 或 all")
    ap.add_argument("--seeds", default="", help="逗号分隔，覆盖默认种子（补跑时用）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.seeds:
        SEEDS = [int(s) for s in args.seeds.split(",") if s.strip()]
        print("使用自定义种子: %s" % SEEDS)

    groups = list(GROUPS) if args.group == "all" else [args.group.upper()]
    for g in groups:
        if g not in GROUPS:
            ap.error("未知组 %s，可选 %s" % (g, list(GROUPS)))

    # 只有真要跑才建目录。--dry-run 是纯打印，不该在磁盘上留任何东西
    #（踩过：dry-run 也会 makedirs，在没有写权限的目录里直接崩掉）
    if not args.dry_run:
        os.makedirs(OUT, exist_ok=True)
        os.makedirs(LOGS, exist_ok=True)

    manifest = []
    t_all = time.time()
    for g in groups:
        cam = GROUPS[g]["camera"]
        prompt = build_prompt(cam)
        print("\n" + "=" * 70)
        print("[组 %s] %s" % (g, GROUPS[g]["desc"]))
        print("运镜句: %r" % cam.strip() if cam.strip() else "运镜句: （空）")
        print("prompt 长度: %d 字符" % len(prompt))
        if args.dry_run:
            print("-" * 70)
            print(prompt)
            continue

        for seed in seeds_for_group(g):
            tag = "%s_%d" % (g, seed)
            t0 = time.time()
            try:
                wf = build_workflow(prompt, seed, g)
                pid = submit(wf)
                print("  [%s] 已提交 %s" % (tag, pid), flush=True)
                hist = wait(pid, timeout=1800, poll=5)
                outs = [o for o in outputs_of(hist, pid) if o[1] in ("videos", "gifs", "images")]
                if not outs:
                    print("  [%s] NO OUTPUT" % tag, flush=True)
                    manifest.append({"group": g, "seed": seed, "status": "no_output"})
                    continue
                item = outs[0][2]
                ext = os.path.splitext(item["filename"])[1] or ".mp4"
                dest = os.path.join(OUT, "%s%s" % (tag, ext))
                download(item, dest)
                dur = time.time() - t0
                print("  [%s] OK -> %s (%.0fs)" % (tag, os.path.basename(dest), dur), flush=True)
                manifest.append({
                    "group": g, "seed": seed, "file": os.path.abspath(dest),
                    "camera_sentence": cam.strip(), "seconds": round(dur, 1),
                    "status": "ok",
                })
            except Exception as e:
                print("  [%s] ERROR %s" % (tag, str(e)[:400]), flush=True)
                manifest.append({"group": g, "seed": seed,
                                 "status": "error", "error": str(e)[:400]})

    if not args.dry_run:
        mp = os.path.join(LOGS, "manifest.json")
        old = []
        if os.path.isfile(mp):
            try:
                old = json.load(open(mp, encoding="utf-8"))
            except Exception:
                old = []
        json.dump(old + manifest, open(mp, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("\n清单已写: %s" % mp)
        print("总耗时 %.1f 分钟" % ((time.time() - t_all) / 60))


if __name__ == "__main__":
    main()
