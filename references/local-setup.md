> 本机（便携版 ComfyUI）相关的绝对路径、模型路径、工作流名单不写进本仓库。
> 需要时由使用者在自己的环境里确认，避免把某台机器的配置当成通用事实。

# 与本地环境的对接

## Python 依赖

`tools/camera_motion.py` 需要：

```
numpy, opencv-python
```

（本机便携版 ComfyUI 的 `python_embeded` 自带这两个，可直接用。）

## 工作流

本 skill 不含工作流文件。用你自己环境里的：

- 快速迭代：取步数最少的那条（例如 4 步 turbo 档）
- 搬运镜验证：需要能把参考视频接到 `ref_videos.ref_video_0` 的 ref2va 工作流

## 路径

脚本所有输出都写到 `--out` 指定目录，默认不依赖任何固定路径。
`runs/` 与 `caps/evidence/` 属于本地数据，建议 **不要** 提交进 git（见 `.gitignore`）。
