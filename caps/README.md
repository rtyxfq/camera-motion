# 运镜胶囊（Camera Capsule）

一个「胶囊」= 一份**测过、可复现、可直接用**的运镜配方。

方法可以一次写对；**配方只能一条条跑出来**。胶囊库才是这个 skill 长期价值所在。

## 目录

```
caps/
  CM-0001_push_in_reveal.md     # 一篇一个胶囊，文件名 = <编号>_<英文短名>.md
  CM-0002_....md
  INDEX.md                      # 由脚本生成，别手改
  evidence/                     # 测量结果 json、对照抽帧，纯文本优先
```

## 格式

```markdown
---
id: CM-0001
name: 缓推揭示
model: h3                        # 必须写，胶囊不跨模型通用
status: verified                 # verified / draft / retired
source:
  kind: reference_video          # reference_video | own_output | film | imagination
  ref: input/REF_CAM_xxx.mp4
  note: 从哪学的
measured:                        # 没有就整体 null，且 status 不能是 verified
  zoom_total: 1.175
  norm_x: 0.003
  norm_y: 0.065
  rot_deg_total: -1.85
  speed_norm_per_s: 0.028
  confidence: 中
h3: |                            # 可直接粘进 prompt 的正文
  The camera pushes in with small amplitude at slow speed, ...
when_to_use: 建立角色存在感后揭示环境；单镜 5~8 秒
avoid: 主体满屏特写时无效
verified_runs:
  - run: runs/20261109_xxx
    result: 实测 zoom 1.175，与声明一致
tags: [push_in, reveal, slow]
---
```

## 硬规则

| 字段 | 规则 |
|---|---|
| `id` | `CM-` + 4 位数字，永不复用、永不改号 |
| `model` | 必填。胶囊只对指定模型有效 |
| `status` | `verified` 必须有至少一条 `verified_runs` |
| `h3`（或对应模型的字段） | 只用该模型 `models/<model>.md` 里确认过的写法 |
| `measured` | 全填或整体 `null`，填了就要和 `evidence/` 里的数据对得上 |
| `source.kind` | `imagination` 最需要 `measured` 支撑（没有参考片可回溯） |

## 从参考片做一个胶囊

```bash
# 1. 测量
python tools/camera_motion.py --video ref.mp4 --out runs/xxx --sheet --emit-prompt

# 2. 看抽帧核对（必做）—— 测不出来 ≠ 没运镜
#    置信度低或「不可判」时，以人眼为准

# 3. 按 models/<model>.md 的映射写成该模型的写法，填进 measured

# 4. 跑一条验证，然后校验
python tools/camera_motion.py --video out.mp4 --check-prompt caps/CM-0001_xxx.md

# 5. 一致 → verified，登记 verified_runs
#    不一致 → draft，写清失败原因和下次怎么改
```

## 反面教材（本机实测，值得记住）

**同一骨架，一条照做一条没有。**
两条 prompt 共用含 "dollies in slowly, tilting up slightly" 的骨架：

- `L2T_S1` → 实测 **Static Shot**
- `L2T_S2` → 实测 **Push In + 放大 17.5%**

结论：**不校验就不知道有没有生效**。这也是 `status: verified` 必须配 `verified_runs` 的原因。

**滚转容易过头。**
一条同时含 push + tilt + roll 的 prompt，实测滚转累计 27.9°，观感是画面歪了。
→ 胶囊里滚转幅度要保守，且单独使用。
