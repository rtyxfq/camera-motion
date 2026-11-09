# camera-motion

给 AI 视频生成模型用的**运镜方法论 skill**：让「运镜」这件事从随手写形容词，变成可测量、可校验、可复用的工程流程。

## 它解决什么

大多数 AI 视频 prompt 里的运镜是这么写的：

```
slow dolly in, cinematic, melancholic
```

这句话在三个层面都有问题，而且**写的人通常不知道**：

| 问题 | 例子 |
|---|---|
| 用了目标模型不认识的词 | `dolly in` ≠ 目标模型的 `Push In` |
| 缺了必要的修饰维度 | 没说幅度、没说速度，全交给模型猜 |
| 写完了不知道有没有生效 | 模型可能压根没推，你却在怪 prompt 不好 |

这个 skill 的职责就是把这三件事变成**可判定的**。

## 核心立场（请先读这三条）

**1. 它不增强任何运镜能力。**
这个 skill **不会**让模型学会它不会的运镜，**不会**提高推镜精度，也**不会**让 `dolly zoom` 变得可用。
它只做两件事：让指令更容易被遵守、让「有没有被遵守」可判定。
真正决定画面的是模型本身——这一点不能靠 prompt 工程绕过。

**2. 分层，模型专属知识不污染通用层。**

```
SKILL.md + references/   ← 通用：运动学定义、测量与校验方法（跨模型复用）
models/<model>.md        ← 专属：词表映射、量化阈值、失败模式（每个模型一份）
tools/camera_motion.py   ← 通用：测的是像素运动，不关心谁生成的
caps/                    ← 数据：一条条实测过的配方，持续增长
```

**3. 没有实测数据的数字不许写。**
任何「`large amplitude` 约等于 22% 画面位移」这类论断，必须有标定记录支撑；
没标定的必须显式标注 `⬜ 未标定`。宁可承认不知道，也不给虚假的确定感。

## 具体处理哪些运镜

按**能给出的确定性**分四档。这张表是判断「这个 skill 现在能帮你多少」的依据，
也是它诚实程度的体现——**能确定的只有 22 条，其余还在等实测**。

### ✅ 直接对应（14 条，可查证官方词表）

换词即可，语义无损：

| 你可能会写 | 写成 |
|---|---|
| dolly in / slow push | `Push In` |
| dolly out / pull back | `Pull Out` |
| zoom in / out | `Zoom In` / `Zoom Out` |
| pan left / right | `Pan Left` / `Pan Right` |
| truck / track left / right | `Truck Left` / `Truck Right` |
| tilt up / down | `Tilt Up` / `Tilt Down` |
| pedestal / boom up / down | `Pedestal Up` / `Pedestal Down` |
| arc / orbit / 360 | `Arc Shot` |
| follow / tracking / 跟拍 | `Tracking Shot` |
| static / locked off / 固定机位 | `Static Shot` |
| handheld / slight shake | `Shake Slightly` |
| shaky cam / 剧烈抖动 | `Shake Strongly` |
| first person / 主观视角 | `POV` |
| roll / dutch roll | `Roll Clockwise` / `Roll Counterclockwise` |

### 🔶 有近似词（8 条，可用但语义有损）

| 你可能会写 | 退而求其次 | 丢了什么 |
|---|---|---|
| crash zoom / snap zoom | `Zoom In` + `large amplitude` + `at fast speed` | 突变感 |
| crane shot / jib up | 近似 `Pedestal Up` | crane 是弧线，Pedestal 是纯垂直，**不等价** |
| side tracking shot | `Truck Left` / `Truck Right` | 无 |
| top-down tracking | `Tracking Shot` + 构图文字 | “俯视”要靠画面描述补 |
| sweeping shot | `Pan` / `Truck` + `large amplitude` | 无专门词 |
| steadicam / gimbal | `Static Shot` 或省略 | 平稳无需专门词；**别用 `Shake` 表达“稳”** |
| leading shot | `Tracking Shot` | 方向关系靠文字补 |
| reveal shot（掠过前景揭晓） | `Truck` + 前景描述 | 揭晓感

### ❌ 官方确认未列（14 条，**只代表没列，不代表做不到**）

`dolly zoom` / `bullet time` / `drone dive` / `FPV` / `hyperlapse` /
`barrel roll` / `corkscrew` / `vortex` / `snorricam` / `underwater drift` /
`slider` / `chase camera` / `stop-motion camera` / `long take`

⚠️ 官方原文用的是 "Available Expression"，**从未说这是封闭集合**。
所以这一档的正确读法是「官方没提供专用词」，不是「模型一定做不到」。

### ⬜ 未知，等实测

包括 `whip pan`——**它在本机 3 个工作流里真实在用，但从未验证过是否生效。**
这是当前风险最高的一项：正在使用，却可能一直是空的。

### 目前能确定的边界

| | |
|---|---|
| 能确定 | 22 条（✅14 + 🔶8） |
| 官方未列 | 14 条，模型是否仍能理解**未知** |
| 幅度/速度量化 | ⬜ 全部未标定 |
| 通用运镜库覆盖 | cinematic B 节 45 条中 37 条（82%） |

**所以现在的正确用法是：用它防止写错词、补全维度、事后校验；
不要指望它告诉你「`large amplitude` 到底给几个百分点」——那个还没测。**

## 快速上手

```
①  测量（参考片或自己的成片）
     python tools/camera_motion.py --video ref.mp4 --out runs/xxx --sheet --emit-prompt

②  查映射，把通用说法翻成目标模型认的写法
     → models/<你的模型>.md

③  跑完再测一次，核对 prompt 声称的运镜到底有没有发生
     python tools/camera_motion.py --video out.mp4 --check-prompt 胶囊.md

④  验证通过的配方沉淀成胶囊
     → caps/README.md
```

## 支持哪些模型

| 模型 | 状态 | 词表 | 标定 |
|---|---|---|---|
| MiniMax H3 | 可用 | ✅ 有 | ⬜ 未做 |
| 其他 | — | 照 `models/_template.md` 自己加 |

**用之前必须先声明目标模型。** 不同模型的词表不通用，套错会削弱效果。

## 已知局限

- **2D 画面分不清 Pan 与 Truck、Tilt 与 Pedestal**（都是内容整体平移）。工具会给双候选，由人按原片定。
- **主体占满画面时测不出运镜**（背景不可见）。工具会明确「拒绝作答」而不是给个错数。
- **模型可能不听话**。实测过：同一条 prompt 骨架，一条照做了推镜，另一条产出是静止的。
  这不是测量错误，是模型行为——所以校验环节不可省。
- **它不能扩展模型能力**。上面 ❌ 那 14 条里，没有一条会因为用了这个 skill 而变得可用。
  想拍 `dolly zoom` 这类效果，只能换模型或后期做，prompt 层面无解。

## 路线图

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | 词表映射（本文档「具体处理哪些运镜」） | ✅ 完成 |
| P1 | 标定：幅度/速度实际对应量、`whip pan` 是否有效、默认是否漂移 | ⬜ 待做 |
| P2 | 首批胶囊（实测配方） | ⬜ 待做 |
| P3 | 效能验证：用 skill vs 不用 skill，生效率对比 | ⬜ 待做 |

P1 是分水岭——做完之后，上面 ⬜ 和「未标定」才会变成具体数字。

## 授权

MIT
