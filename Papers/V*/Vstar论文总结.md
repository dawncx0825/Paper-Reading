# V* 论文总结

**V\*: Guided Visual Search as a Core Mechanism in Multimodal LLMs**
Penghao Wu（UC San Diego，实习于 NYU）、Saining Xie（New York University）
arXiv:2312.14135v2，2023-12-26，共 18 页（正文 9 页 + 附录 A/B/C）

---

## 0. 术语对照表

本篇实际用到的名词，中文只在第一次出现时给，正文里一律用英文原词。

| 英文 | 本篇含义 |
| --- | --- |
| visual search | 视觉搜索。在一幅内容繁多的图里找到并定位某个特定对象的过程 |
| MLLM（multimodal LLM） | 多模态大语言模型 |
| target | 保留英文。问题要用到、但当前看不清的那个对象，也就是搜索的目标 |
| grounding / visual grounding | 保留英文。把问题里指称的对象落实到图中的具体位置 |
| localization / localize | 定位。特指模型模块输出坐标或 heatmap 这个具体动作（与 grounding 区分：grounding 是任务层面的"对上"，localization 是模块层面的"给出位置"） |
| VWM（visual working memory） | 视觉工作记忆 |
| cue | 线索 |
| target-specific cue | 关于 target 本身的线索，模型直接看出"那一块像是 target" |
| contextual cue | 上下文线索，靠常识判断"这类 target 通常出现在什么位置" |
| search cue heatmap | 搜索线索热力图 |
| confidence score | 置信度分数 |
| priority score | 优先级分数 |
| sub-image / patch | 子图。V\* 把当前图像切开得到的那几块，论文两个词混用 |
| search length | 搜索长度。从整图出发到 target 所在子图为止的搜索步数 |
| vision encoder | 视觉编码器 |
| projection module | 投影模块。把视觉特征映射到 LLM 输入空间的那一层 |
| resampler | 保留英文。基于 cross-attention 的投影模块，会减少 token 数（256 → 32） |
| REC（referring expression comprehension） | 指代表达理解 |
| hallucination | 幻觉 |
| fixation | 注视点。人眼在搜索时依次落到的位置 |
| grokking | 保留英文。论文用它指损失在若干步之后突然掉到 0 的现象 |
| SEAL | 本文提出的元架构（meta-architecture），摘要里展开为 Show, sEArch, and TelL |
| V\* Bench | 本文提出的评测集 |

---

## 1. 这篇论文要解决什么问题

论文的出发点是认知科学里的一个观察：人在复杂场景里找东西不是把整幅画面一次看清，而是**有引导地搜索**。引导来自两个方面，一是 top-down feature guidance（自顶向下的特征引导），按颜色、形状、朝向这类属性把注意力导向符合描述的物体；二是 contextual scene guidance（场景上下文引导），靠"现实场景是有结构的"这一常识——物体一般不会随便乱放——直接把注意力投到可能的区域上，从而加快搜索。

当时的 MLLM 没有这套机制。论文指出的直接原因是它们都依赖一个预训练好、通常还冻住的 vision encoder（例如 CLIP 的图像编码器），而这个编码器的训练分辨率只有 224×224 或 336×336，部署时图像还会被缩到更低的分辨率。于是一张高分辨率、内容拥挤的图里，关键细节在进入 LLM 之前就已经丢了。更麻烦的是第二层问题：模型**不知道自己丢了什么**，既不能指出哪部分视觉信息缺失或不清楚，也没有办法主动去把它取回来。结果就是要么拒答，要么编一个答案出来（也就是 hallucination）。

论文举的例子（Figure 1）是问"杯子里的液体是什么颜色"。图的分辨率是 1920×830，杯子只占很小一块。GPT-4V 直接答了"粉色"，而正确答案是绿色。

作者同时指出，当时的 MLLM 评测集（MME、SEED-Bench、MMBench 这一类）追求的是任务类别覆盖广，题目里的视觉元素本身都比较大、比较显眼，因此不会把上面这个问题暴露出来。

论文的三项贡献对应三样东西：**SEAL** 是一个元架构，让 MLLM 能主动判断缺什么、再去找；**V\*** 是 SEAL 里那个 LLM 引导的搜索算法，可以在任意分辨率的图上做有引导的搜索；**V\* Bench** 是一个专门考察高分辨率图上细粒度 grounding 能力的评测集。

> 配图：`figures/figure01_SEAL整体流程.png`（论文 Figure 1）

---

## 2. 方法

SEAL 由两个部分组成：一个 **VQA LLM** 和一个 **visual search model**，两者不直接对话，而是通过 **VWM** 这块共享的存储交互。整体实例化见 Figure 3：左半边是 VQA LLM，右半边是 V\* 的搜索流程。论文用 LLaVA-7B 同时充当这两处的 MLLM。

> 配图：`figures/figure03_SEAL框架实例化.png`（论文 Figure 3）

### 2.1 整条流水线（Algorithm 1）

给定图像 I 和问题 T，VQA LLM 先只看整图特征，判断这些信息够不够回答问题。够就直接答，搜索机制不会被触发——这一点论文写得很明确，visual search 不是每次都跑。不够的话，它要输出一个**列表**，把回答问题所必需、但在整图特征里缺失或不够清楚的 target 一个个列出来。

然后初始化 VWM。VWM 有四个块：`<question>` 放原始问题文本，`<global image>` 放整张原图，`<searched targets>` 放搜索到的 target 裁剪图，`<target location>` 放这些 target 的坐标。

接着对列表里的每个 target 调用一次 V\* 搜索。找到了就从原图上把那块裁下来，连同坐标一起写进 VWM，写入的形式是 `"{target} <object patch> at location [x1, y1, x2, y2]"`；找不到就写一句 `"{target} not existent in the image"`。所有 target 处理完，VQA LLM 再读一遍 VWM 里的全部内容，给出最终回答。

### 2.2 VQA LLM 与 token 预算

VQA LLM 用 CLIP ViT-L/14 做特征提取，输入缩放并 padding 到 224×224；整图和 target 裁剪图都走这同一个编码器。投影模块准备了两种：linear layer 保留编码器输出的全部视觉 token，resampler 把 token 数压下来（256 → 32）。

因为 VWM 里可能同时存着整图和若干个 target，token 会很快堆起来，所以论文设计了一个切换规则：VWM 里没有任何 searched target 时，整图走 linear，保留全部 token；有一到两个 target 时，认为模型该把注意力放在这些 target 上，于是 target 走 linear、整图走 resampler；超过两个 target 时，全部走 resampler 压缩，以控制计算开销。

附录 A.3 给出了喂给 LLM 的实际输入序列：

```
<Image>
Additional visual information to focus on:
{Target Object 1's Name} <Object> at location [x1, y1, x2, y2];
{Target Object 2's Name} <Object> at location [x1, y1, x2, y2];
...
Question
```

其中 `<Image>` 是整图的特征 token，`<Object>` 是 VWM 里某个 target 的特征 token。

### 2.3 VQA LLM 的训练数据（387k）

VQA LLM 要学一件原本的 LLaVA 不会的事：承认自己看不清，并把缺的东西列出来。论文为此构造了三部分数据。

**负样本数据 100k**，教模型"说不知道 + 列出需要什么"。做法有两种：一是问图里根本不存在的一两个物体；二是问图里确实存在、但 bounding box 小于 20×20 的物体的细节——这种尺寸在 CLIP 编码器下基本看不到。这两种情况的标准答案都是直接说明无法回答，并列全所需的 target。数据基于 COCO2017，问题由 GPT-3.5 生成。

**VQA 数据 167k**，教模型在 VWM 里有 target 的情况下正确作答，由三块拼成：GQA 70k、物体属性 51k（来自 VAW 数据集）、空间关系 46k（用 COCO2017 生成两个物体相对位置的问题）。GQA 和属性这两块还额外做了一道筛选（附录 A.1）：先用 InstructBLIP 答一遍，只保留它能答对的题；再用图像 inpainting 模型 LaMa 把问题里提到的物体从图上擦掉，重新问一遍，只保留擦掉之后**答不对**的题。这样筛出来的题，其答案确实依赖那个具体物体，而不是靠语言先验猜出来的。

**LLaVA 指令微调数据 120k**，用来保住通用的多模态问答和指令跟随能力：LLaVA-80K 原样保留，另外从中抽出与 COCO 类别对应、且该图里有 box 标注的名词短语，把这些物体指定为 target，额外造出 40k。

### 2.4 visual search model 的结构

先说这个模型要解决的任务和 REC 的区别。REC 是给一句指代表达、在图里定位对应物体，但它面对的图尺寸固定；visual search 必须适配任意分辨率，有时还得把整幅图翻一遍才能找到，所以**效率本身是评价指标的一部分**——不只是找得准，还要找得快。最朴素的做法是把图切成大小一致的小块逐块检测（航拍图和病理全切片分析里常用），但分辨率一高就太慢。

V\* 的搜索模型由一个 MLLM 加一个定位模块构成。定位模块里有一个图像 backbone 和两个解码器：**D_tl**（target localization decoder）和 **D_cl**（search cue localization decoder）。MLLM 的词表里加了一个 `<LOC>` token。给定图像和一段文本表达，文本先被套进固定模板（`"Please locate the [object] in the image."`）再和图像一起喂进 MLLM，MLLM 输出 `<LOC>`，这个 token 的 embedding v_loc 携带了该表达对应的上下文和位置信息。v_loc 经两个各自独立的 MLP 得到 v_tl 和 v_cl，再分别和视觉编码器出来的 image token 一起送进 D_tl 和 D_cl：D_tl 输出 target 坐标和 confidence score，D_cl 输出 search cue heatmap。论文说 D_cl 的结构类似 SAM 的 mask decoder，D_tl 则是两个线性头，一个回归坐标、一个预测 confidence score。

> 配图：`figures/figure04_两个定位解码器.png`（论文 Figure 4）

训练细节在附录 A.3：MLLM 用 LLaVA-7B-v1.1，图像 backbone 用 OWL-ViT-B/16 的 vision encoder；D_cl 用分割损失（binary cross-entropy + DICE loss），D_tl 用类似 DETR 的 set prediction loss、坐标部分用 focal loss；整体训 100K 步、batch 64、学习率 1e-4。训练时 MLLM 用 LoRA、词嵌入层可训；定位模块的图像编码器和 D_tl 里的坐标 MLP 冻结，confidence score 的 MLP 和 D_cl 可训。

搜索模型的训练数据分两部分：检测与分割数据（COCO-Stuff、LVIS-PACO、refCOCO(+/g)、refCLEF 既当检测又当分割用，Objects365 v2 和 GoldG 只当检测用），以及 VQA 数据（possible location 问答 + LLaVA-80K）。三者采样比例 15:8:15。

这里的 **possible location 问答**是 contextual cue 能力的来源，构造方式值得单看（附录 A.2）：从 COCO2017 随机取图，对每张图取 2 个**不在这张图里、但出现在 CLIP embedding 最相似的 5 张图中**的物体，把这张图的 5 条 caption 和已有物体列表喂给 GPT-3.5，让它用常识回答这些物体最可能出现在哪里，答案要求是"参照图中已有实体的一句十个词以内的短语"（论文 Table 6 给了完整 prompt，示例是 `bird → in the sky`、`flag → on the roof of the building`）。也就是说，模型学的是在看不见目标的前提下、靠场景常识说出它该在哪儿。

### 2.5 V\* 搜索算法（Algorithm 2）

给定当前图像 I_p（第一轮就是整图）和 target 的文本表达 s：

第一步，直接让搜索模型定位 s，同时拿到坐标 + confidence score 和 search cue heatmap。confidence 够高就返回坐标，搜索到此结束。

第二步，没定位到就看那张 heatmap。如果 heatmap 的最大值超过阈值 δ，说明存在明显的 target-specific cue，直接拿它当作指导。如果没超过，就换一个问法问 MLLM：`"What is the most likely location of the [target] in the image?"`，让它用常识结合图像上下文给出一句 contextual cue（例如"玻璃杯最可能出现在餐桌上"），再把这句话里的名词短语拿去让 D_cl 定位，得到 contextual cue 对应的 heatmap。

第三步，把 I_p 递归切成 4 个不重叠、等大的子图。切法按长宽比定：宽大于高的两倍（landscape）竖着切成 4 条，高大于宽的两倍（portrait）横着切成 4 条，其余情况横竖各一刀切成 2×2，目的是让每个子图尽量接近方形。然后用上一步的 heatmap（target-specific 的或 contextual 的）给每个子图算 priority score，把（子图, 优先级）对放进优先队列，按优先级依次取出处理。

> 配图：`figures/figure05_子图划分方式.png`（论文 Figure 5）

这个过程递归进行，直到 target 被定位，或当前子图的尺寸小于预设阈值为止。论文自己在脚注里承认了一个边界情况：简单四等分会把正好卡在子图边界上的 target 切开，必要时可以改用重叠子图或按 heatmap 分布用变尺寸子图——但本文没有这么做。

阈值的具体取值在附录 A.4：confidence 用了两档，高阈值 0.5，整轮搜索都没有任何 target 越过它时，退而取全程 confidence 最高的那个，只要超过低阈值 0.3 就接受（V\* Bench 上的评测就是这么做的）。δ 取 `max(3.0, 6.0 × 0.7^l)`，l 是当前细分层级——即 l=0 时 6.0、l=1 时 4.2、l≥2 之后恒为 3.0，越往深处越容易相信 target-specific cue。另外论文说明，当前的搜索过程只找**单个** target，不做穷尽式地找全所有实例；但如果在整图上一次就直接定位到多个，会把它们都加进 VWM。

> 配图：`figures/figure07_搜索过程示例.png`（论文 Figure 7）

### 2.6 和 A\* 的类比

V\* 这个名字来自 A\*。论文给的对应关系是：子图相当于节点 n，代价函数 g(n) 对所有节点取同一个正常数，启发函数 h(n) 取 search cue heatmap 算出的 priority score 的相反数。区别在目标：A\* 求的是起点到终点的最小代价路径，V\* 只关心定位到目标所需的**总步数**最少。

论文同时把这套主动搜索归到 System II（慢思考）那一类认知过程——复杂任务需要动态分配计算量——并说它可以看作 CoT（chain-of-thought）在视觉侧的对应物。

---

## 3. 实验与主要结果

### 3.1 V\* Bench 是怎么建的

基于 SA-1B 数据集里的 **191 张**高分辨率图，平均分辨率 **2246×1582**。两个子任务：**attribute recognition** 115 题，问某个物体的某类属性（颜色、材质等）；**spatial relationship reasoning** 76 题，问两个物体的相对空间关系。图和题都由人工挑选和撰写，标准是不做准确的 grounding 就很难猜对。

为了能和开源模型定量比较，每题都配了选项：开放式问题 4 个选项，二选一问题 2 个选项，选项经人工复核以排除歧义。

附录 B 里还额外收了两个**探索性**子集，也归进 V\* Bench：**OCR** 30 题，需要识别图中物体上的文字或数字；**GPT-4V-hard** 17 题，是专门挑出来的"GPT-4V 答错而本文模型答对"的样本。

### 3.2 主结果（Table 1）

单位 %，Overall 是 191 题上的总体准确率（按 115:76 加权，与两个子任务的数字自洽）。

| 系统 | Attribute | Spatial | Overall |
| --- | --- | --- | --- |
| Human | 98.26 | 100.00 | 98.95 |
| Random Guess | 26.73 | 50.00 | 35.99 |
| BLIP2 | 26.95 | 53.94 | 37.69 |
| MiniGPT-4 | 30.43 | 50.00 | 38.22 |
| LLaVA | 23.47 | 53.94 | 35.59 |
| InstructBLIP | 25.21 | 47.36 | 34.02 |
| Otter | 26.95 | 56.57 | 38.74 |
| LLaVA-1.5 | 43.47 | 56.57 | 48.68 |
| MM-React | 34.78 | 51.31 | 41.36 |
| VisualChatGPT | 30.43 | 48.68 | 37.69 |
| Visprog | 31.30 | 56.57 | 41.36 |
| Bard | 31.30 | 46.05 | 37.17 |
| Gemini Pro | 40.86 | 59.21 | 48.16 |
| GPT-4V | 51.30 | 60.52 | 54.97 |
| **SEAL（本文）** | **74.78** | **76.31** | **75.39** |

要点：大部分 MLLM 贴着随机猜的水平；LLaVA-1.5 在 attribute 上比 LLaVA 明显好（23.47 → 43.47，Overall 35.59 → 48.68），论文把这部分归因于换了训练分辨率更高的 vision encoder（CLIP-ViT-L-336px），但离本文的搜索策略仍有相当距离；SEAL 只用 Vicuna-7B 就超过了 GPT-4V 二十个百分点，而人类接近满分，说明 MLLM 还有很大提升空间。代价方面，V\* 搜索在单张 A100 上平均每个 target 耗时 **6.0 秒**。

评测协议对两类系统不同：开源端到端模型用 likelihood 方式（取 log-likelihood 最高的选项），LLM-tool-using 系统和商用聊天机器人则直接要求它们输出选项。Bard 和 GPT-4V 走网页端（访问日期 2023-10-31），Gemini Pro 走 API（2023-12-16）。

### 3.3 消融：搜索机制是不是必需的（Table 2）

四组实验都用 Vicuna-7B，都在同一份训练数据上训。

| ID | VWM | Search | Attribute | Spatial | Overall |
| --- | --- | --- | --- | --- | --- |
| 1 | ✗ | N/A | 38.26 | 55.26 | 45.02 |
| 2 | ✓ | 直接查检测模型（GroundingDINO） | 62.60 | 61.84 | 62.30 |
| 3 | ✓ | 直接查检测模型（OWL-ViT） | 60.86 | 65.78 | 62.82 |
| 4 | ✓ | V\*-Search | 74.78 | 76.31 | 75.39 |

结论分两层：训练数据里已经包含属性和空间关系的题，但没有搜索机制的模型（ID 1）依然只有 45.02；把搜索换成现成的开放词表检测器（ID 2/3）能补回一大截，但仍明显不如 V\*，而且论文指出这类检测器用在超高分辨率图上本身就有工程困难。

### 3.4 搜索效率（Table 3、Table 4）

先在 V\* Bench 上比。论文记下了 V\* Bench 里全部 **245 个** target 位置，用 **search length** 衡量效率，只统计搜索后成功定位的样本。两个基线：Random 随机挑子图，Sequential 按倒序光栅扫描顺序挑；每个都分 DFS 和 BFS 两种。

| 策略 | Search Length ↓ |
| --- | --- |
| Random-DFS | 8.94 |
| Random-BFS | 7.18 |
| Sequential-DFS | 11.39 |
| Sequential-BFS | 6.62 |
| **LLM-guided visual search** | **4.65** |
| w/o target-specific cue | 5.22 |
| w/o contextual cue | 5.36 |

两种 cue 各去掉一个都会变慢，说明两者都在起作用。

再和人类注视行为比。数据用 COCO-Search18 的验证集，只取确实需要搜索才能定位的样本。做法是把人的 ground-truth fixation 序列转成二维 heatmap——在每个 fixation 点上叠一个高斯分布，并按点在序列中的次序给第 i 个点乘上权重 γ^i（0<γ<1）——再拿这张 heatmap 当作引导去跑同一套搜索流程。

| 策略 | Search Length ↓ |
| --- | --- |
| Random-DFS | 9.97 |
| Random-BFS | 4.90 |
| Sequential-DFS | 9.82 |
| Sequential-BFS | 4.20 |
| Human Fixation (γ=0.9) | 2.52 |
| Human Fixation (γ=0.8) | 2.70 |
| **LLM-guided visual search** | **2.80** |

V\* 达到了和人类注视相近的效率。注意这两张表的数字不能横着比——数据集不同，COCO-Search18 的图小得多，所以连基线都快很多。Figure 6 给了定性对比：人倾向于盯住中心区域或显著物体，而本文模型关注的是更大的上下文区域。

### 3.5 通用多模态能力有没有受损（Table 5）

LLaVA\* 指的是用本文同一份 VQA 训练数据训出来、但没有 VWM 和搜索机制的 LLaVA，也就是 Table 2 的 ID 1。

| | V\* Bench | MME | POPE | MMBench | SEED-Bench(Img) | MM-Vet | LLaVA_W |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LLaVA\* (7B) | 45.0 | 1051.2 | 76.5 | 34.4 | 41.8 | 30.4 | 62.6 |
| SEAL (7B) | 75.3 (+30.30) | 1128.9 (+77.70) | 82.4 (+5.85) | 33.1 (-1.36) | 41.7 (-0.17) | 27.7 (-2.70) | 59.1 (-3.50) |

综合评测 MME 提升，幻觉评测 POPE 也提升；MMBench 和 SEED-Bench 基本持平；MM-Vet 和 LLaVA-Bench(W) 略降，论文给了两个解释：这两个集规模小且用 GPT-4 打分，本身波动和偏差较大；另外它们有些题会触发对**图表类**目标的搜索，而搜索模型是在常见物体上训的，容易找不到。

附录 C 还记了一个现象：单独用 46K 空间关系数据训 VQA LLM 时，损失不是逐渐下降，而是在若干优化步之后**突然掉到 0**（论文称之为 grokking），说明模型是在某一刻才学会比较数值坐标来判断相对位置的。论文由此建议，如果要把这类数据混进更大规模的指令微调数据里，应该提高它的比例，否则可能凑不够触发这一跳变所需的优化步数。

---

## 4. 局限与待核实之处

**作者自己承认的：** 当前的搜索模型只针对自然图像和常见物体；要扩展到文档、图表、长视频或开放世界环境，需要额外训练和新的算法设计。论文还提了一个改进方向：引入基于卷积的模型来更高效地处理任意分辨率的图像。

**评测设置上值得注意的几点。** 第一，Table 1 里 SEAL 是在专门构造的属性 + 空间关系数据上训过的，其他模型都没有在这类数据上专门训练过，这个比较不是训练数据对齐的；真正对齐的是 Table 2 的 ID 1 和 Table 5 的 LLaVA\*（45.0 → 75.3）。论文安排了这两组实验，但正文没有把这层区别讲明。第二，开源模型走 likelihood、商用系统走直接选选项，两套协议的结果放在同一张表里并不严格可比。第三，V\* Bench 只有 191 题，1 题≈0.52 个百分点，几个点的差距对应的只是一两道题。第四，Human 那一行的 98.26/100.00 没有说明由几个人、怎么测的。

**GPT-4V-hard 子集的构造方式有选择偏差。** 这 17 题是按"GPT-4V 错、本文模型对"筛出来的，在它上面比较两者本身没有意义。论文把它归为探索性子集、也写明了构造方式，算是透明，但结果（Figure 10）仍然不能当作能力对比来读。另外这两个子集被"加进 V\* Bench"，而 Table 1 的 Overall 只覆盖那 191 题，别人复用这个 benchmark 时容易搞混。

**正文和附录对投影模块的描述互相矛盾。** §3.1.1 说 VWM 里有一到两个 target 时，target 走 linear、整图走 resampler；附录 A.3 说训练时**只有一个** target 才让 target 走 linear，否则是整图走 linear、target 走 resampler。两处的切换条件和谁走哪条路都对不上，论文没有说明这是推理与训练的有意差异还是笔误。

**记号不统一。** Figure 3 里两个定位解码器标成 M_tl 和 M_cl，正文和 Figure 4 里却是 D_tl 和 D_cl；VWM 的第四个块在正文里写作 `<target location>`，在 Figure 1 和 Figure 3 里写作 `<target locations>`。另外 SEAL 的展开在摘要里是 "Show, sEArch, and TelL"，在 §3 开头是 "Show, Search and Tell"。

**伪代码和正文没有完全对齐。** Algorithm 1 第 8 行调用 `Visual Search(q, s, δ)`，但整个 `SEALVQA(I, T, δ)` 里没有 `s` 这个变量，按上下文应该是循环变量 `target`。Algorithm 2 也没有体现正文说的"子图尺寸小于阈值就停"这一终止条件。

**和 A\* 的类比偏松。** 论文说 g(n) 对所有节点取同一正常数、h(n) 取 priority score 的相反数，但 Algorithm 2 里优先队列只按 heatmap 算出的优先级排序，没有任何随深度累积的代价项。这种形式更接近贪心最佳优先搜索（greedy best-first），而不是 f = g + h 意义上的 A\*。

**几个数字我没法从论文里核出来。** 一是 §5.3 说"记录了 V\* Bench 里全部 245 个 target 位置"：115 道属性题各 1 个 target、76 道空间题各 2 个 target，应该是 267 个，论文没有解释这 22 个的差额从哪来。二是 Table 5 里 POPE 的 76.5 → 82.4 标成 +5.85、MMBench 的 34.4 → 33.1 标成 -1.36、SEED 的 41.8 → 41.7 标成 -0.17，差值是按更高精度的原始数算的，和表里显示的一位小数对不上。三是附录 A.3 说 D_tl 里的坐标 MLP 在训练中冻结，而定位模块的图像 backbone 用的是 OWL-ViT 的 vision encoder——坐标头大概率是直接沿用 OWL-ViT 的预训练权重，但论文没有明说。

**Table 3 的比较口径有一处松动。** 论文说 search length 只统计"搜索后能成功定位"的样本，但不同策略成功定位的样本集合不一定相同，如果各行统计的是各自的成功子集，平均搜索长度就不是在同一批样本上比的。论文没有说明这一点是怎么处理的。

**参考文献 [53] 和 [54] 是同一篇。** 两条都是 Chenfei Wu 等人的 Visual ChatGPT，但 Table 1 里 MM-React 引的是 [53]、VisualChatGPT 引的是 [54]，MM-React 的引用应该是错的。（[32] 和 [33] 也是重复条目，同为 Mao et al.，这一条影响不大。）
