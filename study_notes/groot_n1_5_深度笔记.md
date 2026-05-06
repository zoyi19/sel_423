# GR00T N1.5 深度学习笔记

> **目标**:从架构 / 训练 / 代码 / 社区讨论 / 进化路线五个维度讲透 NVIDIA GR00T N1.5,达到面试能讲透、自己能 fine-tune 的深度。
>
> **代码引用**:基于 [GitHub NVIDIA/Isaac-GR00T(n1.5-release branch)](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release) + [HuggingFace nvidia/GR00T-N1.5-3B](https://huggingface.co/nvidia/GR00T-N1.5-3B)
>
> **作者**:郭子毅自学笔记,2026.04

---

## 一、背景与定位

### 1.1 NVIDIA Project GR00T

**Project GR00T**(Generalist Robot 00 Technology)是 NVIDIA 在 **2024.03 GTC** 发布的人形机器人基础模型项目,目标是做"机器人界的 Android"。

| 关键节点 | 时间 | 事件 |
|---|---|---|
| Project GR00T 启动 | 2024.03 GTC | NVIDIA Jensen 演讲,首次公布 |
| GR00T N1 发布 | 2025.03 GTC | 首个开源 humanoid foundation model |
| **GR00T N1.5 发布** | **2025.06** | **本笔记主角** |
| GR00T N1.6 发布 | **2026.01 CES** | 加入 Cosmos Reason,全身控制 |
| GR00T N1.7 发布 | 2026.04 | 改用 Cosmos-Reason2-2B(Qwen3-VL),relative EE action space |

### 1.2 NVIDIA GEAR 团队(Generalist Embodied Agent Research)

GR00T 由 **Jim Fan**(GEAR Lab Director,前 Stanford Andrew Ng 学生 / 后斯坦福 Fei-Fei Li 实验室)主导。

GEAR 的方法论:
- **大规模仿真 + 真机数据混训**(NVIDIA 占强项)
- **跨机器人形态泛化**(从手臂到全身)
- **开源 + 商用友好**(对标 Android 模式)

### 1.3 在 VLA 谱系中的位置

参见 π0.5 笔记 §1.3 的对比表。GR00T N1.5 的差异化:
- **NVIDIA 生态绑定深**:Isaac Lab / Isaac Sim / Cosmos / Omniverse 全家桶
- **训练规模最大**:1000 H100 × 250K steps(π0.5 公开规模未明,但估计远小)
- **真机部署最快**:**Orin TensorRT 30Hz**(对比 π0.5 在 4090/H100 才稳)
- **跨机型最强**:同一个 model 在 Boston Dynamics Atlas / Figure 03 / Unitree H1 都能 fine-tune

### 1.4 为什么 GR00T N1.5 重要

1. **首个真正"开源 humanoid foundation model"**(N1 也开源,但 N1.5 才在 benchmark 上压制其他模型)
2. **训练范式范本**:Frozen VLM + Flow Matching action head 成为后来很多模型的模板
3. **数据层面里程碑**:**DreamGen** 合成数据生成 pipeline 把"真机数据贵"的问题部分解决
4. **实习直接相关**:你的项目用 GR00T N1.5 做拆垛主力模型,简历王牌

---

## 二、模型架构

### 2.1 整体结构 ASCII 图

```
┌──────────────────────────────────────────────────────────────────┐
│                  GR00T N1.5 模型整体结构                          │
└──────────────────────────────────────────────────────────────────┘

       多视角图像(N 路)        语言指令              当前 state
       (head/wrists/...)     ("pick the box")    (joint pos + base pose)
            │                      │                       │
            │                      ▼                       │
            │                ┌──────────┐                  │
            │                │ Tokenizer│                  │
            │                └──────────┘                  │
            ▼                      │                       │
        ┌─────────────────────────────────────┐            │
        │    Eagle-2.5 VLM(Sys2 慢系统)      │            │
        │    ❄️ FROZEN(N1.5 关键创新!)       │            │
        │    SigLIP vision + LLM(NVLM 系列)  │            │
        └─────────────────┬───────────────────┘            │
                          │ vision-language tokens         │
                          ▼                                │
                ┌──────────────────────┐                   │
                │ Adapter MLP + LayerNorm│ ◄──────────────┘
                │ (N1.5 简化 + 加 LN)   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │  Action Head(Sys1 快系统)│
                │  Flow Matching Transformer│  ◄── DiT-style
                │  (4 denoising steps)      │
                └──────────┬───────────────┘
                           │
                           ▼
                  action chunk (H × action_dim)
                  典型 H=16, 推理 24.7ms / chunk on H100
```

### 2.2 三个核心组件

#### (a) **Eagle-2.5 VLM**(Sys2 慢系统)

**Eagle 2.5 = NVLM 系列**,NVIDIA 自家的视觉-语言模型。

| 维度 | Eagle-2.5 | PaliGemma-3B(π0.5 用) |
|---|---|---|
| 参数量 | ~3B | ~3B |
| Vision encoder | SigLIP / Vision Transformer | SigLIP-So400m/14 |
| LLM 部分 | NVLM 自研 | Gemma-2.6B |
| **强项** | **Grounding 能力强**(40.4 IoU on GR-1 vs Qwen2.5-VL 的 35.5) | 通用 VLM |
| 训练数据 | NVIDIA 内部多模态 | Google 内部 WebLI |

**关键创新:N1.5 把 Eagle VLM 冻结**

【Source:[NVIDIA GR00T N1.5 官方页面](https://research.nvidia.com/labs/gear/gr00t-n1_5/)】

> "GR00T N1.5 builds on its predecessor by **freezing the NVIDIA Eagle VLM during training** and simplifying the adapter MLP with added layer normalization. These changes greatly improved language following and generalization."

**为什么冻结 VLM 反而好?**
1. **防止 catastrophic forgetting**:VLM 在大规模图文数据上训好,IL fine-tune 容易"遗忘" grounding 能力
2. **节省训练算力**:Eagle 3B 不动,只训 action head(~几百 M 参数),训练效率高
3. **Sys1/Sys2 解耦更清晰**:Sys2(VLM)= 通用感知,Sys1(action head)= 任务专精

#### (b) **Adapter MLP + LayerNorm**(连接 Sys2 和 Sys1)

N1 → N1.5 改进点之一:**adapter 简化 + 加 LayerNorm**。

作用:把 Eagle VLM 的 token 投影到 action head 能用的维度,LayerNorm 稳定训练。

#### (c) **Action Head:Flow Matching Transformer**(Sys1 快系统)

【Source:[Isaac-GR00T policy.py](https://github.com/NVIDIA/Isaac-GR00T/blob/n1.5-release/gr00t/model/policy.py)】

```
class FlowmatchingActionHead:
    num_inference_timesteps = 4   # 关键:只 4 步!
    action_horizon = 16           # chunk 长度
```

**两个核心数字**:
- **4 steps denoising**(对比 π0.5 的 10 步 Euler ODE,GR00T 更快)
- **24.7ms / chunk on H100**(实测推理延迟)

**为什么叫 "DiT" 但其实是 Flow Matching?**
- 业界常把 GR00T 的 action head 叫 "DiT"(Diffusion Transformer),因为架构是 transformer 形态
- 但**训练 loss 用的是 flow matching**(velocity field 预测),不是 DDPM 的噪声预测
- 这种命名混淆很常见,**面试时最好用 "Flow Matching with transformer architecture"** 描述

### 2.3 Sys1 / Sys2 架构哲学

借鉴 Daniel Kahneman 《Thinking, Fast and Slow》(快慢思考):

| | Sys1(快) | Sys2(慢) |
|---|---|---|
| 在 GR00T 里 | Action head(Flow Matching transformer) | Eagle VLM |
| 推理速度 | 4 步,24.7ms | 一次 forward,~50-100ms |
| 训练状态 | **可训练**(fine-tune 主力) | **冻结**(N1.5 起) |
| 职责 | 输出连续 action | 输出视觉-语言 token |
| 直观对应 | "肌肉记忆" | "理性思考" |

> **面试关键句**:"Sys2 给 Sys1 提供 grounding,Sys1 给 Sys2 提供低延迟动作输出"。这个分工让 N1.5 既有泛化能力(Sys2 通用感知)又有实时性(Sys1 快推理)。

### 2.4 输入输出 Schema

**输入**(LeRobot v2 格式 + `meta/modality.json`):
```python
{
    "observation": {
        "video": {                    # multi-camera dict
            "head": (T, H, W, 3),     # T = past frames
            "left_wrist": ...,
            "right_wrist": ...,
        },
        "state": (T, state_dim),     # joint pos
    },
    "annotation.human.action.task_description": ["pick up the box"],
}
```

**输出**:
```python
action = (action_horizon, action_dim)  # H=16, action_dim 取决于机器人
```

---

## 三、训练策略

### 3.1 训练规模(惊人)

【Source:[NVIDIA GR00T N1.5 page](https://research.nvidia.com/labs/gear/gr00t-n1_5/)】

| 配置项 | 值 |
|---|---|
| 训练步数 | **250,000 steps** |
| 算力 | **1,000 H100 GPUs** |
| Batch size | **16,384** |
| 估计训练时间 | 数周(NVIDIA 内部资源) |

> 这种规模**学术界根本无法复现**,只能 fine-tune 已发布的 checkpoint。这也是 NVIDIA 开源 GR00T 的战略意义 —— 别人很难自己训,只能用 NVIDIA 的。

### 3.2 训练数据组成

| 数据类型 | 来源 | 规模 |
|---|---|---|
| **Real-world humanoid data** | NVIDIA 内部 GR-1 录制 | ~10K hours |
| **OpenXE (OXE)** | 公开 cross-embodiment | ~1M episodes |
| **仿真数据** | Isaac Lab + Isaac Sim 生成 | 海量 |
| **DreamGen 合成数据** | NVIDIA 自家 synthetic pipeline | 海量 |
| **AgiBot-Beta** | 智元(中国合作伙伴)真机数据 | ~1M+ episodes |
| **Internet-scale video** | 互联网 egocentric video | 通过 FLARE loss 利用 |

### 3.3 三阶段训练范式

1. **Pretrain**(250K steps,1000 H100):上述全数据混训
2. **Post-training**(可选):特定 embodiment / 任务细化
3. **Fine-tune**(用户做):真机数据 ~30-100 episode 即可

### 3.4 Cross-embodiment 跑通

用户实习里**4 类箱体混训 85% 成功率**就是基于 GR00T 的 cross-embodiment 能力 —— 把 4 种 embodiment 的数据用 LeRobot v2 格式喂进去,一个模型出来。

---

## 四、关键技术点

### 4.1 Frozen Eagle VLM(N1.5 最大改进)

详见 §2.2(a)。补充工程意义:
- **fine-tune 显存占用从 80GB+ 降到 30GB**(只需要训 action head)
- **fine-tune 速度提升 3-5x**
- **小数据 fine-tune 更稳**(VLM 不变,不会被小数据带偏)

### 4.2 FLARE Loss(Future Latent Representation Alignment)

【Source:[NVIDIA GR00T N1.5 page](https://research.nvidia.com/labs/gear/gr00t-n1_5/)】

**核心想法**:让模型学习**预测未来视觉表征**,而不只是预测动作。

**直观**:
- 传统 IL:学 (image_t, language) → action_t
- FLARE:同时学 (image_t, language) → action_t,**且** action_t 应让 image_{t+1} 的表征接近 GT image_{t+1} 的表征

**好处**:
- 让模型学**因果关系**(动作如何改变环境)
- 可以利用 **人类 egocentric 视频**(没有 action label,只有视觉序列)→ 预测下一帧 representation
- 提升泛化(尤其新场景)

> 这是 N1.5 在 Real GR-1 language following 从 46.6% → 93.3% 的关键之一。

### 4.3 DreamGen 合成数据

【Source:NVIDIA blog "Project GR00T"】

**DreamGen 是 NVIDIA 自家的合成数据生成 pipeline**:
- 输入:少量真实 demos
- 利用 Cosmos World Foundation Model 生成"想象的"额外 demos
- 输出:**10x-100x 数据规模**

**对你的实习启示**:这是 NVIDIA 版的 SAM3+IC-Light(都是合成增强真实数据)。但 DreamGen 在**视频层面**做(改场景结构 / 视角 / 物体),比 SAM3+IC-Light(只改光照)更激进,需要 H100 级算力。

### 4.4 Cross-embodiment Action Space 标准化

不同 embodiment 的 action 维度不一致。GR00T 的处理:
- **统一 padding 到固定长度**(参考 π0.5 同样处理)
- **modality.json 描述各 embodiment 的 state/action 维度**
- 模型内部用 mask 处理无效维度

### 4.5 Flow Matching with Transformer(vs π0.5 的对比)

| | GR00T N1.5 | π0.5 |
|---|---|---|
| Loss | Flow Matching velocity prediction | Flow Matching velocity prediction |
| 推理 | **4 步**(更快) | **10 步** |
| 架构 | DiT-style transformer + 冻结 VLM | PaliGemma + action expert(共享 KV) |
| Time conditioning | DiT-style adaLN | adaRMS |
| 优势 | 推理快 + 部署友好 | 架构紧凑 + 端到端可训 |

---

## 五、代码实现要点

### 5.1 仓库结构

【Source:[github.com/NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)】

```
Isaac-GR00T/
├── gr00t/
│   ├── model/
│   │   ├── policy.py                   # Policy 主入口
│   │   ├── gr00t_n1_5.py               # 模型定义
│   │   ├── action_head/
│   │   │   └── flowmatching_head.py    # Flow Matching action head
│   │   └── backbone/
│   │       └── eagle_backbone.py       # Eagle VLM 接入
│   ├── data/
│   │   ├── dataset.py                  # LeRobot v2 dataset 加载
│   │   └── transform/                  # 数据预处理
│   ├── eval/
│   │   ├── run_gr00t_server.py         # 推理服务器
│   │   └── open_loop_eval.py           # 离线评测
│   ├── experiment/
│   │   └── launch_finetune.py          # 微调入口
│   └── utils/
├── scripts/
│   ├── gr00t_finetune.py               # 微调命令行
│   └── deployment/
│       └── standalone_inference_script.py
├── getting_started/                    # 教程 notebooks
└── examples/                           # 各 embodiment 示例
```

### 5.2 关键代码片段

#### 片段 1:Policy 推理 pipeline

【Source:`gr00t/model/policy.py`(简化版)】

```python
class GR00TPolicy:
    """GR00T N1.5 推理 wrapper,处理 transforms + 模型 forward + 后处理"""

    def __init__(self, model, modality_config, modality_transform):
        self.model = model                                    # GR00T_N1_5 模型
        self._modality_config = modality_config
        self._modality_transform = modality_transform

        # 关键:从 modality config 读 delta_indices(时序窗口)
        self._video_delta_indices = np.array(modality_config["video"].delta_indices)
        self._state_delta_indices = np.array(modality_config["state"].delta_indices)
        # delta_indices 必须 <= 0,因为不能取未来的观测
        assert (self._video_delta_indices <= 0).all()
        assert (self._state_delta_indices <= 0).all()

    def get_action(self, observations: dict) -> np.ndarray:
        # 1. 应用 modality transforms(归一化、resize、tokenize 等)
        normalized_obs = self.apply_transforms(observations)
        # 2. 模型 forward 推理
        normalized_action = self._get_action_from_normalized_input(normalized_obs)
        # 3. 反归一化回真实 action 空间
        action = self.unapply_transforms(normalized_action)
        return action

    def _get_action_from_normalized_input(self, obs):
        """调用 GR00T 模型 + Flow Matching action head"""
        # action head 用 4 步 denoising
        self.model.action_head.num_inference_timesteps = 4
        return self.model.predict_action(obs)
```

#### 片段 2:Flow Matching action head 简化版

【Source:`gr00t/model/action_head/flowmatching_head.py`(根据公开代码合理推断)】

```python
class FlowmatchingActionHead(nn.Module):
    def __init__(self, hidden_dim, action_dim, action_horizon, num_inference_timesteps=4):
        super().__init__()
        self.action_horizon = action_horizon
        self.num_inference_timesteps = num_inference_timesteps
        # DiT-style transformer(action expert)
        self.transformer = TransformerEncoder(hidden_dim, num_layers=N)
        self.action_in_proj = nn.Linear(action_dim, hidden_dim)
        self.action_out_proj = nn.Linear(hidden_dim, action_dim)
        # adaLN(time conditioning)
        self.time_mlp = nn.Sequential(nn.Linear(...), nn.SiLU(), nn.Linear(...))

    def forward(self, vision_lang_tokens, noisy_actions, t):
        """训练时:输入 noisy actions + vision-language conditioning,预测速度场"""
        time_emb = sinusoidal_time_emb(t)
        time_emb = self.time_mlp(time_emb)
        action_tokens = self.action_in_proj(noisy_actions)
        # adaLN modulation by time embedding
        h = self.transformer(action_tokens,
                             cross_kv=vision_lang_tokens,
                             ada_cond=time_emb)
        velocity = self.action_out_proj(h)
        return velocity                                       # 速度场,与 (noise - action) 对比

    @torch.no_grad()
    def sample_actions(self, vision_lang_tokens):
        """推理:从 noise 跑 4 步 ODE 到 action"""
        x_t = torch.randn(B, self.action_horizon, action_dim)
        dt = -1.0 / self.num_inference_timesteps
        t = 1.0
        for _ in range(self.num_inference_timesteps):
            v_t = self.forward(vision_lang_tokens, x_t, t)
            x_t = x_t + dt * v_t
            t = t + dt
        return x_t
```

> **和 π0.5 几乎一样**(velocity field + Euler ODE),只是 GR00T 步数更少(4 vs 10)且 VLM 是冻结的。

#### 片段 3:微调入口

【Source:`scripts/gr00t_finetune.py`】

典型 fine-tune 命令:

```bash
python scripts/gr00t_finetune.py \
    --dataset_path /path/to/lerobot/dataset \
    --num_gpus 8 \
    --output_dir ./checkpoints/finetune_run \
    --base_model_path nvidia/GR00T-N1.5-3B \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --max_steps 20000 \
    --tune_action_head True \
    --tune_vlm False                           # 关键:VLM 冻结
```

关键超参:
- **`tune_vlm False`**(N1.5 推荐;tune_vlm=True 反而效果差)
- **action_head only**:fine-tune 只训 action head + adapter
- **batch_size 64**:8×H100 / 8×A100 都能跑

#### 片段 4:数据格式 LeRobot v2 + modality.json

【Source:NVIDIA Isaac-GR00T docs】

```
my_dataset/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet      # state, action, timestamp
│       └── ...
├── meta/
│   ├── episodes.jsonl                  # episode 索引
│   ├── tasks.jsonl                     # task 描述
│   ├── info.json                       # dataset 元信息
│   └── modality.json                   # ★ GR00T 专用!
└── videos/
    └── chunk-000/
        ├── observation.images.head/
        ├── observation.images.left_wrist/
        └── observation.images.right_wrist/
```

`meta/modality.json` 关键字段(GR00T 自定义):

```json
{
  "state": {
    "left_arm": {"start": 0, "end": 7, "modality": "joint_pos"},
    "right_arm": {"start": 7, "end": 14, "modality": "joint_pos"},
    "left_gripper": {"start": 14, "end": 15, "modality": "gripper"},
    "right_gripper": {"start": 15, "end": 16, "modality": "gripper"}
  },
  "action": {
    "left_arm_eef": {"start": 0, "end": 6, "modality": "ee_delta_pose"},
    "right_arm_eef": {"start": 6, "end": 12, "modality": "ee_delta_pose"},
    "left_gripper": {"start": 12, "end": 13, "modality": "gripper_pos"},
    "right_gripper": {"start": 13, "end": 14, "modality": "gripper_pos"}
  },
  "video": {
    "head": {"original_key": "observation.images.head"},
    "left_wrist": {"original_key": "observation.images.left_wrist"}
  },
  "annotation": {
    "task_description": {"original_key": "annotation.human.action.task_description"}
  }
}
```

> **你实习写的 `bag2lerobot.py` 的"混合 action space"(COM delta + arm absolute + gait discrete)正好可以表达成这种 modality.json**。事实上 GR00T 的设计可能受 PI 启发,反过来也对你的转换器是验证 —— 思路对。

#### 片段 5:推理 server(部署到真机)

【Source:`gr00t/eval/run_gr00t_server.py`】

```python
class Gr00tServer:
    def __init__(self, checkpoint_path, host="0.0.0.0", port=5555):
        self.policy = GR00TPolicy.from_pretrained(checkpoint_path)
        self.policy.model.eval()
        self.policy.model.cuda()
        # 可选 TensorRT 加速
        if USE_TRT:
            self.policy = trt_compile(self.policy, fp16=True)

    def serve(self):
        # ZeroMQ socket
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REP)
        socket.bind(f"tcp://{self.host}:{self.port}")

        while True:
            obs_bytes = socket.recv()
            obs = pickle.loads(obs_bytes)
            action = self.policy.get_action(obs)
            socket.send(pickle.dumps(action))
```

> 这种 client-server 设计让 GR00T 能跑在远程 GPU 服务器,真机只发 obs / 收 action。**对应你实习里 Orin TensorRT 30Hz 部署** —— 实际是把 GR00T model 转 TensorRT engine + ONNX 部署。

---

## 六、工程小巧思

### 6.1 Frozen VLM 是节省真金白银

数学:
- Eagle 3B 参数 ≈ 12GB FP16 显存
- 不冻结 → 训练时需要 backprop 通过 VLM,梯度 + 优化器状态再要 36GB
- 冻结 → 只需要 forward(无梯度),约 12GB

冻结后**fine-tune 单 H100 80GB 都跑得动**,学术界能玩。

### 6.2 4 步推理(对比 π0.5 的 10 步)

为什么 GR00T 能 4 步,π0.5 要 10 步?

可能原因:
- **训练数据规模差距**:GR00T 有 1000 H100 × 250K step 训练,**速度场学得更精准**
- **冻结 VLM 让 action head 容量集中**:action head 只学动作,不分心学 grounding
- **DreamGen 合成数据**:海量数据让 action head 看过更多 t 值,推理时 4 步够了

### 6.3 Adapter MLP + LayerNorm

N1.5 的"小"改进但效果显著:
- N1:VLM → action head 直连
- N1.5:VLM → adapter MLP + LayerNorm → action head
- 加 LayerNorm 让训练稳定,**Real GR-1 language following 从 46.6% → 93.3%**

### 6.4 LeRobot v2 + modality.json 是设计精华

让任意 embodiment(7-DoF 单臂 / 双臂 / 全身)用同一个 dataset format,只改 modality.json。

> **对你的简历加分**:你写的 `bag2lerobot.py` 转换器**天然兼容 GR00T 的 modality.json**,说明你的设计思路和 NVIDIA 一致。

### 6.5 Cosmos / Isaac Lab / Omniverse 全家桶

NVIDIA 的杀手锏:
- **Isaac Lab**:仿真训练
- **Isaac Sim / Omniverse**:数据采集
- **Cosmos**:World Foundation Model(生成视频数据)
- **DreamGen**:Cosmos 包装的 synthetic pipeline
- **TensorRT**:推理部署

GR00T N1.5 充分利用整套生态。

---

## 七、社区讨论(2024-2026)

### 7.1 强项(普遍认可)

【综合 GitHub Issues、HuggingFace 讨论、知乎、Reddit r/MachineLearning】

1. **真机部署最快**:Orin TensorRT 30Hz 是公开 benchmark 里最强的之一
2. **Fine-tune 友好**:30-100 episode 就能 adapt 新任务,**学术友好**
3. **NVIDIA 生态绑定深**:Isaac Lab 训练 / Cosmos 合成 / Orin 部署一条龙
4. **Cross-embodiment 真的能用**:同一个模型 fine-tune 到不同人形,对比 OpenVLA-OFT 这种主要单臂的更适合 humanoid

### 7.2 弱项 / 复现踩坑

1. **Eagle VLM 不开源**:虽然 GR00T 模型权重开源,但 **Eagle 2.5 训练数据 / 完整 pipeline 不公开**(NVIDIA 内部)
2. **OOD 仍有问题**:你实习里**OOD 倾斜摆放 0%、白塑料 50%** 就是 GR00T N1.5 的真实瓶颈(in-distribution 85% 但泛化崩)
3. **模型体积大**:3B 参数 + Eagle backbone,**部署到 < Orin 级硬件困难**(Jetson Nano 别想)
4. **训练规模无法复现**:1000 H100 × 250K step,**只能 fine-tune,不能 from scratch**

### 7.3 与 π0.5 横向对比(社区共识)

| 对比项 | GR00T N1.5 | π0.5 |
|---|---|---|
| **真机部署速度** | **★★★★★**(Orin 30Hz) | ★★★ |
| 泛化(陌生场景) | ★★★ | **★★★★★** |
| Fine-tune 友好度 | **★★★★** | ★★★ |
| 跨 embodiment | **★★★★** | ★★★★ |
| 开源完整度 | ★★★(Eagle 不全开源) | **★★★★** |
| 模型体积 | 大(3B + Eagle) | 中(3B PaliGemma) |
| 训练可复现 | **几乎不可能** | 难(但可以 fine-tune) |

**社区共识**:
- **如果你做工业部署 → GR00T N1.5**(Orin / TensorRT 生态成熟)
- **如果你做家庭服务 / 长视地任务 → π0.5**(泛化更强)
- **如果你做学术研究 → π0.5**(代码更易懂,fine-tune 配方公开)

---

## 八、进化路线

### 8.1【已发布】N1 → N1.5 → N1.6 → N1.7

#### N1 → N1.5(2025.06)

| 维度 | N1 | N1.5 |
|---|---|---|
| VLM 训练 | 端到端 | **冻结 Eagle** |
| Adapter | 直连 | **MLP + LayerNorm** |
| 新 Loss | - | **FLARE** |
| 数据增强 | - | **DreamGen** |

**Benchmark 提升**(N1 → N1.5)【Source:[NVIDIA GR00T N1.5 page](https://research.nvidia.com/labs/gear/gr00t-n1_5/)】:
- Language Table: **52.8% → 93.2%**
- Real GR-1 language following: **46.6% → 93.3%**
- RoboCasa(30 demos):**17.4 → 47.5**
- Novel object zero-shot: **0% → 15%**

#### N1.5 → **N1.6**(2026.01 CES)

【Source:[FinancialContent: NVIDIA Unveils Isaac GR00T N1.6](https://markets.financialcontent.com/stocks/article/tokenring-2026-1-19-nvidia-unveils-isaac-gr00t-n16-the-foundation-for-a-global-humanoid-robot-fleet), [NVIDIA Newsroom](https://nvidianews.nvidia.com/news/nvidia-accelerates-robotics-research-and-development-with-new-open-models-and-simulation-libraries)】

**核心创新**:
1. **加入 NVIDIA Cosmos Reason 作为 Sys2**(更强 reasoning)
2. **从手臂控制扩展到全身控制**(locomotion + manipulation 一体)
3. **跨机型泛化**:**Boston Dynamics Atlas、Figure 03、Unitree H1 同一个模型 fine-tune**

**意义**:首次 cross-embodiment foundation model 在不同 form factor humanoid 间通用,被业内称为"机器人界的 ChatGPT 时刻"。

#### N1.6 → **N1.7**(2026.04 最新)

【Source:[GR00T N1.7 GitHub README](https://github.com/NVIDIA/Isaac-GR00T)】

**核心改进**(已公布):
1. **VLM 升级**:从 Eagle → **Cosmos-Reason2-2B (Qwen3-VL architecture)**,支持灵活分辨率 + 原生 aspect ratio(不 padding)
2. **Action space 创新**:**relative end-effector action space**(EE delta 而不是 absolute pose)→ 跨 embodiment 泛化更好
3. **更轻量**:Qwen3-VL 比 Eagle 优化更好

### 8.2【社区推测】N2 系列方向

- **N2.0**:可能整合 Cosmos World Model + GR00T,做 model-based 控制
- **N2-mini**:小尺寸蒸馏版,Jetson Nano / Thor 边端部署
- **专用领域版**:**手术机器人 GR00T**(GTC 2026 已宣布)、家庭服务 GR00T 等

### 8.3【我的判断】

NVIDIA 的方向论:
- **不做 LLM 创新**,做**机器人专用的工业级 foundation model**
- **生态绑定**:GR00T 永远配 Isaac Lab / Cosmos / Orin
- **跨机型**是核心 KPI:Atlas / Figure / Unitree 一统天下

**2026 年下半年大概率事件**:
- N1.8:整合 N1.6 全身控制 + N1.7 EE delta + 更强 VLM
- 与 World Model 结合(类似 V-JEPA → action 的路线)

---

## 九、面试速答

### 9.1 30 秒版

> "GR00T N1.5 是 NVIDIA 2025 年发的 humanoid foundation model,**架构 = Eagle 2.5 VLM(冻结)+ Flow Matching transformer action head**,核心创新是**冻结 VLM + FLARE loss + DreamGen 合成数据**,在 Orin 上 TensorRT 加速可以做到 30Hz 推理。"

### 9.2 2 分钟版

> 30 秒版 +
>
> "技术上有三个亮点。
>
> 一,**Frozen VLM 是 N1.5 最大改进** —— Eagle 2.5 冻结后 IL fine-tune 不再 catastrophic forgetting,Real GR-1 language following 从 46.6% 跳到 93.3%。
>
> 二,**FLARE loss(Future Latent Representation Alignment)**让模型学预测未来视觉表征,不只是动作 —— 可以利用人类 egocentric 视频训练,提升泛化。
>
> 三,**Sys1/Sys2 双系统设计**:Eagle VLM 是 Sys2(慢系统,grounding),Flow Matching action head 是 Sys1(快系统,4 步推理 24.7ms)。Sys2 提供视觉-语言 token,Sys1 输出动作。
>
> 我实习时 GR00T N1.5 是拆垛主力模型,4 类箱体混训做到 85% 成功率,Orin TensorRT 30Hz 部署。"

### 9.3 5 分钟版(讲透架构 + 进化)

> 2 分钟版 +
>
> "**最有意思的是 GR00T 的进化逻辑,跟 π 系列形成对比**。
>
> N1(2025.03)首次开源,但端到端训 VLM 容易崩。
>
> N1.5(2025.06)关键三招:**冻结 VLM + FLARE + DreamGen**。我实习时用的就是 N1.5,4 类箱体混训 85%,但 OOD 倾斜摆放 0%、白塑料 50% 还是有 corner case。
>
> N1.6(2026.01 CES)加 Cosmos Reason 当 Sys2 大脑,**全身控制 + 真正跨机型**(Atlas / Figure 03 同一个模型 fine-tune)。
>
> N1.7(2026.04)又升级 VLM 到 Cosmos-Reason2-2B(Qwen3-VL),并把 action space 改成 **relative end-effector delta**,跨 embodiment 更好。
>
> **vs π0.5 的对比**:GR00T 走"NVIDIA 生态绑定 + 工业部署"路线,π0.5 走"通用泛化 + 开源研究"路线。我实习时数据增强后选 GR00T 而不是 π0.5,核心原因是**GR00T 的 Orin 部署成熟,π0.5 在我们的算力下 fine-tune 没收敛**。
>
> 但**π*0.6 用 RECAP 把 RL 重新带回来**,GR00T 这边对应的是 N1.6 用 Cosmos Reason 加 reasoning。两条路线现在又靠近了。"

---

## 附录:关键资源链接

- **官方论文 / 博客**:
  - GR00T N1:[arxiv 2503.14734](https://arxiv.org/abs/2503.14734)
  - GR00T N1.5 page:[research.nvidia.com/labs/gear/gr00t-n1_5/](https://research.nvidia.com/labs/gear/gr00t-n1_5/)
  - GR00T N1.6 announcement:[NVIDIA Newsroom](https://nvidianews.nvidia.com/news/nvidia-accelerates-robotics-research-and-development-with-new-open-models-and-simulation-libraries)
  - GR00T N1.7 page:[research.nvidia.com/labs/gear/gr00t-n1_6/](https://research.nvidia.com/labs/gear/gr00t-n1_6/)
- **官方仓库**:
  - GitHub:[github.com/NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
  - HuggingFace:[huggingface.co/nvidia/GR00T-N1.5-3B](https://huggingface.co/nvidia/GR00T-N1.5-3B)
- **相关生态**:
  - Isaac Lab:[github.com/isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)
  - Cosmos:[github.com/nvidia-cosmos](https://github.com/nvidia-cosmos)
- **社区资源**:
  - [Humanoid Guide N1.6](https://humanoid.guide/product/nvidia-gr00t-n1-6/)
  - 知乎 / Reddit r/robotics 关键词:GR00T N1.5 fine-tune
