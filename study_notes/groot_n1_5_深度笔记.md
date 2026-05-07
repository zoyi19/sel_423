# GR00T 系列深度学习笔记(精简版,演进到 N1.7)

> **代码引用**:[github.com/NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) + [HuggingFace nvidia](https://huggingface.co/nvidia)
> **作者**:郭子毅,2026.05 更新

---

## 一、一句话定位 + Timeline

### 一句话

> **GR00T = NVIDIA 的开源 humanoid foundation model**。**双系统架构(Sys1 快 + Sys2 慢)+ 大规模工业训练 + 跨机型泛化**。

### GR00T 系列 Timeline(更新到 N1.7)

| 版本 | 时间 | 关键创新 | 一句话 |
|---|---|---|---|
| N1 | 2025.03 GTC | 首个开源 humanoid VLA + 双系统 | 框架奠基 |
| **N1.5** | **2025.06** | **冻结 Eagle + FLARE + DreamGen** | 性能跳跃(46.6%→93.3% language following) |
| **N1.6** | **2026.01 CES** | **Cosmos Reason 接 Sys2 + 全身控制 + 跨机型** | "机器人 ChatGPT 时刻"(Atlas/Figure/Unitree) |
| **N1.7** | **2026.04** | **Cosmos-Reason2-2B + EgoScale + scaling law** | **20K 小时人类视频证明 VLA 的 scaling law** |

### 在 VLA 谱系的位置

- **NVIDIA 生态绑定深**:Isaac Lab / Isaac Sim / Cosmos / Omniverse 全家桶
- **训练规模最大**:N1.5 用 1000 H100 × 250K step,batch size 16384
- **真机部署最快**:Orin TensorRT **30Hz**(同代 π0.5 在 4090/H100 才稳)
- **跨机型最强**:N1.6 起 Atlas / Figure 03 / Unitree H1 同模型 fine-tune

---

## 二、模型架构(简版)

```
multi-cam images   language    state(joint pos + EEF poses)
     │                │                  │
     └──────┬─────────┘                  │
            ▼                            │
   ┌────────────────────────┐            │
   │ Sys2 慢系统 ❄️ FROZEN   │            │
   │ N1: Eagle 2.0          │            │
   │ N1.5: Eagle 2.5        │            │
   │ N1.6: Cosmos Reason    │            │
   │ N1.7: Cosmos-Reason2-2B│            │
   │       (Qwen3-VL 架构)   │            │
   └───────────┬────────────┘            │
               │ vision-language tokens  │
               ▼                         │
       ┌──────────────────────┐ ◄────────┘
       │  Adapter MLP + LN    │ ◄── time emb (adaLN)
       │  (N1.5 简化加 LN)     │
       └──────────┬───────────┘
                  ▼
       ┌──────────────────────────┐
       │ Sys1 快系统(可训练)      │
       │ DiT-style + Flow Matching │
       │ (32 层 DiT in N1.7)       │
       │ 4 步去噪 → 24.7ms/chunk    │
       └──────────┬───────────────┘
                  ▼
            action chunk (B, 16, action_dim)
```

**关键设计哲学(Sys1 / Sys2 解耦)**:
- **Sys2** = VLM(理解 + grounding + reasoning),**慢但聪明,5Hz 异步**
- **Sys1** = action head(去噪 → 连续动作),**快,30Hz 同步**
- **Sys2 给 Sys1 提供 vision-language tokens,Sys1 提供低延迟动作输出**

---

## 三、三大关键技术点

### 3.1 Frozen VLM(N1.5 起的核心改进)

- N1:VLM 端到端训(容易 catastrophic forgetting)
- **N1.5 起冻结 Eagle**(N1.6 / N1.7 沿用)
- **效果**:Real GR-1 language following **46.6% → 93.3%**
- **工程**:训练显存从 80GB+ 降到 30GB,**单 H100 80GB 都跑得动**

### 3.2 FLARE Loss(N1.5 引入)

**FLARE = Future Latent Representation Alignment**

- 同时学:动作 + 预测未来视觉 representation
- 让模型学**因果关系**(动作如何改变环境)
- **可利用人类 egocentric 视频**(无 action 标签也能学)

### 3.3 跨 embodiment 设计(modality.json)

LeRobot v3.0 + GR00T 专用 `meta/modality.json` 描述每个 robot 的:
- state 维度结构(joint_pos / gripper)
- action 维度结构(eef_delta_pose / gripper_pos)
- video 来源映射

→ 同一个 base model **改 modality.json 就能 fine-tune 不同 robot**(从 N1.6 起 Atlas/Figure/Unitree 通用)。

---

## 四、训练 + 推理代码(完整保留)

### 4.1 Policy 推理 pipeline

```python
class GR00TPolicy:
    """GR00T 推理 wrapper:transforms + 模型 forward + 后处理"""

    def __init__(self, model, modality_config, modality_transform):
        self.model = model
        self._modality_config = modality_config
        self._modality_transform = modality_transform

        # 时序窗口(delta_indices 必须 <= 0,不能取未来观测)
        self._video_delta_indices = np.array(modality_config["video"].delta_indices)
        self._state_delta_indices = np.array(modality_config["state"].delta_indices)
        assert (self._video_delta_indices <= 0).all()
        assert (self._state_delta_indices <= 0).all()

    def get_action(self, observations: dict) -> np.ndarray:
        # 1. 应用 modality transforms(归一化 / resize / tokenize)
        normalized_obs = self.apply_transforms(observations)
        # 2. 模型 forward
        normalized_action = self._get_action_from_normalized_input(normalized_obs)
        # 3. 反归一化回真实 action
        action = self.unapply_transforms(normalized_action)
        return action

    def _get_action_from_normalized_input(self, obs):
        self.model.action_head.num_inference_timesteps = 4   # 4 步去噪
        return self.model.predict_action(obs)
```

### 4.2 Flow Matching action head(简化版)

```python
class FlowmatchingActionHead(nn.Module):
    """GR00T 的 action head:DiT-style transformer + flow matching loss"""

    def __init__(self, hidden_dim, action_dim, action_horizon, num_inference_timesteps=4):
        super().__init__()
        self.action_horizon = action_horizon
        self.num_inference_timesteps = num_inference_timesteps
        # DiT-style transformer(32 层 in N1.7)
        self.transformer = TransformerEncoder(hidden_dim, num_layers=32)
        self.action_in_proj  = nn.Linear(action_dim, hidden_dim)
        self.action_out_proj = nn.Linear(hidden_dim, action_dim)
        self.time_mlp = nn.Sequential(nn.Linear(...), nn.SiLU(), nn.Linear(...))

    def forward(self, vision_lang_tokens, noisy_actions, t):
        """训练:输入 noisy actions + VL conditioning,预测速度场"""
        time_emb = sinusoidal_time_emb(t)
        time_emb = self.time_mlp(time_emb)
        action_tokens = self.action_in_proj(noisy_actions)
        # adaLN modulation by time(类似 π0.5 adaRMS,但用 LayerNorm)
        h = self.transformer(action_tokens,
                             cross_kv=vision_lang_tokens,
                             ada_cond=time_emb)
        velocity = self.action_out_proj(h)
        return velocity     # 速度场,与 (noise - action) 对比

    @torch.no_grad()
    def sample_actions(self, vision_lang_tokens):
        """推理:从 noise 跑 4 步 ODE 到 action"""
        x_t = torch.randn(B, self.action_horizon, action_dim)
        dt = -1.0 / self.num_inference_timesteps    # 注意负号
        t = 1.0
        for _ in range(self.num_inference_timesteps):     # 4 步
            v_t = self.forward(vision_lang_tokens, x_t, t)
            x_t = x_t + dt * v_t
            t = t + dt
        return x_t
```

> **和 π0.5 几乎一样**(velocity field + Euler ODE),只是 GR00T **4 步**(vs π0.5 10 步)且 VLM 是冻结的。

### 4.3 训练 step(N1.5 起范式)

```python
def gr00t_train_step(model, batch):
    """
    GR00T 训练:Sys2 冻结,Sys1(DiT)学预测速度场
    
    数据:LeRobot v3.0 + modality.json
      - N1.5 起:真机遥操 + 仿真 + DreamGen 合成
      - N1.7 起:加 EgoScale 20K 小时人类第一人称视频
    """
    obs        = batch['observation']
    action_gt  = batch['action_chunk']
    instruction = batch['language']

    # Sys2 forward(冻结,no_grad 省显存)
    with torch.no_grad():
        vl_tokens = model.sys2_vlm(obs['images'], instruction)

    # Adapter(可训,N1.5 起加 LayerNorm)
    adapted = model.adapter(vl_tokens)

    # === Flow Matching loss ===
    # 1. 采时间步 + 噪声
    B, H, A = action_gt.shape
    t = torch.randint(0, T, (B,))
    noise = torch.randn_like(action_gt)

    # 2. 直线插值加噪
    sqrt_a   = model.action_head.alphas[t].sqrt().view(-1, 1, 1)
    sqrt_1ma = (1 - model.action_head.alphas[t]).sqrt().view(-1, 1, 1)
    action_t = sqrt_a * action_gt + sqrt_1ma * noise

    # 3. DiT 预测速度场
    velocity_pred = model.action_head(
        action_t, t,
        vision_lang_tokens=adapted,
        state=obs['state'],
    )

    # 4. MSE on velocity field
    fm_loss = F.mse_loss(velocity_pred, noise - action_gt)

    # === FLARE loss(N1.5 起,N1.7 也用)===
    # 同时学预测未来视觉 representation
    with torch.no_grad():
        vl_tokens_future = model.sys2_vlm(obs['images_future'], instruction)
    future_pred = model.flare_predictor(adapted)
    flare_loss = F.mse_loss(future_pred, vl_tokens_future)

    return fm_loss + 0.1 * flare_loss
```

### 4.4 N1.7 推理(异步 30Hz 部署)

```python
def deploy_at_30hz(model, robot, instruction):
    """
    N1.6 / N1.7 部署:Sys2 异步低频(5Hz),Sys1 同步高频(30Hz)
    """
    cached_vl = None
    last_sys2_time = 0

    while True:
        obs = robot.get_obs()
        now = time.time()

        # Sys2 低频(5Hz):reasoning 重,缓存复用
        if now - last_sys2_time > 0.2:
            cached_vl = model.sys2_vlm(
                obs['images'],          # N1.7 支持任意分辨率,不用强制 resize
                instruction,
                obs['state'],
            )
            last_sys2_time = now

        # Sys1 高频(30Hz):4 步去噪
        action_chunk = model.action_head.sample(
            condition=cached_vl,
            state=obs['state'],
            num_inference_timesteps=4,
        )
        robot.execute(action_chunk[0, 0])     # 取 chunk 第一步

        time.sleep(0.033)                      # 30Hz
```

**讲述要点**:
1. **Sys2 低频缓存复用** — 节省算力,reasoning 不需要每帧跑
2. **Sys1 高频** — 控制实时性,4 步 DiT denoising ~25ms
3. **N1.7 任意分辨率** — 不用强制 resize 到 224×224

### 4.5 微调入口(N1.7 官方命令)

```bash
# N1.7 一行启动微调
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7 \
    --dataset-path <YOUR_LEROBOT_DATASET> \
    --embodiment-tag UNITREE_G1 \
    --modality-config-path <YOUR_MODALITY_JSON> \
    --num-gpus 1 \
    --output-dir ./finetune_output \
    --max-steps 2000 \
    --global-batch-size 32

# 支持 embodiments:UNITREE_G1, LIBERO_PANDA, OXE_WIDOWX
# 数据格式:LeRobot dataset(沿用 N1.5 起的 modality.json 机制)
```

**关键超参**:
- **`tune_vlm=False`**(N1.5 起默认,Sys2 冻结)
- **action_head only**:fine-tune 只训 DiT + adapter
- **从 N1.6 起跨机型 fine-tune**:base model 不变,只改 modality config

---

## 五、GR00T 系列进化路线(详解)

### N1(2025.03 GTC)

- 首个开源 humanoid foundation model
- 双系统架构奠基(Sys1 / Sys2 解耦)
- VLM = Eagle 2.0
- 性能不算特别强,但**框架范式立住了**

### N1.5(2025.06)三件套

【Source:[NVIDIA GR00T N1.5 page](https://research.nvidia.com/labs/gear/gr00t-n1_5/)】

1. **冻结 Eagle 2.5 VLM**(反直觉但有效)
2. **加 FLARE loss**(预测未来 latent representation)
3. **DreamGen 合成数据**(基于 Cosmos World Model 生成)

**Benchmark 提升**(N1 → N1.5):
- Language Table:**52.8% → 93.2%**
- Real GR-1 language following:**46.6% → 93.3%**
- RoboCasa(30 demos):**17.4 → 47.5**
- Novel object zero-shot:**0% → 15%**

**训练规模**:**1000 H100 × 250K step,batch size 16384**(学术界根本不可能复现)。

### N1.6(2026.01 CES)

【Source:[NVIDIA Newsroom](https://nvidianews.nvidia.com/news/nvidia-accelerates-robotics-research-and-development-with-new-open-models-and-simulation-libraries)】

**三大改动**:
1. **Sys2 升级为 Cosmos Reason**(NVIDIA 自研 reasoning model)
2. **从手臂操作 → 全身控制**(locomotion + manipulation 一体)
3. **跨机型泛化**:Boston Dynamics Atlas / Figure 03 / Unitree H1 同 base 微调

**意义**:首次跨 form factor humanoid 通用 base,被业界称为**"机器人界 ChatGPT 时刻"**。

**checkpoint 已发布**:[`nvidia/GR00T-N1.6-3B`](https://huggingface.co/nvidia/GR00T-N1.6-3B)(~6GB)

### N1.7(2026.04 最新)

【Source:[HuggingFace blog](https://huggingface.co/blog/nvidia/gr00t-n1-7)】

**三大升级**:

1. **VLM 升级到 Cosmos-Reason2-2B**(基于 **Qwen3-VL 架构**,2B 参数)
   - **任意分辨率原生支持**(不再 padding 浪费 token)
   - 推理能力增强(任务分解 + 多步推理)

2. **训练数据从遥操数据 → EgoScale 20,854 小时人类第一人称视频**
   - 包含自我相机 / 腕部相机 / 手部追踪
   - 20+ 任务类别(制造业 / 零售 / 医疗 / 家居)

3. **首次发现机器人灵巧度的 scaling law**
   - 1k 小时 → 20k 小时,**平均任务完成率 > 2×**
   - **VLA 圈的"GPT moment"**:数据规模主导能力

**支持 embodiments**:UNITREE_G1, LIBERO_PANDA, OXE_WIDOWX
**验证平台**:Unitree G1, Bimanual YAM, AGIBot Genie 1
**灵巧手**:22 DoF(支持 contact-rich tasks)
**架构**:32 层 DiT(Action Cascade)+ 4 步推理

**Drop-in 替换 N1.6** — 改 `--base-model-path nvidia/GR00T-N1.7` 一行切换。

---

## 六、5 分钟话术(口头表达,可背)

> **场景**:面试被问"详细讲讲 GR00T 系列你了解的"。**说话节奏:背景 30s + 架构 1min + 创新 1.5min + 进化 1.5min + 反思 30s**

### 【0:00–0:30】开场 + 背景

"GR00T 是 NVIDIA 在 **2024.03 GTC 发布的人形机器人基础模型项目**,由 **GEAR Lab(Generalist Embodied Agent Research)的 Jim Fan 主导**。NVIDIA 想做'机器人界的 Android' —— **靠开源 + 生态绑定 + 大规模工业训练** 占位人形 VLA 市场。"

### 【0:30–1:30】核心架构(1 分钟)

"GR00T 的灵魂是 **Sys1 / Sys2 双系统解耦**,借鉴 Daniel Kahneman《思考,快与慢》。

**Sys2 慢系统是 VLM**(N1.5 用 Eagle 2.5,N1.6 用 Cosmos Reason,**N1.7 用 Cosmos-Reason2-2B,基于 Qwen3-VL 架构**),负责语言理解 + 视觉 grounding + 任务推理,**异步 5Hz 跑**,推理重但聪明。

**Sys1 快系统是 DiT-style action head + Flow Matching**,32 层 transformer,**只跑 4 步去噪就出 chunk**,推理 24.7ms,**Orin TensorRT 部署 30Hz**。

部署时 Sys2 缓存复用,Sys1 高频 — 这是 GR00T **30Hz 实机** 的工程关键,π0.5 因为没显式分层做不到这种异步。"

### 【1:30–3:00】关键技术点(1.5 分钟)

"GR00T N1.5 起有三大关键创新,让性能跳跃式提升:

第一,**冻结 VLM**。N1 是端到端训,容易 catastrophic forgetting。**N1.5 起冻结 Eagle**,Real GR-1 language following 直接从 46.6% 跳到 93.3%,**显存从 80GB 降到 30GB**,fine-tune 单 H100 都能跑。这个反直觉决定是 NVIDIA 工程哲学的体现 —— **大脑不要动,只调动作**。

第二,**FLARE loss**:Future Latent Representation Alignment。**同时学动作 + 预测未来视觉 representation**,让模型理解'动作如何改变环境'的因果关系。还有个好处是**可以利用无 action 标签的人类 egocentric 视频**。

第三,**LeRobot v3.0 + modality.json 跨机型设计**。同一个 base model,**改 modality.json 就能 fine-tune 不同 robot**(从 N1.6 起 Atlas/Figure/Unitree 通用)。

训练规模上:**N1.5 用了 1000 H100 × 250K step,batch size 16384** —— 学术界根本不可能复现,这就是 NVIDIA 开源的战略意义,**别人只能 fine-tune 不能 from scratch**。"

### 【3:00–4:30】进化路线(1.5 分钟)

"GR00T 演进我能完整讲:

**N1(2025.03)** 双系统范式奠基,VLM 是 Eagle 2.0。

**N1.5(2025.06)** 性能跳跃版,三件套(冻结 Eagle 2.5 + FLARE + DreamGen 合成数据),**Real GR-1 language following 46.6% → 93.3%**。我实习用的是这个版本,4 类箱体混训做到 85% 成功率。

**N1.6(2026.01 CES)** 是"机器人 ChatGPT 时刻" —— **加 Cosmos Reason 进 Sys2**,从手臂扩到全身控制,**首次跨机型(Atlas、Figure 03、Unitree H1 同 base)**。HuggingFace 上有 nvidia/GR00T-N1.6-3B。

**最新的 N1.7(2026.04)** 我觉得最有意思 —— VLM 升级到 **Cosmos-Reason2-2B**(基于 Qwen3-VL 架构),**任意分辨率原生支持**,腕部相机不用强制 resize。但**真正的灵魂是 EgoScale 数据集 —— 20,854 小时人类第一人称视频**,含自我相机/腕部/手部追踪。**首次发现机器人灵巧度的 scaling law**:从 1k 小时到 20k 小时,**任务完成率 > 2× 提升**。

这意味着**未来 VLA 不再卡数据规模 —— 人类视频白嫖是新主流路径**,这是机器人圈的'GPT moment'。"

### 【4:30–5:00】个人判断 + 实习关联

"我实习项目主力用的是 **N1.5**,4 类箱体(4311/4322/4611/4633)混训 85% 成功率,Orin TensorRT 30Hz 部署。我们没用 π0.5 是因为 GR00T 推理快、Orin 部署成熟,符合工业要求。

我对 GR00T 系列的判断:**走大而全的工业路线**(规模碾压 + 部署优先 + NVIDIA 生态绑定),和 PI 的 π 系列'小而美研究路线'形成对比。**N1.7 的 EgoScale + scaling law 是把 GR00T 从'工业部署优先'明确转向'数据规模主导'**,这条路线我认为会持续 2-3 年主导整个 VLA 圈。

如果让我继续做,**最想试 N1.6/N1.7 的 Cosmos Reason 在长视地任务上的表现** —— Sys2 加 reasoning 后,'拿杯子 → 倒咖啡 → 端给客人'这种多步任务能力会有质变。"

---

## 七、资源链接

- **官方页面**:
  - [GR00T N1.5](https://research.nvidia.com/labs/gear/gr00t-n1_5/) / [GR00T N1.6](https://research.nvidia.com/labs/gear/gr00t-n1_6/)
  - [HuggingFace blog: GR00T N1.7](https://huggingface.co/blog/nvidia/gr00t-n1-7)
- **论文**:
  - GR00T N1:[arxiv 2503.14734](https://arxiv.org/abs/2503.14734)
- **HuggingFace checkpoints**(全部可下载):
  - [`nvidia/GR00T-N1.5-3B`](https://huggingface.co/nvidia/GR00T-N1.5-3B)
  - [`nvidia/GR00T-N1.6-3B`](https://huggingface.co/nvidia/GR00T-N1.6-3B)
  - **`nvidia/GR00T-N1.7`**(最新,2026.04)
- **GitHub**:
  - [github.com/NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)(主仓,已升级到 N1.7)
  - [github.com/NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)(全身控制)
  - [github.com/NVIDIA-Medtech/GR00T-H](https://github.com/NVIDIA-Medtech/GR00T-H)(医疗版,基于 N1.6)
- **NVIDIA 生态**:
  - [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
  - [Cosmos](https://github.com/nvidia-cosmos)
