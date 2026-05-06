# Study Notes —— VLA 模型深度学习笔记

> 自学笔记,用于面试准备 + 系统理解 VLA 主流模型。
> 维护者:郭子毅 / 2026.04 起

---

## 已完成笔记

| 文件 | 模型 | 字数 | 重点 |
|---|---|---|---|
| [pi05_深度笔记.md](./pi05_深度笔记.md) | **π0.5** + π*0.6 / π0.7 进化 | ~14K | Flow Matching / PaliGemma / RECAP |
| [groot_n1_5_深度笔记.md](./groot_n1_5_深度笔记.md) | **GR00T N1.5** + N1.6 / N1.7 进化 | ~14K | Eagle VLM / FLARE / DreamGen / Sys1-Sys2 |

---

## 9 节统一模板(每份笔记)

```
一、背景与定位        — 公司 / 团队 / 在 VLA 谱系里的位置
二、模型架构          — ASCII 架构图 / VLM / Action head / 输入输出
三、训练策略          — 预训练数据 / 后训练 / 跨机器人迁移
四、关键技术点(3-5)— 核心 trick 详解
五、代码实现要点      — 仓库结构 / 真实代码片段(5+)/ 微调示例
六、工程小巧思        — 数据预处理 / 训练 trick / 推理优化
七、社区讨论          — 强项 / 弱项 / 复现踩坑 / 同类对比
八、进化路线          —【已发布】/【社区推测】/【我的判断】三档
九、面试速答          — 30 秒 / 2 分钟 / 5 分钟版本
```

---

## 横向对比速查

| 维度 | π0.5(PI) | GR00T N1.5(NVIDIA) |
|---|---|---|
| VLM | PaliGemma-3B | Eagle-2.5(冻结) |
| Action head | Flow Matching(10 步 ODE) | Flow Matching transformer(4 步) |
| 推理速度 | ~50ms/chunk(H100) | 24.7ms/chunk(H100)、Orin 30Hz |
| 训练规模 | 未公开,估计百万 episode | 1000 H100 × 250K step |
| 真机泛化 | ★★★★★ | ★★★ |
| 真机部署 | ★★★ | ★★★★★ |
| 开源完整 | ★★★★(JAX+PyTorch) | ★★★(Eagle 不全开源) |
| 进化方向 | π*0.6(RECAP RL)/ π0.7(组合泛化) | N1.6(Cosmos Reason / 全身)/ N1.7(Qwen3-VL) |

---

## 计划扩展(待写)

- [ ] OpenVLA-OFT 深度笔记(离散 action token 路线代表)
- [ ] RDT-1B 深度笔记(Diffusion 路线代表)
- [ ] Diffusion Policy / DP3 深度笔记(2D vs 3D)
- [ ] Lingbot-VLA 深度笔记(对应 UMI 实验里的基线模型)
- [ ] ACT 深度笔记(Aloha 论文,搬箱主力)

---

## 学习资源汇总

### 论文
- π0:[arxiv 2410.24164](https://arxiv.org/abs/2410.24164)
- π0.5:[arxiv 2504.16054](https://arxiv.org/abs/2504.16054)
- π*0.6:[arxiv 2511.14759](https://arxiv.org/abs/2511.14759)
- GR00T N1:[arxiv 2503.14734](https://arxiv.org/abs/2503.14734)

### 仓库
- openpi:[github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)(本地:`/home/leju/selbst/openpi/`)
- Isaac-GR00T:[github.com/NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)

### 综述
- Awesome Embodied VLA:[github.com/jonyzhang2023/awesome-embodied-vla-va-vln](https://github.com/jonyzhang2023/awesome-embodied-vla-va-vln)
