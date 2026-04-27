#!/usr/bin/env python3
"""
Bag to LeRobot Dataset Converter (DELTA ACTION VERSION)

This script converts rosbag data to LeRobot dataset format with DELTA (relative) actions.

Key Features:
- Time-aligned data extraction from rosbag
- Forward fill for missing data points
- **COM actions converted to delta (relative) format:**
  - Position (x, y, z): delta relative to previous frame
  - Rotation (6D representation): delta relative to previous frame
- **Arm actions remain absolute (in radians):** 14 dimensions (left 7 + right 7)
- **Gait mode remains absolute:** 1=walk, 2=stance
- LeRobot dataset conversion
- Support for multiple bag files

Action Format:
- COM delta (9D): [dx, dy, dz, dR11, dR21, dR31, dR12, dR22, dR32]
- Arm absolute (14D): [left_arm_joints(7), right_arm_joints(7)] in radians
- Gait absolute (1D): [gait_mode] where 1=walk, 2=stance

Usage:
    python comControl_bag2lerobot_delta.py --bag_path <path> --target-dir <output>
"""

import rosbag
import numpy as np
import sys
import os
import shutil
import dataclasses
import glob
from pathlib import Path
from typing import Literal, List
from collections import defaultdict

# LeRobot imports
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from my_dataset import MyLeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    print("Warning: LeRobot not available. Dataset conversion functionality will be disabled.")
    LEROBOT_AVAILABLE = False
    LeRobotDataset = None
    MyLeRobotDataset = None

import torch
import tqdm
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
import torch.nn.functional as F

# Import config
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_com import topic_info, action_names, states_names
# LeRobot dataset configuration
USE_FOUR_CAMERA_OBS = True
USE_WBC_OBS = False

# Image feature extraction configuration
USE_IMAGE_FEATURES = True  # Set to True to extract image features instead of storing raw images

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None

@dataclasses.dataclass(frozen=True)
class SmoothingConfig:
    """平滑化配置"""
    enable_smoothing: bool = True
    smoothing_method: str = "savgol"  # "savgol", "moving_avg", "spline", "none"
    window_length: int = 11  # 窗口长度（必须是奇数）- 增加到11获得更强平滑
    polyorder: int = 3  # Savitzky-Golay多项式阶数 - 增加到3获得更平滑曲线
    moving_avg_window: int = 5  # 移动平均窗口大小 - 增加窗口大小
    spline_smoothing_factor: float = 0.5  # 样条平滑因子 - 增加平滑强度

DEFAULT_DATASET_CONFIG = DatasetConfig()
DEFAULT_SMOOTHING_CONFIG = SmoothingConfig()


class ImageFeatureExtractor:
    """图像特征提取器，用于将图像转换为ResNet特征"""
    
    def __init__(self, device='cuda:0'):
        self.device = device
        backbone_model = getattr(torchvision.models, 'resnet18')(
            replace_stride_with_dilation=[False, False, False],
            weights='ResNet18_Weights.IMAGENET1K_V1',
            norm_layer=FrozenBatchNorm2d,
        )
        # 使用IntermediateLayerGetter获取layer4的特征图
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
        self.backbone.eval()
        self.backbone.to(device)

        # 获取ImageNet预处理的均值和标准差
        self.backbone_weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        self.preprocess = self.backbone_weights.transforms()
        self.mean = self.preprocess.mean
        self.std = self.preprocess.std

        self.mean = torch.tensor(self.mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def normalize_img(self, img_chw):
        """归一化图像"""
        if isinstance(img_chw, torch.Tensor):
            img_chw = img_chw.detach().clone().to(self.device)
        else:
            img_chw = torch.tensor(img_chw, dtype=torch.float32).to(self.device)
        if img_chw.ndim == 3:   # (3,H,W) -> (1,3,H,W)
            img_chw = img_chw.unsqueeze(0)
        img_chw = (img_chw - self.mean) / (self.std + 1e-8)
        return img_chw

    def get_img_embed(self, img) -> torch.Tensor:
        """提取图像特征"""
        # 检查输入是否为空或无效
        if img is None:
            raise ValueError("Image is None")
        
        # 确保输入是numpy数组
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        
        # 检查数组是否为空或0维
        if img.size == 0:
            raise ValueError("Image array is empty")
        
        if img.ndim == 0:
            raise ValueError(f"Expected 3D image, got 0D scalar: {img}")
        
        # 确保输入是torch tensor
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.copy()).float()  # 使用copy()避免只读警告
        
        # 检查图像格式并转换为CHW格式
        if img.ndim == 3:
            if img.shape[2] == 3:  # HWC格式 -> CHW格式
                img = img.permute(2, 0, 1)
            elif img.shape[0] == 3:  # 已经是CHW格式
                pass
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")
        else:
            raise ValueError(f"Expected 3D image, got {img.ndim}D")
        
        H_t, W_t = (240, 320)  # 目标尺寸
        img = F.interpolate(
            img.unsqueeze(dim=0), size=(H_t, W_t),
            mode="bilinear", align_corners=False, antialias=True
        )

        img = self.normalize_img(img.squeeze())
        
        with torch.no_grad():
            embed = self.backbone(img)["feature_map"]
        
        return embed


def to_bytes_hex(arr: np.ndarray) -> bytes:
    """将numpy数组转换为十六进制字节"""
    raw = arr.astype(np.float32, copy=False).tobytes(order="C")
    return raw.hex().upper().encode("ascii")


def euler_to_rotation_matrix_first_two_cols(roll, pitch, yaw):
    """
    将欧拉角(roll, pitch, yaw)转换为旋转矩阵的前两列
    
    Args:
        roll: 绕x轴旋转角度(弧度)
        pitch: 绕y轴旋转角度(弧度) 
        yaw: 绕z轴旋转角度(弧度)
    
    Returns:
        6D向量，包含旋转矩阵前两列的6个元素
    """
    # 使用scipy创建旋转矩阵
    r = Rotation.from_euler('xyz', [roll, pitch, yaw])
    rotation_matrix = r.as_matrix()
    
    # 取前两列并展平为6D向量
    # 使用'F' (Fortran order，按列展平) 以保持标准6D旋转表示格式
    #        | R11  R12  R13 |
    #    R = | R21  R22  R23 |
    #        | R31  R32  R33 |
    # [R11, R21, R31, R12, R22, R32] 而不是 [R11, R12, R21, R22, R31, R32]
    first_two_cols = rotation_matrix[:, :2].flatten('F')
    return first_two_cols


class ActionChangeDetector:
    """检测动作变化的工具类"""
    def __init__(self, threshold=0.1, window_size=5):
        self.threshold = threshold
        self.window_size = window_size
        self.action_history = []

    def add_action(self, action):
        """添加新的动作并检测变化"""
        self.action_history.append(action)
        if len(self.action_history) > self.window_size:
            self.action_history.pop(0)
        
        if len(self.action_history) >= 2:
            # 计算动作变化
            action_change = np.linalg.norm(
                np.array(self.action_history[-1]) - np.array(self.action_history[-2])
            )
            return action_change > self.threshold
        return False


class DataConverter:
    """数据转换和对齐类"""
    def __init__(self, use_cmd_pose: bool = False, enable_dynamic_cropping: bool = False, 
                 change_threshold: float = 0.1, change_window_size: int = 5, 
                 use_image_features: bool = False, device: str = 'cuda:0'):
        self.use_cmd_pose = use_cmd_pose
        self.enable_dynamic_cropping = enable_dynamic_cropping
        self.use_image_features = use_image_features
        self.topic_data = defaultdict(dict)
        
        # 初始化图像特征提取器
        if self.use_image_features:
            self.image_extractor = ImageFeatureExtractor(device=device)
            print(f"Image feature extraction enabled with device: {device}")
        else:
            self.image_extractor = None
        
        # 初始化变化检测器
        if self.enable_dynamic_cropping:
            self.change_detector = ActionChangeDetector(threshold=change_threshold, window_size=change_window_size)
        else:
            self.change_detector = None

    def copy_data_and_set_zero(self, data, ts):
        """
        把数据拷贝并且设置为0
        Args:
            data: 原始数据
            ts: 时间戳

        Returns:
            零数据
        """
        # 确保data是numpy数组，然后拷贝数据并设置为0
        if isinstance(data, (list, tuple)):
            data = np.array(data, dtype=np.float32)
        return np.zeros_like(data)

    def forward_fill_on_grid(self, msg_dict, time_grid, shape=None):
        """
        对一个 topic 的消息进行前向填充，按 time_grid 对齐

        参数：
        - msg_dict: 包含 'ts' 和 'data' 的字典
        - time_grid: 均匀的时间戳序列，例如 np.arange(...)
        - shape: 数据形状（用于零数据初始化）

        返回：
        - 一个列表，与 time_grid 等长，每个位置是最近的消息（或真实初始值）
        
        新的填充策略：
        - 第一帧：如果没有消息到达，使用第一个有效消息的真实值（避免delta计算时的大跳变）
        - 后续帧：如果没有消息到达，填充为零（避免使用过时的消息）
        - 有消息后：正常向前填充
        """
        msg_list = sorted(zip(msg_dict['ts'], msg_dict['data']), key=lambda x: x[0])  # 确保按时间升序排列
        result_list = []
        idx = 0

        # 如果没有任何消息，返回全零数组
        if not msg_list:
            print(f"⚠️  WARNING: No messages found for this topic! Filling with zeros.")
            return [np.zeros(shape, dtype=np.float32) for _ in time_grid]

        current_msg = None  # 当前可用的最近消息（用于 forward fill）
        is_first_frame = True  # 标记是否是第一帧

        for t in time_grid:
            # 更新所有时间小于等于当前帧的消息
            while idx < len(msg_list) and msg_list[idx][0] <= t:
                current_msg = msg_list[idx][1]
                idx += 1
            
            # 如果还没有历史消息（当前时间点早于第一个消息）
            if current_msg is None:
                if is_first_frame:
                    # 第一帧：使用第一个有效消息的真实值（避免delta计算时的大跳变）
                    result_list.append(msg_list[0][1])
                else:
                    # 后续帧：填充为零（避免使用未来的消息）
                    if shape is None:
                        result_list.append(np.array(0.0, dtype=np.float32))
                    else:
                        result_list.append(np.zeros(shape, dtype=np.float32))
            else:
                # 有历史消息：正常向前填充
                result_list.append(current_msg)
            
            is_first_frame = False

        return result_list

    def process_rosbag(self, bag_path, dt=0.1, smoothing_config=DEFAULT_SMOOTHING_CONFIG):
        """
        从 bag 文件中读取多个 topic，并以固定时间间隔（如 0.1s）对齐它们的消息。
        缺失时使用前向填充；如果没有历史值则为零数据。

        参数：
        - bag_path: bag 文件路径
        - dt: 时间间隔（秒），例如 0.1 表示每 100ms 一帧

        返回：
        - aligned_dict: dict，key 是 topic，value 是每帧对应的消息（与 time_grid 对齐）
        - time_grid: 均匀的时间戳列表（np.array）
        """
        bag = rosbag.Bag(bag_path)
        topic_data = defaultdict(dict)

        # Step 1: 读取所有感兴趣 topic 的消息，同时统计最早和最晚时间戳
        min_time = float('inf')
        max_time = float('-inf')
        names = []

        # 使用config_com.py中的topic_info，并添加额外的topic
        extended_topic_info = topic_info.copy()
        
        # 添加MPC和步态相关的topic
        extended_topic_info.update({
            'mpc_target_state': {
                "topic": "/humanoid/mpc/targetState",
                "msg_process_fn": self.process_mpc_target_state,
                "shape": 9  # 9维数据 [x, y, z, R11, R21, R31, R12, R22, R32]
            },
            'arm_traj': {
                "topic": "/mm_kuavo_arm_traj", 
                "msg_process_fn": self.process_arm_traj,
                "shape": 14  # 14个关节位置
            },
            'gait_time_name': {
                "topic": "/humanoid_mpc_gait_time_name",
                "msg_process_fn": self.process_gait_time_name,
                "shape": 1  # [gait_name_encoded]
            }
        })

        for name, info in extended_topic_info.items():
            topic = info["topic"]
            msg_process_fn = info["msg_process_fn"]
            shape = info["shape"]
            names.append(name)
            topic_data[name] = defaultdict(list)

            for topic, msg, t in bag.read_messages(topics=[topic]):
                ts = t.to_sec()
                min_time = min(min_time, ts)
                max_time = max(max_time, ts)
                # 处理消息
                msg_process_fn(msg, topic_data, name, ts)

        bag.close()

        # Step 2: 构造统一的时间轴（均匀采样时间点）
        time_grid = np.arange(min_time, max_time + dt, dt)

        # Step 3: 对每个 topic 做前向填充
        aligned_dict = {}
        for name in names:
            # 有些 topic 可能一个消息都没有，默认 []
            aligned_dict[name] = self.forward_fill_on_grid(
                topic_data.get(name, {'ts': [], 'data': []}), 
                time_grid, 
                extended_topic_info[name]['shape']
            )

        # Step 4: 将MPC COM动作转换为delta动作（相对于前一帧）
        if 'mpc_target_state' in aligned_dict and len(aligned_dict['mpc_target_state']) > 0:
            print("🔄 Converting MPC COM actions to delta (relative) actions...")
            aligned_dict['mpc_target_state'] = self.convert_com_to_delta_actions(
                aligned_dict['mpc_target_state'], dt=dt
            )
            print("✅ COM actions converted to delta format")
            
            # Step 5: 应用平滑化处理（如果启用）
            if smoothing_config.enable_smoothing:
                aligned_dict['mpc_target_state'] = self.smooth_delta_actions(
                    aligned_dict['mpc_target_state'], smoothing_config
                )

        return aligned_dict, time_grid

    def create_empty_dataset(
            self,
            repo_id: str,
            robot_type: str,
            mode: Literal["video", "image"] = "video",
            *,
            has_velocity: bool = False,
            has_effort: bool = False,
            dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
            root: str,
            use_image_features: bool = False,
            dt: float = 0.1,
    ):
        """创建空的LeRobot数据集"""
        if not LEROBOT_AVAILABLE:
            raise ImportError("LeRobot is not available. Please install LeRobot to use dataset conversion functionality.")
        
        features = {}
        
        # 根据是否使用图像特征来决定数据类型和形状
        if use_image_features:
            # 使用图像特征模式 - 与preprocess_dataset.py保持一致
            features["observation.embeds.image"] = {
                "dtype": "binary",
                "shape": (1,),
                "names": ["whatever"],
                "type": "visual"  # 明确指定为视觉特征
            }
        else:
            # 使用原始图像模式
            features["observation.image"] = {
                "dtype": mode,
                "shape": (3, 480, 640),
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            }
        
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (len(states_names),),  # 38维: 14 arm + 3 lin_acc + 3 ang_vel + 9 com_6d + 9 prev_delta
            "names": {
                "motors": states_names
            }
        }
        
        features["action"] = {
            "dtype": "float32",
            "shape": (24,),  # 9 (mpc_pose_data: 3 pos + 6 rot_matrix_cols) + 14 (arm_data) + 1 (gait_data)
            "names": {
                "motors": ["mpc_pose_x", "mpc_pose_y", "mpc_pose_z", "mpc_rot_11", "mpc_rot_21", "mpc_rot_31", "mpc_rot_12", "mpc_rot_22", "mpc_rot_32"] + 
                         ["arm_joint_" + str(i) for i in range(14)] + 
                         ["gait_name_encoded"]
            }
        }
        
        features["robot_com_state_env.info"] = {
            "dtype": "float32",
            "shape": (9,),  # 9D com state data [x, y, z, R11, R21, R31, R12, R22, R32]
            "names": {
                "motors": ["com_x", "com_y", "com_z", "com_rot_11", "com_rot_21", "com_rot_31", "com_rot_12", "com_rot_22", "com_rot_32"]
            }
        }
        
        # 只有在非Isaac Sim模式下才添加new_image特征
        if USE_FOUR_CAMERA_OBS:
            if use_image_features:
                # 使用图像特征模式 - 与preprocess_dataset.py保持一致
                features["observation.embeds.chest_image"] = {
                    "dtype": "binary",
                    "shape": (1,),
                    "names": ["whatever"],
                    "type": "visual"  # 明确指定为视觉特征
                }
                features["observation.embeds.left_shoulder_image"] = {
                    "dtype": "binary",
                    "shape": (1,),
                    "names": ["whatever"],
                    "type": "visual"  # 明确指定为视觉特征
                }
                features["observation.embeds.right_shoulder_image"] = {
                    "dtype": "binary",
                    "shape": (1,),
                    "names": ["whatever"],
                    "type": "visual"  # 明确指定为视觉特征
                }
            else:
                # 使用原始图像模式
                features["observation.chest_image"] = {
                    "dtype": mode,
                    "shape": (3, 480, 640),
                    "names": [
                        "channels",
                        "height",
                        "width",
                    ],
                }
                features["observation.left_shoulder_image"] = {
                    "dtype": mode,
                    "shape": (3, 480, 640),
                    "names": [
                        "channels",
                        "height",
                        "width",
                    ],
                }
                features["observation.right_shoulder_image"] = {
                    "dtype": mode,
                    "shape": (3, 480, 640),
                    "names": [
                        "channels",
                        "height",
                        "width",
                    ],
                }

        # 确保root目录不存在，如果存在则删除
        root_path = Path(root)
        if root_path.exists():
            print(f"Root directory {root_path} already exists. Removing it...")
            shutil.rmtree(root_path)
            import time
            time.sleep(0.1)
        
        return MyLeRobotDataset.create(
            repo_id=repo_id,
            fps=int(1.0 / dt),  # 根据dt计算fps
            features=features,
            use_videos=True,  # 与preprocess_dataset.py保持一致，使用图像特征时不需要视频
            image_writer_processes=dataset_config.image_writer_processes,
            image_writer_threads=dataset_config.image_writer_threads,
            root=root,
        )

    def convert_aligned_data_to_lerobot(
            self,
            aligned_data: dict,
            time_grid: np.ndarray,
            target_dir: Path,
            repo_id: str = "kuavo_com_control",
            task: str = "com_control",
            mode: Literal["video", "image"] = "video",
            dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
            overwrite: bool = False,
    ):
        """
        将对齐后的数据转换为LeRobot数据集
        
        Args:
            aligned_data: 对齐后的数据字典
            time_grid: 时间网格
            target_dir: 目标目录
            repo_id: 数据集ID
            task: 任务名称
            mode: 数据模式
            dataset_config: 数据集配置
            overwrite: 是否覆盖现有目录
            
        Returns:
            创建的数据集
        """
        # 如果目标目录已存在，根据overwrite参数决定是否删除
        if target_dir.exists():
            if overwrite:
                print(f"Target directory {target_dir} already exists. Removing it...")
                shutil.rmtree(target_dir)
                # 确保目录完全删除
                import time
                time.sleep(0.1)
            else:
                raise FileExistsError(f"Target directory {target_dir} already exists. Use --overwrite to overwrite it.")
        
        # 确保父目录存在
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建空的LeRobot数据集
        dataset = self.create_empty_dataset(
            repo_id=repo_id,
            robot_type="kuavo45",
            mode=mode,
            has_effort=False,
            has_velocity=False,
            dataset_config=dataset_config,
            root=str(target_dir),
            use_image_features=self.use_image_features,
            dt=0.1,  # 默认dt，实际应该从调用处传入
        )

        # 获取图像数据（如果存在）
        image_data = aligned_data.get('image', [])
        chest_image_data = aligned_data.get('chest_image', [])
        left_shoulder_image_data = aligned_data.get('left_shoulder_image', [])
        right_shoulder_image_data = aligned_data.get('right_shoulder_image', [])
        
        # 获取状态数据
        dof_state_data = aligned_data.get('dof_state', [])
        lin_acc_data = aligned_data.get('lin_acc', [])
        ang_vel_data = aligned_data.get('ang_vel', [])
        humanoid_wbc_observation_data = aligned_data.get('humanoid_wbc_observation', [])
        robot_com_state_data = aligned_data.get('robot_com_state', [])

        # 获取动作数据 - 使用mpc_data, arm_data, gait_data作为action
        mpc_action_data = aligned_data.get('mpc_target_state', [])
        arm_action_data = aligned_data.get('arm_traj', [])
        gait_action_data = aligned_data.get('gait_time_name', [])
        
        num_frames = len(time_grid)
        print(f"Converting {num_frames} frames to LeRobot dataset...")
        
        # 处理每一帧数据
        for i in tqdm.tqdm(range(num_frames), desc="Converting frames"):
            frame = {}
            
            # 处理图像数据
            if i < len(image_data) and image_data[i] is not None:
                try:
                    if self.use_image_features and self.image_extractor is not None:
                        # 提取图像特征
                        embed = self.image_extractor.get_img_embed(image_data[i]).detach().cpu().numpy().astype("float32")
                        frame["observation.embeds.image"] = to_bytes_hex(embed)
                    else:
                        # 使用原始图像（复制数组以避免只读警告）
                        frame["observation.image"] = torch.from_numpy(image_data[i].copy())
                except Exception as e:
                    print(f"Warning: Failed to process image at frame {i}: {e}")
                    # 跳过这个图像，不添加到frame中
            
            if USE_FOUR_CAMERA_OBS:
                if i < len(chest_image_data) and chest_image_data[i] is not None:
                    try:
                        if self.use_image_features and self.image_extractor is not None:
                            # 提取图像特征
                            embed = self.image_extractor.get_img_embed(chest_image_data[i]).detach().cpu().numpy().astype("float32")
                            frame["observation.embeds.chest_image"] = to_bytes_hex(embed)
                        else:
                            # 使用原始图像（复制数组以避免只读警告）
                            frame["observation.chest_image"] = torch.from_numpy(chest_image_data[i].copy())
                    except Exception as e:
                        print(f"Warning: Failed to process chest image at frame {i}: {e}")
                        
                if i < len(left_shoulder_image_data) and left_shoulder_image_data[i] is not None:
                    try:
                        if self.use_image_features and self.image_extractor is not None:
                            # 提取图像特征
                            embed = self.image_extractor.get_img_embed(left_shoulder_image_data[i]).detach().cpu().numpy().astype("float32")
                            frame["observation.embeds.left_shoulder_image"] = to_bytes_hex(embed)
                        else:
                            # 使用原始图像（复制数组以避免只读警告）
                            frame["observation.left_shoulder_image"] = torch.from_numpy(left_shoulder_image_data[i].copy())
                    except Exception as e:
                        print(f"Warning: Failed to process left shoulder image at frame {i}: {e}")
                        
                if i < len(right_shoulder_image_data) and right_shoulder_image_data[i] is not None:
                    try:
                        if self.use_image_features and self.image_extractor is not None:
                            # 提取图像特征
                            embed = self.image_extractor.get_img_embed(right_shoulder_image_data[i]).detach().cpu().numpy().astype("float32")
                            frame["observation.embeds.right_shoulder_image"] = to_bytes_hex(embed)
                        else:
                            # 使用原始图像（复制数组以避免只读警告）
                            frame["observation.right_shoulder_image"] = torch.from_numpy(right_shoulder_image_data[i].copy())
                    except Exception as e:
                        print(f"Warning: Failed to process right shoulder image at frame {i}: {e}")
            
            # ⭐ 获取previous delta action（COM部分的前9维）
            if i == 0:
                # 第一帧：previous delta为零
                previous_delta_com = np.zeros(9, dtype=np.float32)
            else:
                # 其他帧：使用上一帧的delta COM action（前9维）
                if i-1 < len(mpc_action_data) and mpc_action_data[i-1] is not None:
                    previous_delta_com = np.array(mpc_action_data[i-1], dtype=np.float32)
                else:
                    previous_delta_com = np.zeros(9, dtype=np.float32)
            
            # 处理状态数据
            # 新的观测空间: [14 arm + 3 lin_acc + 3 ang_vel + 9 com_6d + 9 prev_delta] = 38维
            state_parts = []
            if i < len(dof_state_data) and dof_state_data[i] is not None:
                state_parts.append(np.array(dof_state_data[i], dtype=np.float32))  # 14维
            if i < len(lin_acc_data) and lin_acc_data[i] is not None:
                state_parts.append(np.array(lin_acc_data[i], dtype=np.float32))  # 3维
            if i < len(ang_vel_data) and ang_vel_data[i] is not None:
                state_parts.append(np.array(ang_vel_data[i], dtype=np.float32))  # 3维
            if i < len(humanoid_wbc_observation_data) and humanoid_wbc_observation_data[i] is not None:
                state_parts.append(np.array(humanoid_wbc_observation_data[i], dtype=np.float32))  # 9维 [x,y,z,R11,R21,R31,R12,R22,R32]
            
            # ⭐ 添加previous delta action到观测中
            state_parts.append(previous_delta_com)  # 9维

            if state_parts:
                state = np.concatenate(state_parts)
                frame["observation.state"] = torch.from_numpy(state).type(torch.float32)
            
            # 处理环境信息
            if i < len(robot_com_state_data) and robot_com_state_data[i] is not None:
                frame["robot_com_state_env.info"] = torch.from_numpy(np.array(robot_com_state_data[i], dtype=np.float32)).type(torch.float32)

            # 处理动作数据 - 使用mpc_pose_data (9维), arm_data, gait_data
            action_parts = []
            if i < len(mpc_action_data) and mpc_action_data[i] is not None:
                # mpc_action_data现在是9维数据(3位置+6旋转矩阵列)
                mpc_pose_data = mpc_action_data[i]
                if len(mpc_pose_data) == 9:
                    action_parts.append(np.array(mpc_pose_data, dtype=np.float32))
                else:
                    print(f"Warning: Expected 9D pose data, got {len(mpc_pose_data)} dimensions")
                    action_parts.append(np.zeros(9, dtype=np.float32))
            if i < len(arm_action_data) and arm_action_data[i] is not None:
                action_parts.append(np.array(arm_action_data[i], dtype=np.float32))
            if i < len(gait_action_data) and gait_action_data[i] is not None:
                action_parts.append(np.array(gait_action_data[i], dtype=np.float32))
            
            if action_parts:
                action = np.concatenate(action_parts)
                frame["action"] = torch.from_numpy(action).type(torch.float32)
            
            # 添加帧到数据集
            if frame:  # 只有当帧不为空时才添加
                dataset.add_frame(frame, task)
        
        # 保存episode
        dataset.save_episode()
        print(f"LeRobot dataset saved to {target_dir}")
        
        return dataset

    def collect_bag_files(self, source_path: str) -> List[Path]:
        """
        收集指定路径下的所有bag文件
        
        Args:
            source_path: 源路径，可以是文件或目录
            
        Returns:
            bag文件路径列表
        """
        source_path = Path(source_path)
        bag_files = []
        
        if source_path.is_file():
            # 如果是单个文件，检查是否为bag文件
            if source_path.suffix == '.bag':
                bag_files.append(source_path)
            else:
                print(f"Error: File {source_path} is not a .bag file")
        elif source_path.is_dir():
            # 如果是目录，递归查找所有bag文件
            bag_files = list(source_path.rglob('*.bag'))
            print(f"Found {len(bag_files)} bag files in directory {source_path}")
        else:
            print(f"Error: Path {source_path} does not exist")
            
        return bag_files

    def convert_multiple_bags_to_lerobot(
            self,
            bag_files: List[Path],
            target_dir: Path,
            repo_id: str = "kuavo_com_control",
            task: str = "com_control",
            mode: Literal["video", "image"] = "video",
            dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
            smoothing_config: SmoothingConfig = DEFAULT_SMOOTHING_CONFIG,
            overwrite: bool = False,
            dt: float = 0.1,
    ):
        """
        将多个bag文件转换为LeRobot数据集
        
        Args:
            bag_files: bag文件路径列表
            target_dir: 目标目录
            repo_id: 数据集ID
            task: 任务名称
            mode: 数据模式
            dataset_config: 数据集配置
            overwrite: 是否覆盖现有目录
            
        Returns:
            创建的数据集
        """
        if not bag_files:
            print("Error: No bag files provided")
            return None
            
        # 如果目标目录已存在，根据overwrite参数决定是否删除
        if target_dir.exists():
            if overwrite:
                print(f"Target directory {target_dir} already exists. Removing it...")
                shutil.rmtree(target_dir)
                import time
                time.sleep(0.1)
            else:
                raise FileExistsError(f"Target directory {target_dir} already exists. Use --overwrite to overwrite it.")
        
        # 确保父目录存在
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建空的LeRobot数据集
        dataset = self.create_empty_dataset(
            repo_id=repo_id,
            robot_type="kuavo45",
            mode=mode,
            has_effort=False,
            has_velocity=False,
            dataset_config=dataset_config,
            root=str(target_dir),
            use_image_features=self.use_image_features,
            dt=0.1,  # 默认dt，实际应该从调用处传入
        )
        
        print(f"Processing {len(bag_files)} bag files...")
        
        # 处理每个bag文件
        for i, bag_file in enumerate(tqdm.tqdm(bag_files, desc="Processing bag files")):
            print(f"Processing bag file {i+1}/{len(bag_files)}: {bag_file}")
            
            try:
                # 处理单个bag文件
                aligned_data, time_grid = self.process_rosbag(str(bag_file), dt=dt, smoothing_config=smoothing_config)
                
                # 获取图像数据（如果存在）
                image_data = aligned_data.get('image', [])
                chest_image_data = aligned_data.get('chest_image', [])
                left_shoulder_image_data = aligned_data.get('left_shoulder_image', [])
                right_shoulder_image_data = aligned_data.get('right_shoulder_image', [])
                
                # 获取状态数据
                dof_state_data = aligned_data.get('dof_state', [])
                lin_acc_data = aligned_data.get('lin_acc', [])
                ang_vel_data = aligned_data.get('ang_vel', [])
                humanoid_wbc_observation_data = aligned_data.get('humanoid_wbc_observation', [])
                robot_com_state_data = aligned_data.get('robot_com_state', [])

                # 获取动作数据 - 使用mpc_data, arm_data, gait_data作为action
                mpc_action_data = aligned_data.get('mpc_target_state', [])
                arm_action_data = aligned_data.get('arm_traj', [])
                gait_action_data = aligned_data.get('gait_time_name', [])
                
                num_frames = len(time_grid)
                print(f"Converting {num_frames} frames from {bag_file.name}...")
                
                # 处理每一帧数据
                for j in range(num_frames):
                    frame = {}
                    
                    # 处理图像数据
                    if j < len(image_data) and image_data[j] is not None:
                        try:
                            if self.use_image_features and self.image_extractor is not None:
                                # 提取图像特征
                                embed = self.image_extractor.get_img_embed(image_data[j]).detach().cpu().numpy().astype("float32")
                                frame["observation.embeds.image"] = to_bytes_hex(embed)
                            else:
                                # 使用原始图像（复制数组以避免只读警告）
                                frame["observation.image"] = torch.from_numpy(image_data[j].copy())
                        except Exception as e:
                            print(f"Warning: Failed to process image at frame {j} in {bag_file.name}: {e}")
                            # 跳过这个图像，不添加到frame中
                    
                    if USE_FOUR_CAMERA_OBS:
                        if j < len(chest_image_data) and chest_image_data[j] is not None:
                            try:
                                if self.use_image_features and self.image_extractor is not None:
                                    # 提取图像特征
                                    embed = self.image_extractor.get_img_embed(chest_image_data[j]).detach().cpu().numpy().astype("float32")
                                    frame["observation.embeds.chest_image"] = to_bytes_hex(embed)
                                else:
                                    # 使用原始图像（复制数组以避免只读警告）
                                    frame["observation.chest_image"] = torch.from_numpy(chest_image_data[j].copy())
                            except Exception as e:
                                print(f"Warning: Failed to process chest image at frame {j} in {bag_file.name}: {e}")
                        if j < len(left_shoulder_image_data) and left_shoulder_image_data[j] is not None:
                            try:
                                if self.use_image_features and self.image_extractor is not None:
                                    # 提取图像特征
                                    embed = self.image_extractor.get_img_embed(left_shoulder_image_data[j]).detach().cpu().numpy().astype("float32")
                                    frame["observation.embeds.left_shoulder_image"] = to_bytes_hex(embed)
                                else:
                                    # 使用原始图像（复制数组以避免只读警告）
                                    frame["observation.left_shoulder_image"] = torch.from_numpy(left_shoulder_image_data[j].copy())
                            except Exception as e:
                                print(f"Warning: Failed to process left shoulder image at frame {j} in {bag_file.name}: {e}")
                        if j < len(right_shoulder_image_data) and right_shoulder_image_data[j] is not None:
                            try:
                                if self.use_image_features and self.image_extractor is not None:
                                    # 提取图像特征
                                    embed = self.image_extractor.get_img_embed(right_shoulder_image_data[j]).detach().cpu().numpy().astype("float32")
                                    frame["observation.embeds.right_shoulder_image"] = to_bytes_hex(embed)
                                else:
                                    # 使用原始图像（复制数组以避免只读警告）
                                    frame["observation.right_shoulder_image"] = torch.from_numpy(right_shoulder_image_data[j].copy())
                            except Exception as e:
                                print(f"Warning: Failed to process right shoulder image at frame {j} in {bag_file.name}: {e}")
                    
                    # ⭐ 获取previous delta action（COM部分的前9维）
                    if j == 0:
                        # 第一帧：previous delta为零
                        previous_delta_com = np.zeros(9, dtype=np.float32)
                    else:
                        # 其他帧：使用上一帧的delta COM action（前9维）
                        if j-1 < len(mpc_action_data) and mpc_action_data[j-1] is not None:
                            previous_delta_com = np.array(mpc_action_data[j-1], dtype=np.float32)
                        else:
                            previous_delta_com = np.zeros(9, dtype=np.float32)
                    
                    # 处理状态数据
                    # 新的观测空间: [14 arm + 3 lin_acc + 3 ang_vel + 9 com_6d + 9 prev_delta] = 38维
                    state_parts = []
                    if j < len(dof_state_data) and dof_state_data[j] is not None:
                        state_parts.append(np.array(dof_state_data[j], dtype=np.float32))  # 14维
                    if j < len(lin_acc_data) and lin_acc_data[j] is not None:
                        state_parts.append(np.array(lin_acc_data[j], dtype=np.float32))  # 3维
                    if j < len(ang_vel_data) and ang_vel_data[j] is not None:
                        state_parts.append(np.array(ang_vel_data[j], dtype=np.float32))  # 3维
                    if j < len(humanoid_wbc_observation_data) and humanoid_wbc_observation_data[j] is not None:
                        state_parts.append(np.array(humanoid_wbc_observation_data[j], dtype=np.float32))  # 9维 [x,y,z,R11,R21,R31,R12,R22,R32]
                    
                    # ⭐ 添加previous delta action到观测中
                    state_parts.append(previous_delta_com)  # 9维
                    
                    if state_parts:
                        state = np.concatenate(state_parts)
                        frame["observation.state"] = torch.from_numpy(state).type(torch.float32)
                    
                    # 处理观测质心状态（用于训练时计算相对动作，保持9D格式）
                    if j < len(robot_com_state_data) and robot_com_state_data[j] is not None:
                        frame["robot_com_state_env.info"] = torch.from_numpy(np.array(robot_com_state_data[j], dtype=np.float32)).type(torch.float32)
                        
                    # 处理动作数据 - 使用mpc_pose_data (9维), arm_data, gait_data
                    action_parts = []
                    if j < len(mpc_action_data) and mpc_action_data[j] is not None:
                        # mpc_action_data现在是9维数据(3位置+6旋转矩阵列)
                        mpc_pose_data = mpc_action_data[j]
                        if len(mpc_pose_data) == 9:
                            action_parts.append(np.array(mpc_pose_data, dtype=np.float32))
                        else:
                            print(f"Warning: Expected 9D pose data, got {len(mpc_pose_data)} dimensions")
                            action_parts.append(np.zeros(9, dtype=np.float32))
                    if j < len(arm_action_data) and arm_action_data[j] is not None:
                        action_parts.append(np.array(arm_action_data[j], dtype=np.float32))
                    if j < len(gait_action_data) and gait_action_data[j] is not None:
                        action_parts.append(np.array(gait_action_data[j], dtype=np.float32))
                    
                    if action_parts:
                        action = np.concatenate(action_parts)
                        frame["action"] = torch.from_numpy(action).type(torch.float32)
                    
                    # 添加帧到数据集
                    if frame:  # 只有当帧不为空时才添加
                        dataset.add_frame(frame, task)
                
                # 为每个bag文件保存一个独立的episode
                dataset.save_episode()
                print(f"Successfully processed {num_frames} frames from {bag_file.name} and saved as episode {i}")
                
            except Exception as e:
                print(f"Error processing bag file {bag_file}: {e}")
                continue
        
        print(f"LeRobot dataset saved to {target_dir} with {len(bag_files)} episodes")
        
        return dataset

    def process_mpc_target_state(self, msg, topic_data, name, ts):
        """处理MPC目标状态消息，提取质心位置和旋转矩阵前2列"""
        # msg是std_msgs/Float64MultiArray
        state_array = np.array(msg.data)
        # 提取质心相关的数据 [x, y, z, roll, pitch, yaw]
        if len(state_array) > 12:
            x, y, z, yaw, pitch, roll = state_array[6:12]
        else:
            # 如果数据不足，用零填充
            x, y, z, yaw, pitch, roll = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        # 将欧拉角转换为旋转矩阵前2列
        rotation_cols = euler_to_rotation_matrix_first_two_cols(roll, pitch, yaw)
        
        # 组合位置和旋转矩阵前2列: [x, y, z, R11, R21, R31, R12, R22, R32]
        pose_data = np.concatenate([[x, y, z], rotation_cols])
        
        topic_data[name]['ts'].append(ts)
        topic_data[name]['data'].append(pose_data)

    def process_arm_traj(self, msg, topic_data, name, ts):
        """处理手臂轨迹消息"""
        # msg是sensor_msgs/JointState
        joint_positions = np.deg2rad(msg.position)
        # joint_positions = np.array(msg.position)
        topic_data[name]['ts'].append(ts)
        topic_data[name]['data'].append(joint_positions)

    def process_gait_time_name(self, msg, topic_data, name, ts):
        """处理步态时间名称消息"""
        # msg是kuavo_msgs/gaitTimeName
        # 将步态名称编码为数字：walk=1, stance=2
        gait_name_encoded = 1 if msg.gait_name == "walk" else 2
        gait_data = np.array([gait_name_encoded])  # 只保存gait_name_encoded
        topic_data[name]['ts'].append(ts)
        topic_data[name]['data'].append(gait_data)

    def convert_com_to_delta_actions(self, com_actions_list, dt=0.1):
        """
        将COM绝对动作转换为delta（相对）动作
        
        COM动作格式: [x, y, z, R11, R21, R31, R12, R22, R32]
        - 位置 (x, y, z): 转换为相对于前一帧的位置增量（考虑时间间隔）
        - 旋转 (6D表示): 转换为相对于前一帧的旋转增量
        
        Args:
            com_actions_list: 绝对COM动作列表，每个元素是9维numpy数组
            dt: 时间间隔（秒），用于计算位置变化率
            
        Returns:
            delta_actions_list: delta COM动作列表
        """
        if len(com_actions_list) == 0:
            return com_actions_list
        
        delta_actions_list = []
        position_deltas = []
        rotation_deltas = []
        
        for i in range(len(com_actions_list)):
            if i == 0:
                # 第一帧：delta为零（与自身比较）
                position_delta = np.zeros(3, dtype=np.float32)
                # 零旋转的6D表示: 单位矩阵的前两列 [1,0,0,0,1,0]
                rotation_delta_6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
                delta_action = np.concatenate([position_delta, rotation_delta_6d])
            else:
                # 其他帧：计算相对于前一帧的delta
                prev_action = com_actions_list[i-1]
                current_action = com_actions_list[i]
                
                # 位置delta: 考虑时间间隔的变化率
                position_delta = (current_action[:3] - prev_action[:3]) / dt
                
                # 旋转delta: 使用6D旋转表示计算相对旋转
                # R_delta = R_current * R_prev^T
                prev_rotation_6d = prev_action[3:9]
                current_rotation_6d = current_action[3:9]
                
                # 重构旋转矩阵
                prev_rotation_matrix = self._reconstruct_rotation_matrix_6d(prev_rotation_6d)
                current_rotation_matrix = self._reconstruct_rotation_matrix_6d(current_rotation_6d)
                
                # 计算旋转增量
                rotation_delta_matrix = current_rotation_matrix @ prev_rotation_matrix.T
                
                # 转换回6D表示
                rotation_delta_6d = self._rotation_matrix_to_6d(rotation_delta_matrix)
                
                delta_action = np.concatenate([position_delta, rotation_delta_6d])
                
                # 收集统计信息（跳过第一帧的零delta）
                position_deltas.append(np.linalg.norm(position_delta))
                rotation_deltas.append(rotation_delta_6d)
            
            delta_actions_list.append(delta_action)
        
        # 打印统计信息
        if len(position_deltas) > 0:
            print(f"📊 Delta Action Statistics (dt={dt}s):")
            print(f"   Total frames: {len(com_actions_list)}")
            print(f"   Position delta (norm) - mean: {np.mean(position_deltas):.6f}, max: {np.max(position_deltas):.6f}, min: {np.min(position_deltas):.6f}")
            print(f"   First 5 position deltas (norm): {[f'{d:.6f}' for d in position_deltas[:5]]}")
            
            # 检查旋转delta的有效性（第一列和第二列的范数应该接近1）
            rotation_col1_norms = [np.linalg.norm(rd[:3]) for rd in rotation_deltas]
            rotation_col2_norms = [np.linalg.norm(rd[3:6]) for rd in rotation_deltas]
            print(f"   Rotation 6D col1 norm - mean: {np.mean(rotation_col1_norms):.6f} (should be ~1.0)")
            print(f"   Rotation 6D col2 norm - mean: {np.mean(rotation_col2_norms):.6f} (should be ~1.0)")
        
        return delta_actions_list
    
    def _reconstruct_rotation_matrix_6d(self, rotation_6d):
        """
        从6D表示重构3x3旋转矩阵
        使用Gram-Schmidt正交化确保是有效的旋转矩阵
        #        | R11  R12  R13 |
        #    R = | R21  R22  R23 |
        #        | R31  R32  R33 |
        Args:
            rotation_6d: 6维向量 [R11, R21, R31, R12, R22, R32]
            
        Returns:
            3x3旋转矩阵
        """
        # 提取前两列
        col1 = rotation_6d[:3]  # [R11, R21, R31]
        col2 = rotation_6d[3:6]  # [R12, R22, R32]
        
        # Gram-Schmidt正交化
        # 第一列归一化
        col1_norm = np.linalg.norm(col1)
        if col1_norm < 1e-8:
            col1_normalized = np.array([1.0, 0.0, 0.0])
        else:
            col1_normalized = col1 / col1_norm
        
        # 第二列正交化并归一化
        col2_projected = col2 - np.dot(col2, col1_normalized) * col1_normalized
        col2_norm = np.linalg.norm(col2_projected)
        if col2_norm < 1e-8:
            col2_normalized = np.array([0.0, 1.0, 0.0])
        else:
            col2_normalized = col2_projected / col2_norm
        
        # 第三列通过叉积得到
        col3_normalized = np.cross(col1_normalized, col2_normalized)
        
        # 组合成旋转矩阵
        rotation_matrix = np.column_stack([col1_normalized, col2_normalized, col3_normalized])
        
        return rotation_matrix
    
    def _rotation_matrix_to_6d(self, rotation_matrix):
        """
        将3x3旋转矩阵转换为6D表示
        
        Args:
            rotation_matrix: 3x3旋转矩阵
        #        | R11  R12  R13 |
        #    R = | R21  R22  R23 |
        #        | R31  R32  R33 |
        Returns:
            6维向量 [R11, R21, R31, R12, R22, R32]
        """
        # 提取前两列
        col1 = rotation_matrix[:, 0]  # [R11, R21, R31]
        col2 = rotation_matrix[:, 1]  # [R12, R22, R32]
        
        # 组合成6D表示
        rotation_6d = np.concatenate([col1, col2])
        
        return rotation_6d

    def smooth_delta_actions(self, delta_actions_list, smoothing_config=SmoothingConfig()):
        """
        对delta动作进行平滑化处理，保持积分面积不变
        
        Args:
            delta_actions_list: delta动作列表，每个元素是9维numpy数组 [dx, dy, dz, dR11, dR21, dR31, dR12, dR22, dR32]
            smoothing_config: 平滑化配置
            
        Returns:
            smoothed_actions_list: 平滑化后的delta动作列表
        """
        if not smoothing_config.enable_smoothing or len(delta_actions_list) <= 2:
            return delta_actions_list
        
        print(f"🔄 Applying {smoothing_config.smoothing_method} smoothing to delta actions...")
        
        # 转换为numpy数组进行处理
        delta_actions_array = np.array(delta_actions_list)  # shape: (n_frames, 9)
        smoothed_actions_array = delta_actions_array.copy()
        
        # 分别处理位置和旋转部分
        # 位置部分 (前3维): [dx, dy, dz]
        position_deltas = delta_actions_array[:, :3]  # shape: (n_frames, 3)
        
        # 对每个位置维度分别进行平滑化
        for dim in range(3):  # x, y, z
            original_values = position_deltas[:, dim]
            smoothed_values = self._apply_smoothing_to_sequence(
                original_values, smoothing_config
            )
            
            # 保持积分面积不变：调整缩放因子
            original_integral = np.sum(original_values)
            smoothed_integral = np.sum(smoothed_values)
            
            if abs(smoothed_integral) > 1e-8:  # 避免除零
                scale_factor = original_integral / smoothed_integral
                smoothed_values = smoothed_values * scale_factor
                print(f"   Position dim {dim}: scale_factor = {scale_factor:.6f}")
            
            smoothed_actions_array[:, dim] = smoothed_values
        
        # 旋转部分保持原样（6D旋转表示本身已经相对平滑）
        # smoothed_actions_array[:, 3:9] = delta_actions_array[:, 3:9]
        
        # 转换回列表格式
        smoothed_actions_list = [smoothed_actions_array[i] for i in range(len(delta_actions_list))]
        
        # 验证积分面积
        self._validate_area_preservation(delta_actions_array, smoothed_actions_array)
        
        print("✅ Delta actions smoothing completed")
        return smoothed_actions_list
    
    def _apply_smoothing_to_sequence(self, sequence, smoothing_config):
        """对单个序列应用平滑化"""
        if smoothing_config.smoothing_method == "savgol":
            # Savitzky-Golay滤波：保持局部形状特征
            window_length = min(smoothing_config.window_length, len(sequence))
            if window_length % 2 == 0:
                window_length -= 1  # 确保是奇数
            if window_length < 3:
                window_length = 3
            
            polyorder = min(smoothing_config.polyorder, window_length - 1)
            return savgol_filter(sequence, window_length, polyorder)
            
        elif smoothing_config.smoothing_method == "moving_avg":
            # 移动平均
            window = min(smoothing_config.moving_avg_window, len(sequence))
            return np.convolve(sequence, np.ones(window)/window, mode='same')
            
        elif smoothing_config.smoothing_method == "spline":
            # 样条插值平滑
            if len(sequence) < 4:
                return sequence
            
            x = np.arange(len(sequence))
            spline = UnivariateSpline(x, sequence, s=smoothing_config.spline_smoothing_factor * len(sequence))
            return spline(x)
            
        else:  # "none"
            return sequence
    
    def _validate_area_preservation(self, original_array, smoothed_array):
        """验证积分面积保持不变"""
        original_integrals = np.sum(original_array[:, :3], axis=0)  # x, y, z的积分
        smoothed_integrals = np.sum(smoothed_array[:, :3], axis=0)
        
        print(f"📊 Area Preservation Validation:")
        for dim, (orig, smooth) in enumerate(zip(original_integrals, smoothed_integrals)):
            error = abs(orig - smooth)
            relative_error = error / max(abs(orig), 1e-8)
            print(f"   Position dim {dim}: original={orig:.6f}, smoothed={smooth:.6f}, error={error:.8f} ({relative_error*100:.2f}%)")
            
            if relative_error > 0.01:  # 1%误差阈值
                print(f"   ⚠️  WARNING: Large area change in dimension {dim}!")




def validate_data_format(aligned_data):
    """
    验证转换后的数据格式是否符合要求
    
    Args:
        aligned_data: 对齐后的数据字典
    """
    print("\n" + "="*60)
    print("🔍 Data Format Validation")
    print("="*60)
    
    # 验证COM delta动作
    if 'mpc_target_state' in aligned_data and len(aligned_data['mpc_target_state']) > 0:
        com_actions = aligned_data['mpc_target_state']
        print(f"\n✅ COM Actions (DELTA format):")
        print(f"   - Total frames: {len(com_actions)}")
        print(f"   - Shape per frame: {com_actions[0].shape} (should be (9,))")
        print(f"   - First frame (should be zero): {com_actions[0]}")
        print(f"   - Second frame delta sample: {com_actions[1] if len(com_actions) > 1 else 'N/A'}")
    
    # 验证手臂动作（弧度）
    if 'arm_traj' in aligned_data and len(aligned_data['arm_traj']) > 0:
        arm_actions = aligned_data['arm_traj']
        print(f"\n✅ Arm Actions (ABSOLUTE, in radians):")
        print(f"   - Total frames: {len(arm_actions)}")
        print(f"   - Shape per frame: {arm_actions[0].shape} (should be (14,))")
        print(f"   - Value range: [{np.min([np.min(a) for a in arm_actions]):.4f}, {np.max([np.max(a) for a in arm_actions]):.4f}] rad")
        print(f"   - Sample values (first frame): {arm_actions[0]}")
        
        # 检查是否在合理的弧度范围内（-π到π或稍大）
        all_arm_values = np.concatenate(arm_actions)
        if np.max(np.abs(all_arm_values)) > 10:
            print(f"   ⚠️  WARNING: Some arm values seem too large for radians (max: {np.max(np.abs(all_arm_values)):.2f})")
        else:
            print(f"   ✓ Arm values are in reasonable radian range")
    
    # 验证步态模式
    if 'gait_time_name' in aligned_data and len(aligned_data['gait_time_name']) > 0:
        gait_actions = aligned_data['gait_time_name']
        gait_values = [g[0] for g in gait_actions]
        print(f"\n✅ Gait Mode (ABSOLUTE):")
        print(f"   - Total frames: {len(gait_actions)}")
        print(f"   - Shape per frame: {gait_actions[0].shape} (should be (1,))")
        print(f"   - Unique values: {np.unique(gait_values)} (should be in [1, 2])")
        print(f"   - Walk (1) count: {gait_values.count(1)}")
        print(f"   - Stance (2) count: {gait_values.count(2)}")
        
        # 检查是否只包含1和2
        if set(gait_values).issubset({1, 2}):
            print(f"   ✓ Gait values are valid (1=walk, 2=stance)")
        else:
            print(f"   ⚠️  WARNING: Found unexpected gait values: {set(gait_values)}")
    
    print("\n" + "="*60)


def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Bag to LeRobot Dataset Converter (DELTA ACTION VERSION)',
        epilog='This converter creates datasets with COM delta actions, arm absolute actions (radians), and gait absolute mode.'
    )
    parser.add_argument('--bag_path', type=str, 
                       default='/home/lab/kuavo-manip/raw_data/grasp_kmpc_pose_green_1_3_test/episode_2.bag',
                       help='Path to the rosbag file or directory containing bag files')
    parser.add_argument('--dt', type=float, default=0.1,
                       help='Time interval for alignment (seconds, default: 0.1s = 10Hz)')
    parser.add_argument('--target-dir', type=str, default='./lerobot_dataset',
                       help='Target directory for LeRobot dataset (default: ./lerobot_dataset)')
    parser.add_argument('--repo-id', type=str, default='kuavo_com_control',
                       help='Repository ID for LeRobot dataset (default: kuavo_com_control)')
    parser.add_argument('--task', type=str, default='com_control',
                       help='Task name for LeRobot dataset (default: com_control)')
    parser.add_argument('--dataset-mode', type=str, choices=['video', 'image'], default='video',
                       help='Dataset mode for LeRobot dataset (default: video)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing target directory (default: False)')
    
    # 平滑化相关参数
    parser.add_argument('--disable-smoothing', action='store_true',
                       help='Disable delta action smoothing (default: False)')
    parser.add_argument('--smoothing-method', type=str, 
                       choices=['savgol', 'moving_avg', 'spline', 'none'], default='savgol',
                       help='Smoothing method for delta actions (default: savgol)')
    parser.add_argument('--savgol-window', type=int, default=11,
                       help='Savitzky-Golay window length (must be odd, default: 11)')
    parser.add_argument('--savgol-polyorder', type=int, default=3,
                       help='Savitzky-Golay polynomial order (default: 3)')
    parser.add_argument('--moving-avg-window', type=int, default=5,
                       help='Moving average window size (default: 5)')
    parser.add_argument('--spline-smoothing-factor', type=float, default=0.5,
                       help='Spline smoothing factor (default: 0.5)')
    
    # 图像特征提取相关参数
    parser.add_argument('--use-image-features', action='store_true',
                       help='Extract image features using ResNet backbone instead of storing raw images (default: False)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device for image feature extraction (default: cuda:0)')
    
    args = parser.parse_args()
    
    try:
        # 检查LeRobot是否可用
        if not LEROBOT_AVAILABLE:
            print("Error: LeRobot is not available. Cannot convert to dataset. Please install LeRobot first.")
            return
        
        print("Converting to LeRobot dataset mode...")
        print("=" * 80)
        print("🚀 DELTA ACTION MODE ENABLED")
        print("   - COM actions: DELTA (relative to previous frame)")
        print("   - Arm actions: ABSOLUTE (in radians)")
        print("   - Gait mode: ABSOLUTE (1=walk, 2=stance)")
        print("=" * 80)
        
        # 创建平滑化配置
        smoothing_config = SmoothingConfig(
            enable_smoothing=not args.disable_smoothing,
            smoothing_method=args.smoothing_method,
            window_length=args.savgol_window,
            polyorder=args.savgol_polyorder,
            moving_avg_window=args.moving_avg_window,
            spline_smoothing_factor=args.spline_smoothing_factor
        )
        
        # 创建数据转换器
        data_converter = DataConverter(
            use_image_features=True, # 直接使用图像特征
            device=args.device
        )
        
        # 检查输入路径是文件还是目录
        input_path = Path(args.bag_path)
        if input_path.is_file():
            # 单个文件模式
            print("\n📦 Single bag file mode")
            try:
                aligned_data, time_grid = data_converter.process_rosbag(args.bag_path, dt=args.dt, smoothing_config=smoothing_config)
                
                # 验证数据格式
                validate_data_format(aligned_data)
                
                dataset = data_converter.convert_aligned_data_to_lerobot(
                    aligned_data=aligned_data,
                    time_grid=time_grid,
                    target_dir=Path(args.target_dir),
                    repo_id=args.repo_id,
                    task=args.task,
                    mode=args.dataset_mode,
                    dataset_config=DEFAULT_DATASET_CONFIG,
                    overwrite=args.overwrite,
                )
                if dataset:
                    print("\n✅ Dataset conversion completed successfully!")
                else:
                    print("\n❌ Dataset conversion failed!")
            except Exception as e:
                print(f"\n❌ Error processing single bag file: {e}")
        elif input_path.is_dir():
            # 批量处理模式
            print("\n📦 Batch processing mode")
            bag_files = data_converter.collect_bag_files(args.bag_path)
            if not bag_files:
                print("❌ No bag files found in the specified directory")
                return
            
            # 验证第一个bag文件的数据格式
            print(f"\n🔍 Validating first bag file: {bag_files[0].name}")
            try:
                first_aligned_data, _ = data_converter.process_rosbag(str(bag_files[0]), dt=args.dt, smoothing_config=smoothing_config)
                validate_data_format(first_aligned_data)
                print(f"✅ First bag validation passed. Processing all {len(bag_files)} bag files...\n")
            except Exception as e:
                print(f"⚠️  Warning: First bag validation failed: {e}")
                print(f"Continuing with conversion anyway...\n")
            
            dataset = data_converter.convert_multiple_bags_to_lerobot(
                bag_files=bag_files,
                target_dir=Path(args.target_dir),
                repo_id=args.repo_id,
                task=args.task,
                mode=args.dataset_mode,
                dataset_config=DEFAULT_DATASET_CONFIG,
                smoothing_config=smoothing_config,
                overwrite=args.overwrite,
                dt=args.dt,
            )
            if dataset:
                print("\n✅ Batch dataset conversion completed successfully!")
            else:
                print("\n❌ Batch dataset conversion failed!")
        else:
            print(f"\n❌ Error: Path {args.bag_path} does not exist")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
