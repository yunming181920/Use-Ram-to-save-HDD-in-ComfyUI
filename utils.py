import torch
import math
import struct
import comfy.checkpoint_pickle
import safetensors.torch
import numpy as np
from PIL import Image
import logging
import itertools
from torch.nn.functional import interpolate
from einops import rearrange
from comfy.cli_args import args
import io
from io import BytesIO
import logging
import gc
from safetensors.torch import load_file
import tempfile
import shutil
import os
from pathlib import Path
import struct
import json
import safetensors.torch
import string
import subprocess
import time
import sys
import stat
MMAP_TORCH_FILES = args.mmap_torch_files
DISABLE_MMAP = args.disable_mmap

ALWAYS_SAFE_LOAD = False

def load_torch_file(ckpt, safe_load=False, device=None, return_metadata=False, use_ram_cache=True):
    """
    加载 torch 文件，支持从内存中加载以优化机械硬盘性能

    Args:
        ckpt: 文件路径
        safe_load: 是否安全加载
        device: 目标设备
        return_metadata: 是否返回元数据
        use_ram_cache: 是否先复制到内存再加载（针对 safetensors）
    """
    if device is None:
        device = torch.device("cpu")

    metadata = None

    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        return _load_safetensors(ckpt, device, return_metadata, use_ram_cache)
    else:
        return _load_torch_checkpoint(ckpt, device, return_metadata, safe_load)


def _load_safetensors(ckpt, device, return_metadata, use_ram_cache):
    """加载 safetensors 文件，可选择从内存加载"""

    file_size = os.path.getsize(ckpt)
    logging.info(f"Safetensors 文件大小: {file_size / 1024 / 1024:.2f} MB")

    # 如果文件较小或禁用 RAM 缓存，直接加载
    if not use_ram_cache or file_size < 100 * 1024 * 1024:  # 小于 100MB 直接加载
        return _load_safetensors_direct(ckpt, device, return_metadata)

    # 文件较大，先复制到内存再加载
    logging.info("将文件复制到内存...")

    z_path = Path("Z:/")
    target_path = z_path / Path(ckpt).name

    # 清空 Z 盘内容
    if z_path.exists() and z_path.is_dir():
        for item in z_path.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except Exception as e:
                logging.warning(f"删除 Z 盘内容失败: {item}, 错误: {e}")

    # 复制文件到 Z 盘
    shutil.copy2(ckpt, target_path)
    logging.info(f"📂 文件已复制到 Z 盘: {target_path}")

    return _load_safetensors_direct(target_path, device, return_metadata)





def _load_safetensors_direct(ckpt, device, return_metadata):
    """直接加载 safetensors 文件"""
    try:
        with safetensors.safe_open(ckpt, framework="pt", device=device.type) as f:
            sd = {}
            for k in f.keys():
                tensor = f.get_tensor(k)
                # 确保张量在正确的设备上
                if tensor.device != device:
                    tensor = tensor.to(device=device, copy=True)
                sd[k] = tensor

            metadata = f.metadata() if return_metadata else None
            return (sd, metadata) if return_metadata else sd

    except Exception as e:
        if len(e.args) > 0:
            message = e.args[0]
            if "HeaderTooLarge" in message:
                raise ValueError(
                    f"{message}\n\nFile path: {ckpt}\n\n"
                    "The safetensors file is corrupt or invalid. "
                    "Make sure this is actually a safetensors file and not a ckpt or pt or other filetype."
                )
            if "MetadataIncompleteBuffer" in message:
                raise ValueError(
                    f"{message}\n\nFile path: {ckpt}\n\n"
                    "The safetensors file is corrupt/incomplete. "
                    "Check the file size and make sure you have copied/downloaded it correctly."
                )
        raise e


def _load_torch_checkpoint(ckpt, device, return_metadata, safe_load):
    """加载传统 torch checkpoint 文件到内存再读取"""
    torch_args = {}

    logging.info(f"将 {ckpt} 文件加载到内存中...")
    try:
        with open(ckpt, "rb") as f:
            file_bytes = f.read()
    except Exception as e:
        logging.error(f"读取文件 {ckpt} 失败: {e}")
        raise

    buffer = BytesIO(file_bytes)

    if safe_load:
        logging.info("使用安全加载模式...")
        pl_sd = torch.load(buffer, map_location=device, weights_only=True, **torch_args)
    else:
        logging.warning(f"WARNING: loading {ckpt} unsafely, upgrade your pytorch to 2.4 or newer")
        pl_sd = torch.load(buffer, map_location=device)

    # 解析状态字典
    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        if len(pl_sd) == 1:
            key = list(pl_sd.keys())[0]
            sd = pl_sd[key]
            if not isinstance(sd, dict):
                sd = pl_sd
        else:
            sd = pl_sd

    metadata = None
    return (sd, metadata) if return_metadata else sd
