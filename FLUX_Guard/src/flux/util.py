import os
from dataclasses import dataclass
from typing import Optional, Union

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors

from flux.model import Flux, FluxParams
from flux.modules.autoencoder import AutoEncoder, AutoEncoderParams
from flux.modules.conditioner import HFEmbedder


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams

    # 本地路径优先
    ckpt_path: Optional[str] = None
    ae_path: Optional[str] = None

    # 可选：HF 下载备用（你不需要可以全设 None）
    repo_id: Optional[str] = None
    repo_flow: Optional[str] = None
    repo_ae: Optional[str] = None


configs = {
    "flux-dev": ModelSpec(
        repo_id=None,
        repo_flow=None,
        repo_ae=None,
        ckpt_path="/share/flux1-dev.safetensors",
        ae_path="/share/flux/ae.safetensors",
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id=None,  # 如需HF下载，填 repo_id="black-forest-labs/FLUX.1-schnell" 之类
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        ae_path=os.getenv("AE"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing[:50]))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected[:50]))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing[:50]))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected[:50]))


def _resolve_ckpt_path(cfg: ModelSpec, hf_download: bool) -> str:
    if cfg.ckpt_path and os.path.exists(cfg.ckpt_path):
        return cfg.ckpt_path

    if cfg.ckpt_path and not os.path.exists(cfg.ckpt_path):
        raise FileNotFoundError(f"本地模型路径不存在：{cfg.ckpt_path}")

    if hf_download and cfg.repo_id and cfg.repo_flow:
        return hf_hub_download(repo_id=cfg.repo_id, filename=cfg.repo_flow, cache_dir="./hf_cache")

    raise ValueError("未配置本地 ckpt_path，且未启用/无法使用 HF 下载（repo_id/repo_flow 为空）。")


def _resolve_ae_path(cfg: ModelSpec, hf_download: bool) -> Optional[str]:
    if cfg.ae_path and os.path.exists(cfg.ae_path):
        return cfg.ae_path

    if cfg.ae_path and not os.path.exists(cfg.ae_path):
        raise FileNotFoundError(f"本地 AE 路径不存在：{cfg.ae_path}")

    if hf_download and cfg.repo_id and cfg.repo_ae:
        return hf_hub_download(repo_id=cfg.repo_id, filename=cfg.repo_ae, cache_dir="./hf_cache")

    return None


def load_flow_model(
    name: str,
    device: Union[str, torch.device] = "cuda",
    hf_download: bool = False,
) -> Flux:
    if name not in configs:
        raise ValueError(f"模型名称 '{name}' 不存在，支持：{', '.join(configs.keys())}")

    cfg = configs[name]
    ckpt_path = _resolve_ckpt_path(cfg, hf_download=hf_download)

    # meta 初始化省内存
    with torch.device("meta"):
        model = Flux(cfg.params).to(torch.bfloat16)

    print(f"加载 Flow 权重：{ckpt_path} -> {device}")
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_safetensors(ckpt_path, device=str(device))
    else:
        state_dict = torch.load(ckpt_path, map_location=device)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False, assign=True)
    if missing_keys or unexpected_keys:
        print_load_warning(missing_keys, unexpected_keys)

    model = model.to(dtype=torch.bfloat16).eval()
    return model.to(device)


def load_t5(device: Union[str, torch.device] = "cuda", max_length: int = 512) -> HFEmbedder:
    return HFEmbedder(
        "/share/t5-v1_1-xxl",
        max_length=max_length,
        is_clip=False,
        torch_dtype=torch.bfloat16,
    ).to(device)


def load_clip(device: Union[str, torch.device] = "cuda") -> HFEmbedder:
    return HFEmbedder(
        "/share/openaiclip-vit-large-patch14",
        max_length=77,
        is_clip=True,
        torch_dtype=torch.bfloat16,
    ).to(device)


def load_ae(name: str, device: Union[str, torch.device] = "cuda", hf_download: bool = False) -> AutoEncoder:
    if name not in configs:
        raise ValueError(f"AE 模型名 '{name}' 不存在，支持：{', '.join(configs.keys())}")

    cfg = configs[name]
    ae_path = _resolve_ae_path(cfg, hf_download=hf_download)

    print("Init AE")
    with torch.device("meta" if ae_path is not None else device):
        ae = AutoEncoder(cfg.ae_params)

    if ae_path is not None:
        sd = load_safetensors(ae_path, device=str(device))
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        if missing or unexpected:
            print_load_warning(missing, unexpected)
        ae = ae.to(dtype=torch.bfloat16)

    return ae.to(device)
