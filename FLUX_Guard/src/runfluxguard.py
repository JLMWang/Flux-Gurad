#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"

import re
import time
import json
import argparse
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ========== 你的工程内依赖 ==========
from flux.sampling import denoise_inver, denoise_gen, get_schedule, unpack, prepare_from_cached
from flux.util import configs, load_ae, load_clip, load_flow_model, load_t5
from auto_mask import generate_mask, adapt_mask_to_latent
import random
import numpy as np
import torch
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path
from typing import List, Tuple

_THREADS_CONFIGURED = False

def configure_threads_once():
    global _THREADS_CONFIGURED
    if _THREADS_CONFIGURED:
        return
    _THREADS_CONFIGURED = True

    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)



def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ✅ 关键：确定性 + 不涨显存
    torch.backends.cudnn.benchmark = False          # 必须关，否则会漂
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # ✅ 不要强行关 mem_efficient（否则显存上升明显）
    # 可选：关 flash（更稳一点），但保留 mem_efficient（省显存）
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        # 不要强制 enable_math_sdp(True)
    except Exception:
        pass
def natural_sort_key(s: str) -> list:
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]
def get_image_pairs(
    source_folder: str,
    adv_folder: str,  # 含4张对抗图像的文件夹
    adv_img_index: int  # 指定选对抗文件夹里的第几张（0-3，按自然排序）
) -> List[Tuple[str, str]]:
    """
    核心逻辑：
    - 源侧：文件夹（多张图像）
    - 对抗侧：文件夹（固定4张），指定选1张
    - 所有源图像都配对这1张选中的对抗图（一批运行处理所有源图，共用1张对抗图）
    """
    img_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    # ========== 步骤1：处理源文件夹（所有图像） ==========
    source_files = [
        f for f in os.listdir(source_folder)
        if Path(f).suffix.lower() in img_extensions and os.path.isfile(os.path.join(source_folder, f))
    ]
    source_files.sort(key=natural_sort_key)  # 自然排序
    source_paths = [os.path.join(source_folder, f) for f in source_files]
    if not source_paths:
        raise ValueError(f"源文件夹 {source_folder} 无有效图像！")

    # ========== 步骤2：处理对抗文件夹（仅选指定的1张） ==========
    # 获取对抗文件夹的4张图（自然排序）
    adv_files = [
        f for f in os.listdir(adv_folder)
        if Path(f).suffix.lower() in img_extensions and os.path.isfile(os.path.join(adv_folder, f))
    ]
    adv_files.sort(key=natural_sort_key)  # 按名称自然排序（保证选图固定）
    if len(adv_files) != 4:
        raise ValueError(f"对抗文件夹必须包含4张图像，当前找到 {len(adv_files)} 张！")
    # 验证索引范围（0-3）
    if adv_img_index < 0 or adv_img_index >= 4:
        raise ValueError(f"adv_img_index 必须是0-3，当前输入：{adv_img_index}")
    # 选中指定的1张对抗图
    target_adv_path = os.path.join(adv_folder, adv_files[adv_img_index])
    print(f"选中对抗图像：{adv_files[adv_img_index]}（索引{adv_img_index}）")

    # ========== 步骤3：所有源图像配对这1张对抗图 ==========
    img_pairs = [(src_path, target_adv_path) for src_path in source_paths]
    print(f"共配对 {len(img_pairs)} 对（所有源图像 → 同1张对抗图）")
    return img_pairs

# -------------------------- 采样配置 --------------------------
@dataclass
class SamplingOptions:
    source_prompt: str
    target_prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None
    attack_start_step: int


# -------------------------- 图像编码：RGB -> latent --------------------------
@torch.inference_mode()
def encode(init_image: np.ndarray, torch_device: torch.device, ae: torch.nn.Module) -> torch.Tensor:
    img_tensor = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1.0
    img_tensor = img_tensor.unsqueeze(0).to(torch_device)  # [1,3,H,W]
    img_tensor = img_tensor.to(torch.bfloat16)
    latent = ae.encode(img_tensor)
    return latent


# -------------------------- 方案1：启动时预计算缓存文本编码 --------------------------
@torch.inference_mode()
def precompute_prompt_cache(
    name: str,
    torch_device: torch.device,
    source_prompt: str,
    target_prompt: str,
    is_main_process: bool,
):
    if is_main_process:
        print("=== 预计算并缓存文本编码（T5/CLIP 只跑一次）===")

    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip_text = load_clip(torch_device)

    source_txt = t5([source_prompt]).detach()
    source_vec = clip_text([source_prompt]).detach()
    target_txt = t5([target_prompt]).detach()
    target_vec = clip_text([target_prompt]).detach()

    del t5, clip_text

    cache = {
        "source_txt": source_txt,
        "source_vec": source_vec,
        "target_txt": target_txt,
        "target_vec": target_vec,
    }

    if is_main_process:
        print("文本编码缓存完成：后续每张图将直接复用 txt/vec（不再加载 T5/CLIP）\n")
    return cache


# -------------------------- 反演包装（带监控 callback） --------------------------
@torch.inference_mode()
def denoise_inversion(
    model: torch.nn.Module,
    timesteps,
    ae: torch.nn.Module,
    img_height: int,
    img_width: int,
    device: torch.device,
    save_root: str,
    is_main_process: bool,
    **kwargs,
):
    total_steps = len(timesteps) - 1
    pbar = tqdm(total=total_steps, desc="反演去噪", unit="step", disable=not is_main_process)

    def progress_callback(step, total):
        pbar.update(1)
        if is_main_process:
            pbar.set_postfix({"inv": f"{step + 1}/{total}"})
    result, info = denoise_inver(
        model=model,
        img=kwargs["img"],
        img_ids=kwargs["img_ids"],
        txt=kwargs["txt"],
        txt_ids=kwargs["txt_ids"],
        vec=kwargs["vec"],
        timesteps=timesteps,
        info=kwargs["info"],
        guidance=1.0,
        callback=progress_callback,
        ae=ae,
        img_height=img_height,
        img_width=img_width,
        device=device,
        save_mid_imgs=False,
        save_root_dir=save_root,
        is_main_process=is_main_process,
        record_inv_pred=True,
    )

    pbar.close()
    if is_main_process:
        print(f"反演完成：{total_steps} steps")
    return result, info


# -------------------------- 生成包装（带监控 callback） --------------------------
def denoise_generation(
    model: torch.nn.Module,
    timesteps,
    ae: torch.nn.Module,
    img_height: int,
    img_width: int,
    device: torch.device,
    save_root: str,
    guidance: float,
    is_main_process: bool,
    attack_start_step: int,
    **kwargs,
):
    total_steps = len(timesteps) - 1
    pbar = tqdm(total=total_steps, desc="生成去噪", unit="step", disable=not is_main_process)

    use_inv_pred_steps = list(range(5, 20))

    def progress_callback(step, total):
        pbar.update(1)
        if is_main_process:
            pbar.set_postfix({"gen": f"{step + 1}/{total}", "g": f"{guidance:.2f}"})
    result_latent, info = denoise_gen(
        model=model,
        img=kwargs["img"],
        img_ids=kwargs["img_ids"],
        txt=kwargs["txt"],
        txt_ids=kwargs["txt_ids"],
        vec=kwargs["vec"],
        timesteps=timesteps,
        info=kwargs["info"],
        guidance=guidance,
        callback=progress_callback,
        ae=ae,
        img_height=img_height,
        img_width=img_width,
        device=device,
        save_mid_imgs=False,
        save_root_dir=save_root,
        is_main_process=is_main_process,
        attack_start_step=attack_start_step,
        init_img_tensor=kwargs.get("init_img_tensor", None),
        tgt_image=kwargs.get("tgt_image", None),
        use_inv_pred_steps=use_inv_pred_steps,
    )

    pbar.close()
    if is_main_process:
        print(f"生成完成：{total_steps} steps | guidance={guidance}")
    return result_latent, info


# -------------------------- 单图处理 --------------------------
def process_single_image(
    src_img_path: str,
    adv_img_path: str,
    args,
    model,
    ae,
    torch_device,
    is_main_process: bool,
    opts_base: SamplingOptions,
    text_cache: dict,
    idx: int,
):
    seed_all(int(args.seed))
    img_name = Path(src_img_path).stem
    output_root = args.output_dir
    final_dir = os.path.join(output_root, "final")
    log_dir = os.path.join(output_root, "logs")
    img_feature_path = os.path.join(args.feature_path, img_name)

    # ========== 新增：文件存在性检查 ==========
    final_path = os.path.join(final_dir, f"{img_name}.png")  # 和后续保存路径完全一致
    if os.path.exists(final_path):
        if is_main_process:
            print(f"[SKIP] 已存在生成文件：{final_path}，跳过处理")
        return  # 直接返回，跳过后续所有处理逻辑
    # ===========================================

    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    if is_main_process:
        os.makedirs(img_feature_path, exist_ok=True)
    # 源图像预处理
    # 源图像预处理：读一张 resize 一张到 512x512
    init_pil = Image.open(src_img_path).convert("RGB")
    init_pil = init_pil.resize((512, 512), Image.BICUBIC)
    init_image = np.array(init_pil)  # uint8, [H,W,3]

    new_h, new_w = 512, 512
    src_width, src_height = new_w, new_h
    # 编码
    ae = ae.to(torch_device)
    src_latent = encode(init_image, torch_device, ae)

    init_img_tensor = torch.from_numpy(init_image).permute(2, 0, 1).float() / 255.0
    init_img_tensor = init_img_tensor.unsqueeze(0).to(torch_device)

    # 掩码
    mask_path, raw_mask, valid_ratio, ps_weight_map = generate_mask(
        prompt=opts_base.target_prompt,
        img_path=src_img_path,
        save_path=None,  # 不保存mask
        device=str(torch_device),
        similarity_threshold=0.35,
        use_18_regions=True,
    )

    if valid_ratio < 10.0:
        args.inject = 3
        if is_main_process:
            print(f"[INFO] 有效区域占比 {valid_ratio:.2f}% < 10%，设置 inject=3")
    else:
        args.inject = 5
        if is_main_process:
            print(f"[INFO] 有效区域占比 {valid_ratio:.2f}% ≥ 10%，设置 inject=5")

    # 适配编辑区域 mask 到图像尺度
    if raw_mask.shape[0] != new_h or raw_mask.shape[1] != new_w:
        resized_mask = cv2.resize(
            raw_mask,
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        resized_mask = raw_mask

    # PS 权重图 resize 到图像尺度
    if ps_weight_map.shape[0] != new_h or ps_weight_map.shape[1] != new_w:
        ps_resized = cv2.resize(
            ps_weight_map.astype(np.float32),
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        ps_resized = ps_weight_map.astype(np.float32)

    # PS 权重图归一化处理
    ps_max = ps_resized.max(initial=1e-6)
    ps_u8 = np.clip(ps_resized / ps_max, 0.0, 1.0) * 255.0
    ps_u8 = ps_u8.astype(np.uint8)


    flow_module = model.module if isinstance(model, DDP) else model
    model_dtype = next(flow_module.parameters()).dtype

    mask_tensor = adapt_mask_to_latent(
        mask=resized_mask,
        img_height=new_h,
        img_width=new_w,
        device=torch_device,
        model_dtype=model_dtype,
    )
    ps_mask_tensor = adapt_mask_to_latent(
        mask=ps_u8,
        img_height=new_h,
        img_width=new_w,
        device=torch_device,
        model_dtype=model_dtype,
    )
    # 对抗图像
    tgt_image_tensor = None
    if adv_img_path:
        tgt_pil = Image.open(adv_img_path).convert("RGB").resize((new_w, new_h), Image.BILINEAR)
        tgt_np = np.array(tgt_pil, dtype=np.float32) / 255.0
        tgt_image_tensor = torch.from_numpy(tgt_np).permute(2, 0, 1).unsqueeze(0).to(torch_device)

    # opts
    opts = SamplingOptions(
        source_prompt=opts_base.source_prompt,
        target_prompt=opts_base.target_prompt,
        width=src_width,
        height=src_height,
        num_steps=opts_base.num_steps,
        guidance=opts_base.guidance,
        seed=args.seed,  # 仅记录，不再用于设置RNG
        attack_start_step=opts_base.attack_start_step,
    )

    info: dict = {
        "feature_path": img_feature_path,
        "feature": {},
        "inject_step": args.inject,
        "similarity_log": [],
        "mask_tensor": mask_tensor,  # 编辑区域 mask
        "ps_mask_tensor": ps_mask_tensor,  # 整脸 PS 权重
        "is_main_process": is_main_process,
    }
    # 使用缓存 txt/vec
    inp_inversion = prepare_from_cached(img=src_latent, txt=text_cache["source_txt"], vec=text_cache["source_vec"])
    inp_generation = prepare_from_cached(img=src_latent, txt=text_cache["target_txt"], vec=text_cache["target_vec"])

    timesteps_inversion = get_schedule(opts.num_steps, inp_inversion["img"].shape[1], shift=(args.name != "flux-schnell"))
    timesteps_generation = get_schedule(opts.num_steps, inp_generation["img"].shape[1], shift=(args.name != "flux-schnell"))

    # 反演
    z_noise, info = denoise_inversion(
        model=model,
        timesteps=timesteps_inversion,
        ae=ae,
        img_height=opts.height,
        img_width=opts.width,
        device=torch_device,
        save_root=output_root,
        is_main_process=is_main_process,
        img=inp_inversion["img"],
        img_ids=inp_inversion["img_ids"],
        txt=inp_inversion["txt"],
        txt_ids=inp_inversion["txt_ids"],
        vec=inp_inversion["vec"],
        info=info,
    )

    # 生成
    inp_generation["img"] = z_noise
    x_edited, info = denoise_generation(
        model=model,
        timesteps=timesteps_generation,
        ae=ae,
        img_height=opts.height,
        img_width=opts.width,
        device=torch_device,
        save_root=output_root,
        guidance=opts.guidance,
        is_main_process=is_main_process,
        attack_start_step=opts.attack_start_step,
        img=inp_generation["img"],
        img_ids=inp_generation["img_ids"],
        txt=inp_generation["txt"],
        txt_ids=inp_generation["txt_ids"],
        vec=inp_generation["vec"],
        info=info,
        init_img_tensor=init_img_tensor,
        tgt_image=tgt_image_tensor,
    )
    # 解码 & 保存（主进程）——统一保存 PNG
    if is_main_process:
        src_stem = Path(src_img_path).stem
        final_path = os.path.join(final_dir, f"{src_stem}.png")
        batch_x = unpack(x_edited.float(), opts.height, opts.width)
        assert batch_x.shape[0] == 1, f"expected B=1, got {batch_x.shape[0]}"
        x = batch_x[0].unsqueeze(0)  # [1,C,H,W]
        # ✅ 关键：让输入 dtype 与 AE 参数 dtype 一致
        ae_dtype = next(ae.parameters()).dtype  # e.g. torch.bfloat16
        x = x.to(device=torch_device, dtype=ae_dtype)
        with torch.no_grad():
            x_decoded = ae.decode(x)  # dtype 一致就不会报错
        # 保存前用 float32 做映射与 clamp
        img01 = ((x_decoded.float() + 1.0) * 0.5).clamp(0, 1)

        save_image(img01, final_path)

    # 解码 & 保存（主进程）
    # if is_main_process:
    #     final_filename = Path(src_img_path).name
    #     final_path = os.path.join(final_dir, f"final_{final_filename}.jpg")
    #
    #     batch_x = unpack(x_edited.float(), opts.height, opts.width)
    #     for x in batch_x:
    #         x = x.unsqueeze(0)
    #         with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
    #             x_decoded = ae.decode(x)
    #         x_decoded = x_decoded.clamp(-1, 1)
    #         x_decoded = rearrange(x_decoded[0], "c h w -> h w c")
    #         img_out = Image.fromarray((127.5 * (x_decoded + 1.0)).cpu().byte().numpy())
    #         img_out.save(final_path, quality=95, subsampling=0)
    #
    #     adv_state = info.get("adv_state", {})
    #     similarity_log = adv_state.get("similarity_log", None)
    #     if similarity_log:
    #         log_path = os.path.join(log_dir, f"{img_name}_similarity_log.json")
    #         with open(log_path, "w", encoding="utf-8") as f:
    #             json.dump(similarity_log, f, indent=2, ensure_ascii=False)


# -------------------------- 主函数 --------------------------
def main(args):
    configure_threads_once()  # ✅ 只做一次
    seed_all(int(args.seed))
    # 分布式 / 单卡
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        torch_device = torch.device(f"cuda:{local_rank}")
        is_main_process = (local_rank == 0)
        ddp_enabled = True
    else:
        local_rank = 0
        torch_device = torch.device(args.device)
        is_main_process = True
        ddp_enabled = False

    if is_main_process:
        print(f"[INIT] LOCAL_RANK={local_rank} device={torch_device}")

    name = args.name
    if name not in configs:
        raise ValueError(f"未知模型名：{name}，支持模型：{', '.join(configs.keys())}")

    num_steps = args.num_steps if args.num_steps is not None else (4 if name == "flux-schnell" else 25)
    if is_main_process:
        print(f"[CFG] model={name} num_steps={num_steps} guidance={args.guidance} attack_start={args.attack_start_step}")

    # 图像对
    img_pairs = get_image_pairs(args.source_img_folder, args.adv_img_folder,args.adv_img_index)

    # 文本缓存
    text_cache = precompute_prompt_cache(
        name=name,
        torch_device=torch_device,
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        is_main_process=is_main_process,
    )
    # 加载模型/AE
    if is_main_process:
        print("[LOAD] loading Flow + AE ...")

    model_device = "cpu" if args.offload else torch_device
    ae_device = "cpu" if args.offload else torch_device

    model = load_flow_model(name, device=model_device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    ae = load_ae(name, device=ae_device)

    if not args.offload:
        model = model.to(torch_device)
        ae = ae.to(torch_device)

    if ddp_enabled:
        if args.offload:
            model = model.to(torch_device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    opts_base = SamplingOptions(
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        width=0,
        height=0,
        num_steps=num_steps,
        guidance=args.guidance,
        seed=None,
        attack_start_step=args.attack_start_step,
    )

    if is_main_process:
        print(f"\n[RUN] start batch: {len(img_pairs)} pairs")

    for idx, (src_img_path, adv_img_path) in enumerate(img_pairs, 1):
        if is_main_process:
            print(f"\n{'='*80}\n[RUN] {idx}/{len(img_pairs)}: {Path(src_img_path).name}")

        try:
            process_single_image(
                src_img_path=src_img_path,
                adv_img_path=adv_img_path,
                args=args,
                model=model,
                ae=ae,
                torch_device=torch_device,
                is_main_process=is_main_process,
                opts_base=opts_base,
                text_cache=text_cache,
                idx=idx,
            )
        except Exception as e:
            msg = str(e)
            print(f"[ERR] image {src_img_path}: {msg}")
            if "CUDA out of memory" in msg:
                print("\n❌ CUDA OOM，退出。")
                if ddp_enabled and dist.is_initialized():
                    dist.destroy_process_group()
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flux batch edit + mem monitor + reserved warmup + allocated guard")

    parser.add_argument("--name", default="flux-dev", type=str)

    parser.add_argument("--source_img_folder", default=" ", type=str)
    parser.add_argument("--adv_img_folder", default="/share/w/target", type=str)  # 含4张对抗图的文件夹
    parser.add_argument("--adv_img_index", type=int, default=3, help="从对抗文件夹选第几张（0-3，按名称自然排序）")
    parser.add_argument("--source_prompt", default="A person", type=str)
    parser.add_argument("--target_prompt", default="A person wearing red lipstick and red-purple hair", type=str)

    parser.add_argument("--feature_path", default="feature", type=str)
    parser.add_argument("--output_dir", default=" ", type=str)
    parser.add_argument("--guidance", type=float, default=5.5)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--attack_start_step", type=int, default=20)
    parser.add_argument("--inject", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2)

    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    main(args)
