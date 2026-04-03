import math
import os
from typing import Callable, Optional, Dict, List, Tuple

import lpips
import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
import torch.nn as nn
import torch.nn.functional as F
from models import irse, ir152, facenet
from tqdm import tqdm

def normalize_face_input(img: Tensor, target_size: tuple) -> Tensor:
    img = F.interpolate(img, size=target_size, mode="bilinear", align_corners=False)
    img = img * 2.0 - 1.0
    return img.contiguous()
@torch.inference_mode()
def prepare_from_cached(img: Tensor, txt: Tensor, vec: Tensor) -> Dict[str, Tensor]:
    bs, c, h, w = img.shape
    img_seq = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    img_ids = torch.zeros(h // 2, w // 2, 3, device=img.device)
    img_ids[..., 1] = torch.arange(h // 2, device=img.device)[:, None]
    img_ids[..., 2] = torch.arange(w // 2, device=img.device)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    txt_ids = torch.zeros(bs, txt.shape[1], 3, device=img.device)

    return {
        "img": img_seq,
        "img_ids": img_ids,
        "txt": txt.to(img.device),
        "txt_ids": txt_ids,
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> List[float]:
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )


def denoise_inver(
    model: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: List[float],
    info: dict,
    guidance: float = 1.0,
    callback: Optional[Callable] = None,
    save_mid_imgs: bool = False,
    ae: Optional[torch.nn.Module] = None,
    img_height: Optional[int] = None,
    img_width: Optional[int] = None,
    device: Optional[torch.device] = None,
    save_root_dir: str = "./output",
    is_main_process: bool = True,
    record_inv_pred: bool = True,
) -> Tuple[Tensor, dict]:
    inject_list = [True] * info.get("inject_step", 0) + [False] * (
        len(timesteps[:-1]) - info.get("inject_step", 0)
    )
    timesteps_inverted = timesteps[::-1]
    inject_list_inverted = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    total_steps = len(timesteps_inverted) - 1

    if record_inv_pred:
        info["inv_pred"] = []
        info["inv_pred_mid"] = []

    current_img = img

    # =========================================================
    # OPT-3：t_vec / t_vec_mid 预分配，循环内 fill_ 复用
    # =========================================================
    t_vec = torch.empty((img.shape[0],), device=img.device, dtype=img.dtype)
    t_vec_mid = torch.empty_like(t_vec)

    for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps_inverted[:-1], timesteps_inverted[1:])):
        # 替代 torch.full
        t_vec.fill_(t_curr)

        info["t"] = t_prev
        info["inverse"] = True
        info["second_order"] = False
        info["inject"] = inject_list_inverted[step_idx]

        with torch.no_grad():
            pred, info = model(
                img=current_img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                info=info,
            )

            img_mid = current_img + (t_prev - t_curr) * 0.5 * pred

            # 替代 torch.full
            t_vec_mid.fill_(t_curr + (t_prev - t_curr) * 0.5)

            info["second_order"] = True
            pred_mid, info = model(
                img=img_mid,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec_mid,
                guidance=guidance_vec,
                info=info,
            )

        if record_inv_pred:
            info["inv_pred"].append(pred.detach().to("cpu"))
            info["inv_pred_mid"].append(pred_mid.detach().to("cpu"))

        first_order = (pred_mid - pred) / ((t_prev - t_curr) * 0.5)
        current_img = current_img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order

        if callback is not None:
            callback(step_idx, total_steps)

    return current_img, info


def load_mobile_face_net(device):
    fr_model_m = irse.MobileFaceNet(512)
    path = os.getenv("MOBILEFACE_PTH", "./models/mobile_face.pth")
    fr_model_m.load_state_dict(torch.load(path, map_location="cpu"))
    fr_model_m.to(device).eval()
    return fr_model_m


def load_facenet(device):
    fr_model_facenet = facenet.InceptionResnetV1(num_classes=8631, device=device)
    path = os.getenv("FACENET_PTH", "./models/facenet.pth")
    fr_model_facenet.load_state_dict(torch.load(path, map_location="cpu"))
    fr_model_facenet.to(device).eval()
    return fr_model_facenet


def load_ir152(device):
    fr_model_152 = ir152.IR_152((112, 112))
    path = os.getenv("IR152_PTH", "./models/ir152.pth")
    fr_model_152.load_state_dict(torch.load(path, map_location="cpu"))
    fr_model_152.to(device).eval()
    return fr_model_152


def load_irse50(device):
    fr_model_50 = irse.Backbone(50, 0.6, "ir_se")
    path = os.getenv("IRSE50_PTH", "./models/irse50.pth")
    fr_model_50.load_state_dict(torch.load(path, map_location="cpu"))
    fr_model_50.to(device).eval()
    return fr_model_50


def get_face_models(device):
    mobileface_model = load_mobile_face_net(device)
    facenet_model = load_facenet(device)
    ir152_model = load_ir152(device)
    irse50_model = load_irse50(device)

    mobileface_size = (112, 112)
    facenet_size = (160, 160)
    ir152_size = (112, 112)
    irse50_size = (112, 112)

    return (
        mobileface_model,
        mobileface_size,
        facenet_model,
        facenet_size,
        irse50_model,
        irse50_size,
        ir152_model,
        ir152_size,
    )

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

def _imagenet_norm(x: torch.Tensor, mean, std):
    # x: [B,3,H,W] in [0,1]
    return (x - mean) / std

class VGG19FeatureExtractor(nn.Module):
    """
    提取 VGG19.features 指定层的特征（按 layer index 取）
    常用 index：
      17: relu3_4（浅层，细节/纹理）
      22: relu4_2（深层，结构/语义）
    """
    def __init__(self, device, out_indices=(17, 22)):
        super().__init__()
        self.out_indices = set(out_indices)

        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg.to(device=device, dtype=torch.float32)

        # 注册 mean/std buffer，避免每次 forward 创建 tensor
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1,3,1,1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> dict:
        # x: [B,3,H,W] in [0,1]
        x = x.to(dtype=torch.float32)
        x = _imagenet_norm(x, self.mean, self.std)

        feats = {}
        h = x
        for i, layer in enumerate(self.vgg):
            h = layer(h)
            if i in self.out_indices:
                feats[i] = h
        return feats

class MultiLevelPerceptualLoss(nn.Module):
    """
    两项感知损失：
      - detail: 用浅层（默认 relu3_4, idx=17）保纹理/边缘
      - struct: 用深层（默认 relu4_2, idx=22）保结构/语义
    """
    def __init__(self, extractor: VGG19FeatureExtractor,
                 idx_detail=17):
        super().__init__()
        self.ext = extractor
        self.idx_detail = idx_detail
    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        # img1/img2: [B,3,H,W] in [0,1]
        if img1.shape[1] == 1: img1 = img1.repeat(1,3,1,1)
        if img2.shape[1] == 1: img2 = img2.repeat(1,3,1,1)
        f1 = self.ext(img1)
        f2 = self.ext(img2)
        # MSE 是最稳的（也可换成 L1）
        loss_detail = F.mse_loss(f1[self.idx_detail], f2[self.idx_detail])
        return  loss_detail

def init_vgg_perceptual(device,
                        out_indices=(17,)):  # struct 权重大一些，强约束结构
    ext = VGG19FeatureExtractor(device=device, out_indices=out_indices)
    loss_fn = MultiLevelPerceptualLoss(ext,idx_detail=17)
    return loss_fn


def get_classifier_guidance(
    img_mid: Tensor,
    clean_latent: Tensor,
    clean_rgb_img: Tensor,
    tgt_image: Tensor,
    img_height: int,
    img_width: int,
    device: torch.device,
    step_idx: int,
    ae: nn.Module,
    lr: float,
    state: Optional[dict] = None,
    mask_tensor: Optional[Tensor] = None,
    ps_mask_tensor: Optional[Tensor] = None,  # 掩码张量（与 latent 同形状）
    outside_grad_factor: float = 0.95,
    outside_update_factor: float = 0.95,
    model: Optional[Flux] = None,
    img_ids: Optional[Tensor] = None,
    txt: Optional[Tensor] = None,
    txt_ids: Optional[Tensor] = None,
    vec: Optional[Tensor] = None,
    t_vec: Optional[Tensor] = None,
    guidance_vec: Optional[Tensor] = None,
    info: Optional[dict] = None,
    pred_clean_ref: Optional[Tensor] = None,
    w_pred: float = 1.0,
) -> Tensor:
    if state is None:
        state = {}

    orig_dtype = img_mid.dtype  # 记住传入的 dtype（通常 bf16）
    img_mid = img_mid.detach().clone().to(device=device, dtype=torch.float32)  # ✅ 用 fp32 做优化变量
    # 初始化（只做一次缓存到 state）
    if "vgg_multi_perc" not in state:
        state["vgg_multi_perc"] = init_vgg_perceptual(device).to(device)

    multi_perc = state["vgg_multi_perc"]
    # if "lpips" not in state:
    #     state["lpips"] = lpips.LPIPS(net="vgg").to(device)
    #     for p in state["lpips"].parameters():
    #         p.requires_grad_(False)
    #     state["lpips"].eval()
    # loss_fn_lpips = state["lpips"]



    if "face_models" not in state:
        state["face_models"] = get_face_models(device=device)
        (m1, _, m2, _, m3, _, m4, _) = state["face_models"]
        for mm in [m1, m2, m3, m4]:
            for p in mm.parameters():
                p.requires_grad_(False)
            mm.eval()

    (
        mobileface_model,
        mobileface_size,
        facenet_model,
        facenet_size,
        irse50_model,
        irse50_size,
        ir152_model,
        ir152_size,
    ) = state["face_models"]

    img_mid.requires_grad_(True)
    optimizer = torch.optim.Adam([img_mid], lr=lr)

    clean_img = clean_rgb_img.to(device, dtype=torch.float32)
    if clean_img.dim() == 3:
        clean_img = clean_img.unsqueeze(0)
    if clean_img.shape[1] == 1:
        clean_img = clean_img.repeat(1, 3, 1, 1)
    clean_img = clean_img.clamp(0.0, 1.0)

    tgt_image = tgt_image.to(device, dtype=torch.float32)
    if tgt_image.dim() == 3:
        tgt_image = tgt_image.unsqueeze(0)
    if tgt_image.shape[1] == 1:
        tgt_image = tgt_image.repeat(1, 3, 1, 1)
    tgt_image = tgt_image.clamp(0.0, 1.0)

    target_model_name = "mobileface"

    # =========================================================
    # OPT-1：缓存 tgt_image 的 ref embeddings（同一张图全程不变）
    # =========================================================
    cache_key = "ref_face_embeds"
    tgt_ptr = int(tgt_image.data_ptr())
    need_recompute = True
    if cache_key in state:
        c = state[cache_key]
        if c.get("tgt_ptr", None) == tgt_ptr and c.get("device", None) == str(device):
            need_recompute = False

    if need_recompute:
        with torch.no_grad():
            ref_rgb_mobileface = normalize_face_input(tgt_image, mobileface_size)
            ref_rgb_facenet = normalize_face_input(tgt_image, facenet_size)
            ref_rgb_ir152 = normalize_face_input(tgt_image, ir152_size)
            ref_rgb_irse50 = normalize_face_input(tgt_image, irse50_size)

            ref_embedding_mobileface = mobileface_model(ref_rgb_mobileface)
            ref_embedding_facenet = facenet_model(ref_rgb_facenet)
            ref_embedding_ir152 = ir152_model(ref_rgb_ir152)
            ref_embedding_irse50 = irse50_model(ref_rgb_irse50)

        state[cache_key] = {
            "tgt_ptr": tgt_ptr,
            "device": str(device),
            "mobileface": ref_embedding_mobileface.detach(),
            "facenet": ref_embedding_facenet.detach(),
            "ir152": ref_embedding_ir152.detach(),
            "irse50": ref_embedding_irse50.detach(),
        }

    ref_embedding_mobileface = state[cache_key]["mobileface"]
    ref_embedding_facenet = state[cache_key]["facenet"]
    ref_embedding_ir152 = state[cache_key]["ir152"]
    ref_embedding_irse50 = state[cache_key]["irse50"]

    #pred_clean_ref = pred_clean_ref.to(device=device, dtype=img_mid.dtype).detach()

    adv_iter = 0
    while adv_iter < 10:
        batch_x_mid = unpack(img_mid.float(), img_height, img_width)
        try:
            ae_dtype = next(ae.parameters()).dtype
        except StopIteration:
            ae_dtype = torch.float32
        if batch_x_mid.dtype != ae_dtype:
            batch_x_mid = batch_x_mid.to(ae_dtype)

        use_autocast = (device.type == "cuda") and (ae_dtype == torch.bfloat16)
        if use_autocast:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                x_mid = ae.decode(batch_x_mid).to(torch.float32)
        else:
            x_mid = ae.decode(batch_x_mid).to(torch.float32)

        x_mid = ((x_mid.clamp(-1, 1) + 1.0) / 2.0).to(torch.float32)
        x_mid = x_mid.clamp(0.0, 1.0)

        rec_embedding_mobileface = mobileface_model(normalize_face_input(x_mid[:, :3], mobileface_size))
        rec_embedding_facenet = facenet_model(normalize_face_input(x_mid[:, :3], facenet_size))
        rec_embedding_ir152 = ir152_model(normalize_face_input(x_mid[:, :3], ir152_size))
        rec_embedding_irse50 = irse50_model(normalize_face_input(x_mid[:, :3], irse50_size))

        cos_sim_mobileface = F.cosine_similarity(
            ref_embedding_mobileface, rec_embedding_mobileface, dim=1, eps=1e-8
        ).mean()
        cos_sim_facenet = F.cosine_similarity(ref_embedding_facenet, rec_embedding_facenet, dim=1, eps=1e-8).mean()
        cos_sim_ir152 = F.cosine_similarity(ref_embedding_ir152, rec_embedding_ir152, dim=1, eps=1e-8).mean()
        cos_sim_irse50 = F.cosine_similarity(ref_embedding_irse50, rec_embedding_irse50, dim=1, eps=1e-8).mean()

        cosine_similarities = {
            "mobileface": cos_sim_mobileface,
            "facenet": cos_sim_facenet,
            "ir152": cos_sim_ir152,
            "irse50": cos_sim_irse50,
        }
        if target_model_name in cosine_similarities:
            del cosine_similarities[target_model_name]
        L_id = 1-(torch.stack(list(cosine_similarities.values())).mean())
        perc_detail = multi_perc(x_mid, clean_img)
        lp_limit = 0.3
        wid_min, wid_max = 0.05, 1
        lp_val = float(perc_detail.detach().cpu())

        # ------------------- 核心修改：线性 → sigmoid 计算 w_id -------------------
        k = 20.0  # sigmoid灵敏度（推荐默认值，可根据需要调整）
        delta = lp_val - lp_limit  # 失真偏差（和原逻辑一致）
        # sigmoid核心公式：替代原来的线性计算
        w_id = wid_min + (wid_max - wid_min) / (1 + math.exp(k * delta))
        # 保留原有的上下限裁剪逻辑
        w_id = max(wid_min, min(wid_max, w_id))
        # ------------------- 保留你原有的损失计算逻辑 -------------------
        total_loss = (
                w_id * L_id
                + 0.5 * perc_detail
        )
        #print("L_id", float(L_id), "perc", float(perc_detail), "w_id", w_id)


        #total_loss = (0.1 * L_id + 1.5 * lpips_loss)
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        #torch.nn.utils.clip_grad_norm_([img_mid], max_norm=1.0)
        # with torch.no_grad():
        #     if mask_tensor is not None:
        #         E = mask_tensor.to(img_mid.device, img_mid.dtype)  # 编辑区域
        #         if ps_mask_tensor is not None:
        #             # 原转换逻辑
        #             S = ps_mask_tensor.to(img_mid.device, img_mid.dtype)
        #             base = outside_grad_factor
        #             low = 0.6 * base
        #             high = 0.9 * base
        #             non_edit_scale = low + (high - low) * S
        #             grad_scale = E + (1.0 - E) * non_edit_scale
        #         else:
        #             # 没有 PS 图时退回原来的统一 2e-5
        #             grad_scale = E + outside_grad_factor * (1.0 - E)
        #
        #         img_mid.grad *= grad_scale
        #     else:
        #         # 极端情况：没有编辑 mask，只用 PS 控制整脸（一般不会走到）
        #         if ps_mask_tensor is not None:
        #             S = ps_mask_tensor.to(img_mid.device, img_mid.dtype)
        #             base = outside_grad_factor
        #             low = 0.2 * base
        #             high = 1.5 * base
        #             non_edit_scale = low + (high - low) * S
        #             img_mid.grad *= non_edit_scale
                # 否则就不缩放
        optimizer.step()
        adv_iter += 1

    return img_mid.detach().to(dtype=orig_dtype)  # ✅ 回到原 dtype，后续 denoise 继续省显存/一致



def denoise_gen(
    model: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: List[float],
    info: dict,
    guidance: float = 4.0,
    callback: Optional[Callable] = None,
    save_mid_imgs: bool = False,
    ae: Optional[torch.nn.Module] = None,
    img_height: Optional[int] = None,
    img_width: Optional[int] = None,
    device: Optional[torch.device] = None,
    save_root_dir: str = "./output",
    is_main_process: bool = True,
    use_inv_pred_steps: Optional[List[int]] = None,
    tgt_image: Optional[Tensor] = None,
    attack_start_step: int = 15,
    outside_grad_factor: float = 0.95,
    outside_update_factor: float = 0.95,
    init_img_tensor: Optional[Tensor] = None,
    w_pred: float = 1.0,
) -> Tuple[Tensor, dict]:
    assert device is not None, "必须显式传入 device"

    if not info.get("_flux_frozen", False):
        for p in model.parameters():
            p.requires_grad_(False)
        info["_flux_frozen"] = True

    inject_list = [True] * info.get("inject_step", 0) + [False] * (
        len(timesteps[:-1]) - info.get("inject_step", 0)
    )
    dtype = next(model.parameters()).dtype
    guidance_vec = torch.full((img.shape[0],), guidance, device=device, dtype=dtype)
    total_steps = len(timesteps) - 1

    inv_pred_list = info.get("inv_pred", None)
    inv_pred_mid_list = info.get("inv_pred_mid", None)
    if use_inv_pred_steps is not None and (inv_pred_list is None or inv_pred_mid_list is None):
        raise ValueError("use_inv_pred_steps 已设置，但 info 中没有 inv_pred / inv_pred_mid，请先在反演阶段开启记录。")
    use_inv_pred_set = set(use_inv_pred_steps) if use_inv_pred_steps is not None else None

    mask_tensor = info.get("mask_tensor", None)
    if mask_tensor is not None:
        mask_tensor = mask_tensor.to(device=device, dtype=dtype)

    adv_state = info.get("adv_state", None)
    if adv_state is None:
        adv_state = {}
        info["adv_state"] = adv_state
    adv_state["total_steps"] = total_steps
    adv_state["attack_start_step"] = attack_start_step
    adv_state["is_main_process"] = is_main_process

    current_img = img
    current_img_adv = img.clone()

    # =========================================================
    # OPT-3：t_vec / t_vec_mid 预分配 + fill_ 复用
    # =========================================================
    t_vec = torch.empty((img.shape[0],), device=device, dtype=dtype)
    t_vec_mid = torch.empty_like(t_vec)

    for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        dt = t_prev - t_curr

        info["step_idx"] = step_idx
        info["total_steps"] = total_steps
        info["attack_start_step"] = attack_start_step
        info["ae"] = ae
        info["init_img_tensor"] = init_img_tensor
        info["img_height"] = img_height
        info["img_width"] = img_width
        info["device"] = device

        # 替代 torch.full
        t_vec.fill_(t_curr)

        info["t"] = t_curr
        info["inverse"] = False
        info["inject"] = inject_list[step_idx]

        # =========================================================
        # OPT-2：只有 do_adv 时才 decode clean_rgb_img
        # =========================================================
        do_adv = (tgt_image is not None) and (step_idx >= attack_start_step)

        # 1) clean
        with torch.no_grad():
            info["second_order"] = False
            pred_clean, _ = model(
                img=current_img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                info=info,
            )

            pred_clean_ref = pred_clean.detach()
            if use_inv_pred_set is not None and step_idx in use_inv_pred_set and inv_pred_list is not None and 0 <= step_idx < len(inv_pred_list):
                inv_pred = inv_pred_list[step_idx].to(device=device, dtype=dtype)
                if mask_tensor is not None:
                    pred_clean = mask_tensor * pred_clean + (1.0 - mask_tensor) * inv_pred

            img_mid_clean = current_img + 0.5 * dt * pred_clean

            # 替代 torch.full
            t_vec_mid.fill_(t_curr + 0.5 * dt)

            info["second_order"] = True
            pred_mid_clean, _ = model(
                img=img_mid_clean,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec_mid,
                guidance=guidance_vec,
                info=info,
            )

            if use_inv_pred_set is not None and step_idx in use_inv_pred_set and inv_pred_mid_list is not None and 0 <= step_idx < len(inv_pred_mid_list):
                inv_pred_mid = inv_pred_mid_list[step_idx].to(device=device, dtype=dtype)
                if mask_tensor is not None:
                    pred_mid_clean = mask_tensor * pred_mid_clean + (1.0 - mask_tensor) * inv_pred_mid

            first_order_clean = (pred_mid_clean - pred_clean) / (0.5 * dt)
            current_img = current_img + dt * pred_clean + 0.5 * dt * dt * first_order_clean

        clean_rgb_img = None
        if do_adv:
            with torch.no_grad():
                batch_x_clean = unpack(current_img.float(), img_height, img_width)
                try:
                    ae_dtype = next(ae.parameters()).dtype
                except StopIteration:
                    ae_dtype = torch.float32
                if batch_x_clean.dtype != ae_dtype:
                    batch_x_clean = batch_x_clean.to(ae_dtype)
                use_autocast = (device.type == "cuda") and (ae_dtype == torch.bfloat16)
                if use_autocast:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        clean_rgb_img = ae.decode(batch_x_clean)
                else:
                    clean_rgb_img = ae.decode(batch_x_clean)

                clean_rgb_img = ((clean_rgb_img.clamp(-1, 1) + 1.0) / 2.0).to(torch.float32)
                if clean_rgb_img.dim() == 3:
                    clean_rgb_img = clean_rgb_img.unsqueeze(0)

        # 2) adv + PGD
        if tgt_image is not None:
            with torch.no_grad():
                info["second_order"] = False
                pred_adv, _ = model(
                    img=current_img_adv,
                    img_ids=img_ids,
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    info=info,
                )

            if use_inv_pred_set is not None and step_idx in use_inv_pred_set and inv_pred_list is not None and 0 <= step_idx < len(inv_pred_list):
                inv_pred = inv_pred_list[step_idx].to(device=device, dtype=dtype)
                if mask_tensor is not None:
                    pred_adv = mask_tensor * pred_adv + (1.0 - mask_tensor) * inv_pred

            img_mid_adv = current_img_adv + 0.5 * dt * pred_adv

            # 替代 torch.full
            t_vec_mid.fill_(t_curr + 0.5 * dt)

            with torch.no_grad():
                info["second_order"] = True
                pred_mid_adv, _ = model(
                    img=img_mid_adv,
                    img_ids=img_ids,
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec_mid,
                    guidance=guidance_vec,
                    info=info,
                )

            if use_inv_pred_set is not None and step_idx in use_inv_pred_set and inv_pred_mid_list is not None and 0 <= step_idx < len(inv_pred_mid_list):
                inv_pred_mid = inv_pred_mid_list[step_idx].to(device=device, dtype=dtype)
                if mask_tensor is not None:
                    pred_mid_adv = mask_tensor * pred_mid_adv + (1.0 - mask_tensor) * inv_pred_mid

            first_order_adv = (pred_mid_adv - pred_adv) / (0.5 * dt)
            current_img_adv = current_img_adv + dt * pred_adv + 0.5 * dt * dt * first_order_adv

            if do_adv:
                current_img_adv = get_classifier_guidance(img_mid=current_img_adv, clean_latent=current_img,
                                                          clean_rgb_img=clean_rgb_img, tgt_image=tgt_image,
                                                          img_height=img_height, img_width=img_width, device=device,
                                                          step_idx=step_idx, ae=ae, lr=0.02, state=adv_state,
                                                          mask_tensor=mask_tensor,
                                                          ps_mask_tensor=info.get("ps_mask_tensor", None),
                                                          outside_grad_factor=outside_grad_factor,
                                                          outside_update_factor=outside_update_factor, model=model,
                                                          img_ids=img_ids, txt=txt, txt_ids=txt_ids, vec=vec,
                                                          t_vec=t_vec, guidance_vec=guidance_vec, info=info,
                                                          pred_clean_ref=pred_clean_ref, w_pred=w_pred)

        if callback is not None:
            callback(step_idx, total_steps)


    return current_img_adv, info
