# auto_mask.py（完整修正版：眼影严格基于BiSeNet原生眼睛区域生成）

import os
import re
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from sentence_transformers import SentenceTransformer, util
from model import BiSeNet  # 需确保BiSeNet模型文件在对应路径

# landmark 库
try:
    import face_alignment
except ImportError:
    face_alignment = None

# -------------------------- 配置路径 --------------------------
_BISENET_CKPT = "/root/AAAAADV/MASQUE-main/79999_iter.pth"  # 你的BiSeNet权重路径
LOCAL_SBERT_PATH = "/share/newpro/all-mpnet-base-v2"  # 你的SBERT模型路径

# -------------------------- BiSeNet 19 类映射 --------------------------
_grouped = {
    "background": [0],
    "skin": [1],
    "eyebrows": [2, 3],
    "eyes": [4, 5],
    "eye_glasses": [6],
    "ears": [7, 8, 9],
    "nose": [10],
    "mouth": [11],
    "lips": [12, 13],
    "neck": [14, 15],
    "cloth": [16],
    "hair": [17],
    "hat": [18],
}

# -------------------------- 18 个编辑区域语义定义 --------------------------
EDIT_REGION_DESC = {
    "beard": "the beard and moustache area on the lower face around chin and jawline",
    "face_shape": "the overall outline of the face including cheeks, chin and jawline but excluding hair",
    "chin": "the lowest part of the face below the lower lip and between the jaw corners",
    "skin": "all facial skin including cheeks, forehead and chin but excluding hair, eyes, eyebrows, lips and nose",
    "hair": "the hair on the head including bangs and sideburns",
    "hairline": "the boundary line where the hair meets the forehead skin",
    "forehead": "the forehead area between eyebrows and hairline",
    "bangs": "the hair that falls over the forehead (刘海，前额的头发部分)",
    "mouth": "the mouth opening area around and inside the lips",
    "lips": "the upper and lower lips only（仅上下嘴唇，不包含嘴巴内部）",
    "nose": "the nose including bridge and nostrils",
    "eyebrows": "the strips of hair above the eyes, left and right eyebrows",
    "eyeshadow": "the eyelid area including upper and lower eyelids (primary area for eyeshadow application)",
    "eye_pupils": "the inner eyeball region including iris and pupil",
    "eyes": "the eye region including eyeballs, eyelashes, eyelids and eyeliner",
    "makeup": "all typical facial makeup regions such as eyeshadow, blush and lipstick",
    "cheeks": "the cheek areas on both sides of the nose",
    "expression": "expression related regions including eyebrows, eyes and mouth",
    "eyeglasses": "eyeglasses worn on the face including frames and lenses",
}

EDIT_REGIONS = list(EDIT_REGION_DESC.keys())
EDIT_TEXTS = list(EDIT_REGION_DESC.values())

# ================== 感知敏感度 ==================
PS_VALUE = {
    "eyebrows": 0.6,
    "eyes": 0.75,
    "nose": 0.65,
    "mouth": 0.8,
    "skin": 0.9,
}
_ps_max = max(PS_VALUE.values())
_ps_min = min(PS_VALUE.values())


def ps_to_weight(v: float, w_min: float = 0.05, w_max: float = 0.25) -> float:
    ratio = (_ps_max - v) / (_ps_max - _ps_min + 1e-8)
    return w_min + (w_max - w_min) * ratio


REGION_PS_WEIGHT = {k: ps_to_weight(v) for k, v in PS_VALUE.items()}

# -------------------------- 18区域→BiSeNet语义组合 --------------------------
EDIT_REGION_TO_SEG_GROUPS = {
    "beard": ["skin"],
    "face_shape": ["skin"],
    "chin": ["skin"],
    "skin": ["skin"],
    "hair": ["hair"],
    "hairline": ["hair"],
    "forehead": ["skin"],
    "bangs": ["hair", "forehead"],
    "mouth": ["mouth", "lips"],
    "lips": ["lips"],
    "nose": ["nose"],
    "eyebrows": ["eyebrows"],
    "eyeshadow": ["skin"],
    "eye_pupils": ["eyes"],
    "eyes": ["eyes"],
    "makeup": ["eyes", "lips", "skin"],
    "cheeks": ["skin"],
    "expression": ["eyebrows", "eyes", "mouth"],
    "eyeglasses": ["eye_glasses"],
}

# -------------------------- 全局模型/变换 --------------------------
_sbert = None
_edit_text_emb = None
_bisenet = None
_fa = None

_to_tensor = T.Compose([
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# -------------------------- 关键词→区域映射 --------------------------
_makeup_keywords = {
    r"\beyes\b": "eyes",
    r"\beye\b": "eyes",
    r"\beyeshadow\b": "eyeshadow",
    r"\beye shadow\b": "eyeshadow",
    r"\beyeliner\b": "eyes",
    r"\beye liner\b": "eyes",
    r"\bmascara\b": "eyes",
    r"\beye makeup\b": "eyes",
    r"\beye color\b": "eyes",
    r"\beye shape\b": "eyes",
    r"\bmakeup\b": "makeup",
    r"\blipstick\b": "lips",
    r"\blip gloss\b": "lips",
    r"\blips\b": "lips",
    r"\blip\b": "lips",
    r"\bblush\b": "cheeks",
    r"\brouge\b": "cheeks",
    r"\bbeard\b": "beard",
    r"\bmustache\b": "beard",
    r"\bforehead\b": "forehead",
    r"\bhairline\b": "hairline",
    r"\bhair\b": "hair",
    r"\bbangs\b": "bangs",
    r"\bbang\b": "bangs",
    r"\bfringe\b": "bangs",
    r"\beyebrow\b": "eyebrows",
    r"\beyebrows\b": "eyebrows",
    r"\bchin\b": "chin",
    r"\bnose\b": "nose",
    r"\bglasses\b": "eyeglasses",
    r"\beyeglasses\b": "eyeglasses",
    r"\bskin\b": "skin",
    r"\bmouth\b": "mouth",
}


# -------------------------- 释放模型GPU占用 --------------------------
def release_model_gpu():
    global _bisenet, _sbert, _fa, _edit_text_emb
    if _bisenet is not None:
        _bisenet = _bisenet.cpu()
        del _bisenet
        _bisenet = None
        print("[MASK] BiSeNet 模型已释放")

    if _sbert is not None:
        _sbert = _sbert.cpu()
        del _sbert
        _sbert = None
        _edit_text_emb = None

    if _fa is not None:
        del _fa
        _fa = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -------------------------- 初始化模型 --------------------------
def _ensure_text_model():
    global _sbert, _edit_text_emb
    if _sbert is None:
        _sbert = SentenceTransformer(LOCAL_SBERT_PATH)
        _edit_text_emb = _sbert.encode(EDIT_TEXTS, convert_to_tensor=True)


def _ensure_seg_model(device: str = "cuda"):
    global _bisenet
    if _bisenet is None:
        _bisenet = BiSeNet(n_classes=19)
        _bisenet.to(device)
        _bisenet.load_state_dict(torch.load(_BISENET_CKPT, map_location=device))
        _bisenet.eval()


def _ensure_landmark_model(device: str = "cuda"):
    global _fa
    if _fa is None:
        if face_alignment is None:
            raise ImportError("未安装 face-alignment，请先执行 pip install face-alignment")
        _fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=device,
            flip_input=False
        )


# -------------------------- landmark 获取 --------------------------
def get_landmarks_68(img_rgb: np.ndarray, device: str = "cuda") -> np.ndarray:
    _ensure_landmark_model(device=device)
    preds = _fa.get_landmarks(img_rgb)
    if preds is None or len(preds) == 0:
        raise RuntimeError("[LDMK] 未检测到人脸")
    return preds[0]


# -------------------------- Prompt→区域映射 --------------------------
def regions_from_prompt(prompt: str, similarity_threshold: float = 0.35):
    _ensure_text_model()
    p = prompt.lower().strip()
    if not p:
        print("[MASK] 空 prompt，不定位任何区域")
        return []

    matched_regions = set()
    keyword_hit_regions = set()

    for pattern, region in _makeup_keywords.items():
        if re.search(pattern, p):
            matched_regions.add(region)
            keyword_hit_regions.add(region)

    if keyword_hit_regions:
        regions = sorted(keyword_hit_regions)
        return regions

    prompt_emb = _sbert.encode(p, convert_to_tensor=True)
    sims = util.cos_sim(prompt_emb, _edit_text_emb).squeeze().cpu().numpy()

    for region, sim in zip(EDIT_REGIONS, sims):
        print(f"  {region:18s}: {sim:.3f}")

    for region, sim in zip(EDIT_REGIONS, sims):
        th = similarity_threshold
        if region == "eyes":
            th -= 0.08
        elif region == "eyebrows":
            th += 0.15
        elif region == "bangs":
            th -= 0.05
        elif region == "lips":
            th -= 0.03
        elif region in ["makeup", "expression", "eyeglasses"]:
            th += 0.10

        if sim >= th:
            matched_regions.add(region)

    if not matched_regions:
        return []

    if "lips" in matched_regions and "mouth" in matched_regions:
        matched_regions.discard("mouth")

    if "bangs" in matched_regions:
        for r in ["hair", "forehead"]:
            if r in matched_regions and r != "bangs":
                matched_regions.discard(r)

    if "eyes" in matched_regions:
        for r in ["makeup", "eyeshadow", "eyeglasses"]:
            if r in matched_regions and r != "eyes":
                matched_regions.discard(r)

    priority = {
        "eyeshadow": 4,
        "eyes": 4,
        "lips": 4,
        "bangs": 3,
        "eye_pupils": 3,
        "eyeglasses": 2,
        "makeup": 2,
        "beard": 2,
        "mouth": 1,
        "hair": 1,
        "forehead": 1,
        "eyebrows": 0,
    }
    regions = sorted(
        list(matched_regions),
        key=lambda r: priority.get(r, 1),
        reverse=True,
    )
    return regions


# -------------------------- 兼容接口 --------------------------
def class_ids_from_prompt(prompt: str, similarity_threshold: float = 0.35):
    regions = regions_from_prompt(prompt, similarity_threshold=similarity_threshold)
    seg_groups = set()
    for r in regions:
        gs = EDIT_REGION_TO_SEG_GROUPS.get(r, [])
        seg_groups.update(gs)
    class_ids = set()
    for g in seg_groups:
        for cid in _grouped.get(g, []):
            class_ids.add(cid)
    cids = sorted(class_ids)
    return cids


# -------------------------- 分割&基础Mask --------------------------
def segment_image(img_path: str, device: str = "cuda"):
    _ensure_seg_model(device=device)
    img_pil = Image.open(img_path).convert("RGB").resize((512, 512), Image.BILINEAR)
    img_np = np.array(img_pil)

    try:
        landmarks = get_landmarks_68(img_np, device=device)
    except Exception as e:
        landmarks = None

    ten = _to_tensor(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = _bisenet(ten)[0]
        parsing = out.squeeze(0).detach().cpu().numpy().argmax(0)

    eyebrows_mask = np.isin(parsing, [2, 3])
    eyes_mask = np.isin(parsing, [4, 5])
    overlap = eyebrows_mask & eyes_mask
    if np.any(overlap):
        parsing[overlap] = 4

    return parsing, landmarks


def _build_eye_mask_from_landmark(landmarks_512: np.ndarray, H: int, W: int) -> np.ndarray:
    if landmarks_512 is None:
        return np.zeros((H, W), dtype=bool)
    left_eye_pts = landmarks_512[36:42].astype(np.int32)
    right_eye_pts = landmarks_512[42:48].astype(np.int32)

    eye_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(eye_mask, left_eye_pts, 1)
    cv2.fillConvexPoly(eye_mask, right_eye_pts, 1)

    eye_mask = cv2.dilate(eye_mask, np.ones((8, 8), np.uint8)) > 0
    return eye_mask


def build_basic_masks(parsing: np.ndarray):
    masks = {}
    masks["skin"] = (parsing == 1)
    masks["eyebrows"] = np.isin(parsing, [2, 3])
    masks["eyes"] = np.isin(parsing, [4, 5])
    masks["glasses"] = (parsing == 6)
    masks["nose"] = (parsing == 10)
    masks["mouth"] = (parsing == 11)
    masks["lips"] = np.isin(parsing, [12, 13])
    masks["neck"] = np.isin(parsing, [14, 15])
    masks["cloth"] = (parsing == 16)
    masks["hair"] = (parsing == 17)
    masks["hat"] = (parsing == 18)
    masks["background"] = (parsing == 0)
    masks["ears"] = np.isin(parsing, [7, 8, 9])
    return masks


def build_ps_weight_map(parsing: np.ndarray, constructed_eyes_mask: np.ndarray = None) -> np.ndarray:
    basic = build_basic_masks(parsing)
    H, W = parsing.shape
    ps_map = np.zeros((H, W), dtype=np.float32)

    if constructed_eyes_mask is not None:
        if constructed_eyes_mask.shape != (H, W):
            constructed_eyes_mask = cv2.resize(
                constructed_eyes_mask.astype(np.uint8),
                (W, H),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        basic["eyes"] = constructed_eyes_mask

    region_bool = {
        "eyebrows": basic["eyebrows"],
        "eyes": basic["eyes"],
        "nose": basic["nose"],
        "mouth": basic["mouth"] | basic["lips"],
        "skin": basic["skin"],
    }

    for name, mask in region_bool.items():
        if not mask.any():
            continue
        w = REGION_PS_WEIGHT[name]
        ps_map[mask] = w

    return ps_map


# -------------------------- 掩码膨胀工具函数 & 配置参数 --------------------------
NOSE_DILATE_K = (6, 6)
LIPS_DILATE_K = (6, 6)
MOUTH_DILATE_K = (6, 6)


def dilate_mask(mask: np.ndarray, kernel_size: tuple) -> np.ndarray:
    if not mask.any():
        return mask
    kernel = np.ones(kernel_size, np.uint8)
    mask_uint8 = mask.astype(np.uint8)
    mask_dilated_uint8 = cv2.dilate(mask_uint8, kernel)
    return mask_dilated_uint8 > 0


def build_eyeshadow_from_eye_mask(
        mask_eyes: np.ndarray,
        mask_skin: np.ndarray,
        inner_k: int = 3,
        outer_k: int = 11,
) -> np.ndarray:
    """
    严格沿 eye 边缘生成 eyeshadow：
    eyeshadow = 外扩 eye - 内缩 eye + 多层膨胀
    """
    if not mask_eyes.any():
        return np.zeros_like(mask_eyes, dtype=bool)

    eye_uint = mask_eyes.astype(np.uint8)

    # 内缩（最小化内缩，仅排除眼球中心）：inner_k=1（最小可能值）
    inner = cv2.dilate(
        eye_uint,
        np.ones((inner_k, inner_k), np.uint8)
    )

    # 外扩（大幅增大外扩范围）：outer_k=25（比之前的15进一步扩大）
    outer = cv2.dilate(
        eye_uint,
        np.ones((outer_k, outer_k), np.uint8)
    )

    # 基础眼影区域：外扩 - 内缩
    eyeshadow = (outer > 0) & (~inner.astype(bool))
    eyeshadow &= mask_skin  # 强制限制在皮肤区域，避免溢出

    # 第一次膨胀：用5x5核扩大整体范围（核心扩大步骤）
    kernel1 = np.ones((5, 5), np.uint8)
    eyeshadow = cv2.dilate(eyeshadow.astype(np.uint8), kernel1) > 0

    # 第二次膨胀：用3x3核平滑边缘，同时进一步扩大
    kernel2 = np.ones((3, 3), np.uint8)
    eyeshadow = cv2.dilate(eyeshadow.astype(np.uint8), kernel2) > 0

    # 最终再与皮肤掩码做一次交集，确保没有溢出
    eyeshadow &= mask_skin

    return np.array(eyeshadow, dtype=bool)

# ========== 2. 构造眼影 (Strictly BiSeNet Eyes) ==========
# 传入激进的参数，最大化眼影范围



# -------------------------- 18区域Mask构造 --------------------------
def build_18_region_masks(parsing: np.ndarray, landmarks_512: np.ndarray | None):
    """
    构造18个编辑区域的掩码
    核心逻辑：
    1. Eyeshadow: 严格基于 BiSeNet 检测到的眼睛区域 (basic['eyes']) 生成，不参考 Landmark 或膨胀后的眼睛。
    2. Eyes: 最终输出的 eyes 掩码依然是 (BiSeNet | Landmark) 经过大幅膨胀后的结果。
    3. Nose/Lips/Mouth: 均添加膨胀处理，且 lips 排除 mouth。
    """
    H, W = parsing.shape
    yy, xx = np.mgrid[0:H, 0:W]
    basic = build_basic_masks(parsing)

    def to_u8(m: np.ndarray) -> np.ndarray:
        return (m.astype(np.uint8) * 255)

    # ========== 1. 基础区域初始化 ==========
    mask_skin = basic["skin"]
    mask_eyebrows = basic["eyebrows"]
    mask_lips = basic["lips"]
    mask_mouth = basic["mouth"]
    mask_nose = basic["nose"]
    mask_hair = basic["hair"]
    mask_glasses = basic["glasses"].copy()

    # ========== 2. 构造眼影 (Strictly BiSeNet Eyes) ==========
    # 核心修改：只使用 basic['eyes'] (BiSeNet 原生结果)
    mask_eyes_bisenet_only = basic["eyes"].copy()

    mask_eyeshadow = build_eyeshadow_from_eye_mask(
        mask_eyes=mask_eyes_bisenet_only,  # 仅基于 BiSeNet 原生眼睛区域
        mask_skin=mask_skin,
        inner_k=1,  # 最小内缩，仅排除眼球中心
        outer_k=18,  # 大幅外扩，覆盖更广区域
    )

    # ========== 3. 构造“最终”眼睛区域 (Dilated Eyes) ==========
    # 包含 BiSeNet 和 Landmark 的结果，用于最终输出和眼镜遮挡判断
    mask_eyes_landmark = np.zeros_like(mask_eyes_bisenet_only)
    if landmarks_512 is not None:
        mask_eyes_landmark = _build_eye_mask_from_landmark(landmarks_512, H, W)

    # 合并
    mask_eyes_base = mask_eyes_bisenet_only | mask_eyes_landmark

    # 强力扩张
    EYE_DILATE_K1 = 12
    EYE_DILATE_K2 = 12
    mask_eyes_final = mask_eyes_base.copy()

    if mask_eyes_final.any():
        mask_eyes_final = cv2.dilate(
            mask_eyes_final.astype(np.uint8),
            np.ones((EYE_DILATE_K1, EYE_DILATE_K1), np.uint8)
        )
        mask_eyes_final = cv2.dilate(
            mask_eyes_final,
            np.ones((EYE_DILATE_K2, EYE_DILATE_K2), np.uint8)
        ) > 0

    if not mask_eyes_final.any() and landmarks_512 is None:
        mask_eyes_final = (
                cv2.dilate(basic["eyebrows"].astype(np.uint8),
                           np.ones((15, 15), np.uint8)) > 0
        )

    # ========== 4. 鼻子/嘴唇/嘴巴 膨胀处理 ==========
    mask_nose = dilate_mask(mask_nose, NOSE_DILATE_K)
    mask_mouth = dilate_mask(mask_mouth, MOUTH_DILATE_K)
    mask_lips = dilate_mask(mask_lips, LIPS_DILATE_K)
    mask_lips = mask_lips & (~mask_mouth)  # lips 剔除 mouth 内部

    if landmarks_512 is None:
        mask_mouth = mask_mouth | mask_lips

    # ========== 5. 分模式构造其他掩码 ==========
    if landmarks_512 is None:
        # --- 无 Landmark 模式 ---
        if not mask_glasses.any() and mask_eyes_final.any():
            eyes_uint = mask_eyes_final.astype(np.uint8)
            dil = cv2.dilate(eyes_uint, np.ones((19, 19), np.uint8)) > 0
            border = cv2.morphologyEx(dil.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
            ring = border & (~mask_eyes_final)
            mask_glasses = ring & (mask_skin | mask_eyes_final | mask_eyebrows)

        mask_face_shape = (mask_skin & (~mask_hair) & (~mask_eyes_final) & (~mask_eyebrows)
                           & (~mask_lips) & (~mask_mouth) & (~mask_nose))

        lower_band = yy > (H * 0.6)
        mask_beard = mask_face_shape & lower_band
        mask_forehead = mask_skin & (yy < H * 0.35)
        mask_bangs = mask_hair & mask_forehead

        masks = {
            "beard": to_u8(mask_beard),
            "face_shape": to_u8(mask_face_shape),
            "chin": to_u8(mask_face_shape & lower_band),
            "skin": to_u8(mask_skin),
            "hair": to_u8(mask_hair),
            "hairline": to_u8(mask_hair),
            "forehead": to_u8(mask_forehead),
            "bangs": to_u8(mask_bangs),
            "mouth": to_u8(mask_mouth),
            "lips": to_u8(mask_lips),
            "nose": to_u8(mask_nose),
            "eyebrows": to_u8(mask_eyebrows),
            "eyeshadow": to_u8(mask_eyeshadow),  # 使用 BiSeNet Eyes 生成
            "eye_pupils": to_u8(mask_eyes_final),
            "eyes": to_u8(mask_eyes_final),  # 使用 Dilated Eyes
            "makeup": to_u8(mask_eyes_final | mask_lips | (mask_skin & (~mask_nose) & (~(yy < H * 0.4)))),
            "cheeks": to_u8(mask_skin & (~mask_nose) & (~(yy < H * 0.4))),
            "expression": to_u8(mask_eyebrows | mask_eyes_final | mask_mouth),
            "eyeglasses": mask_glasses,
        }

    else:
        # --- 有 Landmark 模式 ---
        lm = landmarks_512.astype(np.float32)

        if not mask_glasses.any():
            eye_pts = lm[36:48]
            ex_min, ex_max = eye_pts[:, 0].min(), eye_pts[:, 0].max()
            ey_min, ey_max = eye_pts[:, 1].min(), eye_pts[:, 1].max()
            eye_w = ex_max - ex_min
            eye_h = ey_max - ey_min

            outer_x_margin = 1.5 * eye_w
            outer_top_margin = 1.2 * eye_h
            outer_bottom_margin = 1.6 * eye_h

            x0 = int(max(ex_min - outer_x_margin, 0))
            x1 = int(min(ex_max + outer_x_margin, W))
            y0 = int(max(ey_min - outer_top_margin, 0))
            y1 = int(min(ey_max + outer_bottom_margin, H))

            outer_band = np.zeros_like(parsing, dtype=bool)
            outer_band[y0:y1, x0:x1] = True

            inner_core = mask_eyes_final.copy()
            if inner_core.any():
                inner_core = cv2.dilate(inner_core.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

            mask_glasses = outer_band & (~inner_core)
            mask_glasses &= (mask_skin | mask_eyes_final | mask_eyebrows)

        def _eye_inner_box(indices):
            pts = lm[indices]
            x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
            y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
            cx = 0.5 * (x_min + x_max)
            cy = 0.5 * (y_min + y_max)
            w, h = (x_max - x_min) * 0.4, (y_max - y_min) * 0.4
            x0, x1 = int(cx - w / 2), int(cx + w / 2)
            y0, y1 = int(cy - h / 2), int(cy + h / 2)
            box = np.zeros_like(parsing, dtype=bool)
            box[max(y0, 0):min(y1, H), max(x0, 0):min(x1, W)] = True
            return box

        pupil_L = _eye_inner_box(range(36, 42))
        pupil_R = _eye_inner_box(range(42, 48))
        mask_eye_pupils = (pupil_L | pupil_R) & mask_eyes_final

        if basic["eyebrows"].any():
            mask_eyebrows = cv2.dilate(mask_eyebrows.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

        mouth_pts = lm[48:60].astype(np.int32)
        mouth_poly = cv2.fillConvexPoly(np.zeros_like(parsing, dtype=np.uint8), mouth_pts, 1).astype(bool)
        mask_mouth = mask_mouth | mouth_poly
        mask_mouth = dilate_mask(mask_mouth, MOUTH_DILATE_K)

        mask_face_shape = mask_skin.copy()
        for m in [basic["hair"], basic["hat"], basic["ears"], basic["neck"], basic["cloth"],
                  mask_eyebrows, mask_eyes_final, mask_eyeshadow, mask_lips, mask_mouth, mask_nose]:
            mask_face_shape[m] = False

        y_mouth_center = lm[48:60, 1].mean()
        y_chin = lm[8, 1]
        margin = 0.1 * (y_chin - y_mouth_center)
        x_jaw_left, x_jaw_right = lm[4, 0], lm[12, 0]
        mask_chin = (mask_skin & (yy >= y_mouth_center) & (yy <= y_chin + margin)
                     & (xx >= x_jaw_left) & (xx <= x_jaw_right))

        mask_beard_chin = cv2.dilate(mask_chin.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        mask_beard_chin &= mask_skin
        nose_base_y = lm[31:36, 1].max()
        upper_lip_y = lm[50:54, 1].min()
        mouth_left_x, mouth_right_x = lm[48, 0], lm[54, 0]
        x0, x1 = int(max(mouth_left_x - 5, 0)), int(min(mouth_right_x + 5, W))
        y0, y1 = int(max(nose_base_y, 0)), int(min(upper_lip_y + 3, H))
        moustache_band = np.zeros_like(parsing, dtype=bool)
        if y1 > y0 and x1 > x0:
            moustache_band[y0:y1, x0:x1] = True
        moustache_band &= mask_skin & (~mask_nose) & (~mask_lips)
        mask_beard = mask_beard_chin | moustache_band

        brow_pts = lm[17:27]
        y_brow_top = brow_pts[:, 1].min()
        x_brow_left, x_brow_right = brow_pts[:, 0].min(), brow_pts[:, 0].max()
        mask_forehead = (mask_skin & (yy < y_brow_top) & (xx >= x_brow_left) & (xx <= x_brow_right))
        mask_forehead[mask_hair] = False

        mask_bangs = mask_hair & (yy < y_brow_top) & (xx >= x_brow_left) & (xx <= x_brow_right)
        if mask_bangs.any():
            mask_bangs = cv2.dilate(mask_bangs.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

        edges = cv2.morphologyEx(mask_hair.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        forehead_dil = cv2.dilate(mask_forehead.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        mask_hairline = edges & forehead_dil

        x_nose_center = lm[27:36, 0].mean()
        y_top_cheek = lm[29, 1]
        y_bot_cheek = lm[33, 1] + (lm[57, 1] - lm[33, 1]) * 0.8
        x_jaw_left2, x_jaw_right2 = lm[2, 0], lm[14, 0]
        left_box = ((xx > x_jaw_left2) & (xx < x_nose_center) & (yy >= y_top_cheek) & (yy <= y_bot_cheek))
        right_box = ((xx < x_jaw_right2) & (xx > x_nose_center) & (yy >= y_top_cheek) & (yy <= y_bot_cheek))
        mask_cheeks = (left_box | right_box) & mask_skin & (~mask_nose) & (~mask_mouth)

        mask_makeup = mask_eyeshadow | mask_lips | mask_cheeks
        mask_expression = mask_eyebrows | mask_eyes_final | mask_mouth

        masks = {
            "beard": to_u8(mask_beard),
            "face_shape": to_u8(mask_face_shape),
            "chin": to_u8(mask_chin),
            "hair": to_u8(mask_hair),
            "hairline": to_u8(mask_hairline),
            "forehead": to_u8(mask_forehead),
            "bangs": to_u8(mask_bangs),
            "mouth": to_u8(mask_mouth),
            "lips": to_u8(mask_lips),
            "nose": to_u8(mask_nose),
            "eyebrows": to_u8(mask_eyebrows),
            "eyeshadow": to_u8(mask_eyeshadow),  # 使用 BiSeNet Eyes 生成
            "eye_pupils": to_u8(mask_eye_pupils),
            "eyes": to_u8(mask_eyes_final),  # 使用 Dilated Eyes
            "makeup": to_u8(mask_makeup),
            "cheeks": to_u8(mask_cheeks),
            "expression": to_u8(mask_expression),
            "eyeglasses": mask_glasses,
        }

    all_eyes_mask = mask_eyes_final | basic["eyebrows"]
    glasses_bool = mask_glasses.astype(bool)
    mask_glasses_bool = glasses_bool & (~all_eyes_mask)
    masks["eyeglasses"] = to_u8(mask_glasses_bool)

    return masks


def make_mask(parsing: np.ndarray, class_ids):
    if not class_ids:
        return np.zeros_like(parsing, dtype=np.uint8)
    mask = np.zeros_like(parsing, dtype=np.uint8)
    for cid in class_ids:
        mask[parsing == cid] = 255
    return mask


def generate_mask(prompt: str, img_path: str, save_path: str,
                  device: str = "cuda", similarity_threshold: float = 0.35,
                  use_18_regions: bool = True):
    parsing, landmarks = segment_image(img_path, device=device)

    region_masks = build_18_region_masks(parsing, landmarks)

    constructed_eyes_mask = (region_masks["eyes"] > 0).astype(bool)

    ps_weight_map = build_ps_weight_map(parsing, constructed_eyes_mask=constructed_eyes_mask)

    is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0
    save_mask = False
    if save_path is not None and is_main_process:
        save_mask = True
    else:
        save_path = None

    if save_mask:
        mask_dir = os.path.dirname(save_path)
        os.makedirs(mask_dir, exist_ok=True)
        parsing_vis_path = os.path.join(mask_dir, f"parsing_vis_{os.path.basename(save_path)}")
        parsing_vis = (parsing * (255 / 18)).astype(np.uint8)
        cv2.imwrite(parsing_vis_path, parsing_vis)

    if use_18_regions:
        regions = regions_from_prompt(prompt, similarity_threshold=similarity_threshold)
        if not regions:
            mask = np.zeros_like(parsing, dtype=np.uint8)
        else:
            mask_bool = np.zeros_like(parsing, dtype=bool)
            for r in regions:
                if r not in region_masks:
                    continue
                mask_bool |= (region_masks[r] > 0)
            mask = (mask_bool.astype(np.uint8) * 255)
    else:
        cids = class_ids_from_prompt(prompt, similarity_threshold=similarity_threshold)
        mask = make_mask(parsing, cids)

    if save_mask:
        cv2.imwrite(save_path, mask)
        valid_ratio = (mask.mean() / 255.0) * 100.0
    else:
        valid_ratio = (mask.mean() / 255.0) * 100.0
        print(f"[MASK] 非主进程（rank {os.environ.get('LOCAL_RANK', 0)}），跳过掩码保存（有效区域占比: {valid_ratio:.2f}%）")

    release_model_gpu()
    return save_path, mask, valid_ratio, ps_weight_map


# -------------------------- 掩码适配 --------------------------
def adapt_mask_to_latent(mask: np.ndarray, img_height: int, img_width: int,
                         device: str, model_dtype: torch.dtype):
    h_blocks = img_height // 16
    w_blocks = img_width // 16
    total_blocks = h_blocks * w_blocks

    if mask.shape[0] != img_height or mask.shape[1] != img_width:
        raise ValueError(f"掩码尺寸（{mask.shape[0]}x{mask.shape[1]}）与图像尺寸（{img_height}x{img_width}）不匹配！")

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.0, sigmaY=1.0)
    mask_f = mask.astype(np.float32) / 255.0
    mask_latent = cv2.resize(mask_f, (w_blocks, h_blocks), interpolation=cv2.INTER_AREA)

    mask_tensor = torch.from_numpy(mask_latent).view(1, total_blocks, 1)
    mask_tensor = mask_tensor.to(device, dtype=model_dtype)
    return mask_tensor