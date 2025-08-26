# /ComfyUI/custom_nodes/MangaInpaintToolsInteractive/nodes.py
#
# 真の最終完成版: 条件付き潜在スケーラーにAND/ORの判定ロジック選択機能を追加
# 依存ライブラリ: opencv-python, numpy, torch

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import json


# --- グローバル定数 ---
MAX_PANELS = 16

# --------------------------------------------------------------------
# Node 1: MangaPanelDetector_Ultimate (変更なし)
# --------------------------------------------------------------------
class MangaPanelDetector_Ultimate:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "image": ("IMAGE",), "frame_color_hex": ("STRING", {"default": "#FFFFFF"}), "color_tolerance": ("INT", {"default": 10}), "gap_closing_scale": ("INT", {"default": 5}), "final_line_thickness": ("INT", {"default": 5}), "sort_panels_by": (["top-to-bottom", "left-to-right", "largest-first"],), "min_area": ("INT", {"default": 5000}), }}
    RETURN_TYPES, FUNCTION, CATEGORY = ("MASK", "INT", "IMAGE", "IMAGE"), "detect_panels", "Manga Inpaint"
    RETURN_NAMES = ("mask_batch", "panel_count", "DEBUG_color_mask", "DEBUG_cleaned_frame")
    def hex_to_rgb(self, h): h = h.lstrip('#'); return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    def detect_panels(self, image, frame_color_hex, color_tolerance, gap_closing_scale, final_line_thickness, sort_panels_by, min_area):
        base_img_cv2 = (image[0].cpu().numpy() * 255).astype(np.uint8)
        img_h, img_w = base_img_cv2.shape[0], base_img_cv2.shape[1]
        rgb_tuple = self.hex_to_rgb(frame_color_hex)
        frame_color = np.array(rgb_tuple)
        lower = np.maximum(0, frame_color - color_tolerance).astype(np.uint8)
        upper = np.minimum(255, frame_color + color_tolerance).astype(np.uint8)
        color_mask = cv2.inRange(base_img_cv2, lower, upper)
        color_mask_debug = torch.from_numpy(cv2.cvtColor(color_mask, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0).unsqueeze(0).to(image.device)
        kernel_close = np.ones((gap_closing_scale, gap_closing_scale), np.uint8)
        closed_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_close)
        kernel_dilate = np.ones((final_line_thickness, final_line_thickness), np.uint8)
        final_frame_mask = cv2.dilate(closed_mask, kernel_dilate, iterations=1)
        cleaned_frame_debug = torch.from_numpy(cv2.cvtColor(final_frame_mask, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0).unsqueeze(0).to(image.device)
        inverted_mask = cv2.bitwise_not(final_frame_mask)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_mask, 4, cv2.CV_32S)
        panels_meta = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > min_area: panels_meta.append({'label_index': i, 'box': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]), 'area': stats[i, cv2.CC_STAT_AREA]})
        if not panels_meta: return (torch.zeros((1, img_h, img_w), device=image.device), 0, color_mask_debug, cleaned_frame_debug)
        if sort_panels_by == "largest-first": panels_meta.sort(key=lambda item: item['area'], reverse=True)
        elif sort_panels_by == "top-to-bottom": panels_meta.sort(key=lambda item: (item['box'][1], item['box'][0]))
        else: panels_meta.sort(key=lambda item: (item['box'][0], item['box'][1]))
        mask_list = [torch.from_numpy(np.where(labels == item['label_index'], 255, 0).astype(np.uint8)).to(image.device, dtype=torch.float32).unsqueeze(0) / 255.0 for item in panels_meta[:MAX_PANELS]]
        if not mask_list: return (torch.zeros((1, img_h, img_w), device=image.device), 0, color_mask_debug, cleaned_frame_debug)
        return (torch.cat(mask_list, dim=0), len(mask_list), color_mask_debug, cleaned_frame_debug)

# --------------------------------------------------------------------
# Node 2: CropPanelForInpaint_Advanced (変更なし)
# --------------------------------------------------------------------
class CropPanelForInpaint_Advanced:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "image": ("IMAGE",), "mask_batch": ("MASK",), "panel_index": ("INT", {"default": 1}), "fill_color_hex": ("STRING", {"default": "#000000"}), }}
    RETURN_TYPES, FUNCTION, CATEGORY = ("IMAGE", "MASK"), "crop", "Manga Inpaint"
    RETURN_NAMES = ("cropped_panel", "cropped_mask")
    def hex_to_rgb(self, h): h = h.lstrip('#'); return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    def crop(self, image, mask_batch, panel_index, fill_color_hex):
        index = panel_index - 1
        if image.shape[0] == 0 or mask_batch.shape[0] == 0 or index < 0 or index >= mask_batch.shape[0]: return (torch.zeros((1, 64, 64, 3)), torch.zeros((1, 64, 64)))
        image_tensor, mask = image[0], mask_batch[index]
        coords = torch.nonzero(mask, as_tuple=False)
        if coords.shape[0] == 0: return (torch.zeros((1, 64, 64, 3)), torch.zeros((1, 64, 64)))
        fill_rgb = self.hex_to_rgb(fill_color_hex)
        fill_color = torch.tensor([c / 255.0 for c in fill_rgb], device=image.device, dtype=torch.float32)
        masked_img = torch.where(mask.unsqueeze(-1) > 0.5, image_tensor, fill_color)
        y1, y2, x1, x2 = coords[:, 0].min(), coords[:, 0].max(), coords[:, 1].min(), coords[:, 1].max()
        return (masked_img[y1:y2+1, x1:x2+1, :].unsqueeze(0), mask[y1:y2+1, x1:x2+1].unsqueeze(0))

# --------------------------------------------------------------------
# ★★★ Node 3: ConditionalLatentScaler_Final (判定ロジック選択機能付き) ★★★
# --------------------------------------------------------------------
class ConditionalLatentScaler_Final:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "threshold_pixel_width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "threshold_pixel_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "comparison_logic": (["AND (両方)", "OR (どちらか一方)"],), # ★ 新しい設定
                "scale_factor_if_small": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 8.0, "step": 0.1}),
                "scale_factor_if_large": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1}),
                "upscale_method": (["bicubic", "bilinear", "nearest-exact"],),
            }
        }
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("scaled_samples", "info")
    FUNCTION, CATEGORY = "scale", "Manga Inpaint"
    
    def scale(self, samples, threshold_pixel_width, threshold_pixel_height, comparison_logic,
              scale_factor_if_small, scale_factor_if_large, upscale_method):
        s = samples.copy()
        latent_samples = s["samples"]
        if latent_samples.shape[0] == 0: return (s, "Empty latent input.")
        
        latent_h, latent_w = latent_samples.shape[2], latent_samples.shape[3]
        current_pixel_w, current_pixel_h = latent_w * 8, latent_h * 8
        
        # ★ 新しい判定ロジック
        is_small = False
        if comparison_logic == "AND (両方)":
            if current_pixel_w < threshold_pixel_width and current_pixel_h < threshold_pixel_height:
                is_small = True
        else: # "OR (どちらか一方)"
            if current_pixel_w < threshold_pixel_width or current_pixel_h < threshold_pixel_height:
                is_small = True
        
        scale_factor = scale_factor_if_small if is_small else scale_factor_if_large
        
        if scale_factor == 1.0:
            info = f"Not scaled. Current: {current_pixel_w}x{current_pixel_h}"
            return (s, info)

        new_latent_h, new_latent_w = int(latent_h * scale_factor), int(latent_w * scale_factor)
        s["samples"] = F.interpolate(latent_samples, size=(new_latent_h, new_latent_w), mode=upscale_method)
        
        info = f"Scaled by x{scale_factor:.2f}. From {current_pixel_w}x{current_pixel_h} -> {new_latent_w*8}x{new_latent_h*8}"
        return (s, info)

# --------------------------------------------------------------------
# Node 4: AssembleSinglePanel (変更なし)
# --------------------------------------------------------------------
class AssembleSinglePanel:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "base_image": ("IMAGE",), "generated_panel": ("IMAGE",), "mask_batch": ("MASK",), "panel_index": ("INT", {"default": 1}), }}
    RETURN_TYPES, FUNCTION, CATEGORY = ("IMAGE",), "assemble", "Manga Inpaint"
    def assemble(self, base_image, generated_panel, mask_batch, panel_index):
        canvas_tensor = base_image[0].clone()
        index = panel_index - 1
        if generated_panel.shape[0] == 0 or mask_batch.shape[0] == 0 or index < 0 or index >= mask_batch.shape[0]: return (base_image,)
        image_to_paste, mask = generated_panel[0], mask_batch[index]
        coords = torch.nonzero(mask, as_tuple=False)
        if coords.shape[0] == 0: return (base_image,)
        y1, y2, x1, x2 = coords[:, 0].min(), coords[:, 0].max(), coords[:, 1].min(), coords[:, 1].max()
        h, w = y2 - y1 + 1, x2 - x1 + 1
        if h <= 0 or w <= 0: return (base_image,)
        img_chw = image_to_paste.permute(2, 0, 1).unsqueeze(0)
        resized_chw = F.interpolate(img_chw, size=(h.item(), w.item()), mode='bilinear', align_corners=False)
        resized_image = resized_chw.squeeze(0).permute(1, 2, 0)
        target_region = canvas_tensor[y1:y2+1, x1:x2+1]
        sub_mask = mask[y1:y2+1, x1:x2+1].unsqueeze(-1)
        pasted_region = torch.where(sub_mask > 0.5, resized_image, target_region)
        canvas_tensor[y1:y2+1, x1:x2+1] = pasted_region
        return (canvas_tensor.unsqueeze(0),)

# --------------------------------------------------------------------
# ★★★ Node 0: InteractivePanelCreator (改訂版) ★★★
# UIで描画したコマ割りのレイアウト画像を生成するノード
# --------------------------------------------------------------------
class InteractivePanelCreator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                # JSからのデータを受け取るための隠しウィジェット
                "regions_json": ("STRING", {"multiline": True, "default": "[]", "widget": "hidden"}),
            }
        }

    # 戻り値の型をIMAGEに変更
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "create_layout_image"
    CATEGORY = "Manga Inpaint"

    def create_layout_image(self, width, height, regions_json):
        try:
            regions = json.loads(regions_json)
        except json.JSONDecodeError:
            print("InteractivePanelCreator: Invalid JSON data. Returning empty image.")
            regions = []

        # 黒いキャンバスを作成 (H, W)
        canvas = torch.zeros((height, width), dtype=torch.float32)

        if regions:
            # 各領域をループしてキャンバスに白で描画
            for region in regions:
                x = int(region.get("x", 0))
                y = int(region.get("y", 0))
                w = int(region.get("w", 0))
                h = int(region.get("h", 0))

                if w <= 0 or h <= 0:
                    continue
                
                # 矩形領域を白(1.0)で塗りつぶす
                # 座標が画像の範囲を超えないようにクリッピング
                x_end = min(x + w, width)
                y_end = min(y + h, height)
                canvas[y:y_end, x:x_end] = 1.0

        # ComfyUIのIMAGE形式 (Batch, H, W, Channels) に変換
        # チャンネル次元を追加し、3チャンネルに複製 (R, G, Bがすべて同じ値のグレースケール画像になる)
        image_rgb = canvas.unsqueeze(-1).repeat(1, 1, 3)
        # バッチ次元を追加
        image_batch = image_rgb.unsqueeze(0)

        return (image_batch,)