# /ComfyUI/custom_nodes/MangaInpaintToolsInteractive/nodes.py
#
# 真の最終完成版: ノイズ除去処理をMORPH_CLOSEに刷新し、パラメータの挙動を直感的に
# 依存ライブラリ: opencv-python, numpy, torch

import torch
import numpy as np
import cv2

# --- グローバル定数 ---
MAX_PANELS = 16

# --------------------------------------------------------------------
# Node 1: MangaPanelDetector_Ultimate (真の最終版)
# --------------------------------------------------------------------
class MangaPanelDetector_Ultimate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "frame_color_hex": ("STRING", {"default": "#FFFFFF"}),
                "color_tolerance": ("INT", {"default": 10, "min": 0, "max": 255}),
                "gap_closing_scale": ("INT", {"default": 5, "min": 1, "max": 100}), # ★ パラメータ名を変更
                "final_line_thickness": ("INT", {"default": 5, "min": 1, "max": 50}),# ★ パラメータ名を変更
                "sort_panels_by": (["top-to-bottom", "left-to-right", "largest-first"],),
                "min_area": ("INT", {"default": 5000, "min": 0, "max": 999999}),
            }
        }

    RETURN_TYPES = ("MASK", "INT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("mask_batch", "panel_count", "DEBUG_color_mask", "DEBUG_cleaned_frame")
    FUNCTION, CATEGORY = "detect_panels", "Manga Inpaint"

    def hex_to_rgb(self, h): h = h.lstrip('#'); return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def detect_panels(self, image, frame_color_hex, color_tolerance, 
                      gap_closing_scale, final_line_thickness, sort_panels_by, min_area):
        
        base_image_tensor = image[0]
        img_h, img_w = base_image_tensor.shape[0], base_image_tensor.shape[1]
        base_img_cv2 = (base_image_tensor.cpu().numpy() * 255).astype(np.uint8)

        # Step 1: 手動指定された色でマスクを作成
        rgb_tuple = self.hex_to_rgb(frame_color_hex)
        frame_color = np.array(rgb_tuple)
        lower = np.maximum(0, frame_color - color_tolerance).astype(np.uint8)
        upper = np.minimum(255, frame_color + color_tolerance).astype(np.uint8)
        color_mask = cv2.inRange(base_img_cv2, lower, upper)
        color_mask_debug = torch.from_numpy(cv2.cvtColor(color_mask, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0).unsqueeze(0).to(image.device)

        # ★★★ 修正点: 処理をMORPH_CLOSEに変更 ★★★
        # Step 2: 枠線の隙間を埋める
        kernel_close = np.ones((gap_closing_scale, gap_closing_scale), np.uint8)
        closed_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_close)

        # Step 3: 最終的な枠線の太さを確保
        kernel_dilate = np.ones((final_line_thickness, final_line_thickness), np.uint8)
        final_frame_mask = cv2.dilate(closed_mask, kernel_dilate, iterations=1)
        cleaned_frame_debug = torch.from_numpy(cv2.cvtColor(final_frame_mask, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0).unsqueeze(0).to(image.device)
        
        # Step 4: 連結成分ラベリングでコマ領域を判定 (4方向連結)
        inverted_mask = cv2.bitwise_not(final_frame_mask)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_mask, 4, cv2.CV_32S)

        panels_meta = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_area: panels_meta.append({'label_index': i, 'box': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]), 'area': area})
        
        if not panels_meta: return (torch.zeros((1, img_h, img_w), device=image.device), 0, color_mask_debug, cleaned_frame_debug)

        # Step 5: 検出したパネルをソート
        if sort_panels_by == "largest-first": panels_meta.sort(key=lambda item: item['area'], reverse=True)
        elif sort_panels_by == "top-to-bottom": panels_meta.sort(key=lambda item: (item['box'][1], item['box'][0]))
        else: panels_meta.sort(key=lambda item: (item['box'][0], item['box'][1]))

        # Step 6: 最終的なマスクを生成
        mask_list = []
        for item in panels_meta[:MAX_PANELS]:
            mask_np = np.where(labels == item['label_index'], 255, 0).astype(np.uint8)
            mask_list.append((torch.from_numpy(mask_np).to(image.device, dtype=torch.float32) / 255.0).unsqueeze(0))

        if not mask_list: return (torch.zeros((1, img_h, img_w), device=image.device), 0, color_mask_debug, cleaned_frame_debug)
            
        return (torch.cat(mask_list, dim=0), len(mask_list), color_mask_debug, cleaned_frame_debug)

# --------------------------------------------------------------------
# Node 2 & 3 (変更なし)
# --------------------------------------------------------------------
class CropPanelForInpaint_Advanced:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "image": ("IMAGE",), "mask_batch": ("MASK",), "panel_index": ("INT", {"default": 1, "min": 1, "max": MAX_PANELS}), "fill_color_hex": ("STRING", {"default": "#000000"}), }}
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
        fill_color_tensor = torch.tensor([c / 255.0 for c in fill_rgb], device=image.device, dtype=torch.float32)
        expanded_mask = mask.unsqueeze(-1)
        masked_full_image = torch.where(expanded_mask > 0.5, image_tensor, fill_color_tensor)
        y1, y2, x1, x2 = coords[:, 0].min(), coords[:, 0].max(), coords[:, 1].min(), coords[:, 1].max()
        return (masked_full_image[y1:y2+1, x1:x2+1, :].unsqueeze(0), mask[y1:y2+1, x1:x2+1].unsqueeze(0))

class AssembleSinglePanel:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "base_image": ("IMAGE",), "generated_panel": ("IMAGE",), "mask_batch": ("MASK",), "panel_index": ("INT", {"default": 1, "min": 1, "max": MAX_PANELS}), }}
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
        resized_chw = torch.nn.functional.interpolate(img_chw, size=(h.item(), w.item()), mode='bilinear', align_corners=False)
        resized_image = resized_chw.squeeze(0).permute(1, 2, 0)
        target_region = canvas_tensor[y1:y2+1, x1:x2+1]
        sub_mask = mask[y1:y2+1, x1:x2+1].unsqueeze(-1)
        pasted_region = torch.where(sub_mask > 0.5, resized_image, target_region)
        canvas_tensor[y1:y2+1, x1:x2+1] = pasted_region
        return (canvas_tensor.unsqueeze(0),)