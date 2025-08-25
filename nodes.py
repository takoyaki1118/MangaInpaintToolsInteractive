# /ComfyUI/custom_nodes/MangaInpaintToolsInteractive/nodes.py

import torch
import numpy as np
import cv2

# --------------------------------------------------------------------
# Node 1: MangaPanelDetectorAdvanced
# Cannyエッジ検出を用いて、絵が描かれた画像からもコマを検出する高機能版
# --------------------------------------------------------------------
class MangaPanelDetectorAdvanced:
    MAX_PANELS = 8

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_sigma": ("INT", {"default": 5, "min": 1, "max": 15, "step": 2}),
                "canny_low_threshold": ("INT", {"default": 50, "min": 0, "max": 255}),
                "canny_high_threshold": ("INT", {"default": 150, "min": 0, "max": 255}),
                "sort_panels_by": (["top-to-bottom", "left-to-right", "largest-first"],),
                "min_area": ("INT", {"default": 5000, "min": 0, "max": 999999}),
            }
        }

    RETURN_TYPES = ("MASK", "INT")
    RETURN_NAMES = ("mask_batch", "panel_count")
    FUNCTION, CATEGORY = "detect_panels", "Manga Inpaint"

    def detect_panels(self, image, blur_sigma, canny_low_threshold, canny_high_threshold, sort_panels_by, min_area):
        base_image_tensor = image[0].unsqueeze(0)
        img_h, img_w = base_image_tensor.shape[1], base_image_tensor.shape[2]
        base_img_cv2 = (base_image_tensor.cpu().numpy().squeeze(0) * 255).astype(np.uint8)

        # 1. グレースケール変換とぼかし（ノイズ除去）
        gray = cv2.cvtColor(base_img_cv2, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_sigma, blur_sigma), 0)

        # 2. Cannyエッジ検出で枠線を抽出
        edges = cv2.Canny(blurred, canny_low_threshold, canny_high_threshold)

        # 3. 検出されたエッジから輪郭を見つける
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return (torch.zeros((1, img_h, img_w), device=image.device), 0)
        
        # 輪郭とメタデータ（バウンディングボックス、面積）をリスト化
        contours_with_meta = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > min_area: # 小さすぎる領域をここで除外
                x, y, w, h = cv2.boundingRect(c)
                contours_with_meta.append({'contour': c, 'box': (x, y, w, h), 'area': area})

        # ソート
        if sort_panels_by == "largest-first":
            contours_with_meta.sort(key=lambda item: item['area'], reverse=True)
        elif sort_panels_by == "top-to-bottom":
            contours_with_meta.sort(key=lambda item: (item['box'][1], item['box'][0]))
        else: # left-to-right
            contours_with_meta.sort(key=lambda item: (item['box'][0], item['box'][1]))

        # マスクを生成
        mask_list = []
        num_detected = min(len(contours_with_meta), self.MAX_PANELS)
        for i in range(num_detected):
            contour = contours_with_meta[i]['contour']
            mask_np = np.zeros((img_h, img_w), dtype=np.uint8)
            # 輪郭の内側を塗りつぶしてマスクを作成
            cv2.drawContours(mask_np, [contour], -1, 255, thickness=cv2.FILLED)
            mask_tensor = torch.from_numpy(mask_np).to(image.device, dtype=torch.float32) / 255.0
            mask_list.append(mask_tensor.unsqueeze(0))

        if not mask_list:
            return (torch.zeros((1, img_h, img_w), device=image.device), 0)
            
        return (torch.cat(mask_list, dim=0), len(mask_list))

# --------------------------------------------------------------------
# Node 2: CropPanelForInpaint (変更なし)
# --------------------------------------------------------------------
class CropPanelForInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask_batch": ("MASK",),
                "panel_index": ("INT", {"default": 1, "min": 1, "max": MangaPanelDetectorAdvanced.MAX_PANELS}),
            }
        }
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("cropped_panel", "cropped_mask")
    FUNCTION, CATEGORY = "crop", "Manga Inpaint"

    def crop(self, image, mask_batch, panel_index):
        index = panel_index - 1
        if image.shape[0] == 0 or mask_batch.shape[0] == 0 or index < 0 or index >= mask_batch.shape[0]:
            dummy_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device=image.device)
            dummy_mask = torch.zeros((1, 64, 64), dtype=torch.float32, device=image.device)
            return (dummy_img, dummy_mask)
        image_tensor = image[0]; mask = mask_batch[index]
        coords = torch.nonzero(mask, as_tuple=False)
        if coords.shape[0] == 0:
            dummy_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device=image.device)
            dummy_mask = torch.zeros((1, 64, 64), dtype=torch.float32, device=image.device)
            return (dummy_img, dummy_mask)
        y_coords, x_coords = coords[:, 0], coords[:, 1]
        y1, y2, x1, x2 = y_coords.min(), y_coords.max(), x_coords.min(), x_coords.max()
        cropped_image = image_tensor[y1:y2+1, x1:x2+1, :]
        cropped_mask = mask[y1:y2+1, x1:x2+1]
        return (cropped_image.unsqueeze(0), cropped_mask.unsqueeze(0))

# --------------------------------------------------------------------
# Node 3: AssembleSinglePanel
# 生成された1枚の画像を、指定されたインデックスの場所に合成するノード
# --------------------------------------------------------------------
class AssembleSinglePanel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "generated_panel": ("IMAGE",),
                "mask_batch": ("MASK",),
                "panel_index": ("INT", {"default": 1, "min": 1, "max": MangaPanelDetectorAdvanced.MAX_PANELS}),
            }
        }
    RETURN_TYPES, FUNCTION, CATEGORY = ("IMAGE",), "assemble", "Manga Inpaint"
    
    def assemble(self, base_image, generated_panel, mask_batch, panel_index):
        canvas_tensor = base_image[0].clone()
        index = panel_index - 1
        
        if generated_panel.shape[0] == 0 or mask_batch.shape[0] == 0 or index < 0 or index >= mask_batch.shape[0]:
            return (base_image,) # 何もせず元の画像を返す

        image_to_paste = generated_panel[0]
        mask = mask_batch[index]
        
        coords = torch.nonzero(mask, as_tuple=False)
        if coords.shape[0] == 0: return (base_image,)
        
        y_coords, x_coords = coords[:, 0], coords[:, 1]
        y1, y2, x1, x2 = y_coords.min(), y_coords.max(), x_coords.min(), x_coords.max()
        h, w = y2 - y1 + 1, x2 - x1 + 1
        if h <= 0 or w <= 0: return (base_image,)

        img_to_resize_chw = image_to_paste.permute(2, 0, 1).unsqueeze(0)
        resized_chw = torch.nn.functional.interpolate(img_to_resize_chw, size=(h.item(), w.item()), mode='bilinear', align_corners=False)
        resized_image = resized_chw.squeeze(0).permute(1, 2, 0)
        
        target_region = canvas_tensor[y1:y2+1, x1:x2+1]
        sub_mask = mask[y1:y2+1, x1:x2+1].unsqueeze(-1)
        pasted_region = torch.where(sub_mask > 0.5, resized_image, target_region)
        canvas_tensor[y1:y2+1, x1:x2+1] = pasted_region
            
        return (canvas_tensor.unsqueeze(0),)