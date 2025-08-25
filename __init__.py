# /ComfyUI/custom_nodes/MangaInpaintToolsInteractive/__init__.py

from .nodes import (
    MangaPanelDetector_Ultimate,
    CropPanelForInpaint_Advanced,
    ConditionalLatentScaler_Final, # ★ クラス名をFinalに変更
    AssembleSinglePanel,
)

NODE_CLASS_MAPPINGS = {
    "MangaPanelDetector_Ultimate": MangaPanelDetector_Ultimate,
    "CropPanelForInpaint_Advanced": CropPanelForInpaint_Advanced,
    "ConditionalLatentScaler_Final": ConditionalLatentScaler_Final, # ★
    "AssembleSinglePanel": AssembleSinglePanel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MangaPanelDetector_Ultimate": "① Detect Panels by Color (Ultimate)",
    "CropPanelForInpaint_Advanced": "② Crop Panel (Shape Aware)",
    "ConditionalLatentScaler_Final": "③ Conditionally Scale Latent", # ★
    "AssembleSinglePanel": "④ Assemble Single Panel",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
print("### Loading: Manga Inpaint Tools (Interactive) - v_final ###")