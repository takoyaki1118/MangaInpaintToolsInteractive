# /ComfyUI/custom_nodes/MangaInpaintToolsInteractive/__init__.py

from .nodes import (
    MangaPanelDetector_Final,
    CropPanelForInpaint_Advanced,
    AssembleSinglePanel,
)

NODE_CLASS_MAPPINGS = {
    "MangaPanelDetector_Final": MangaPanelDetector_Final,
    "CropPanelForInpaint_Advanced": CropPanelForInpaint_Advanced, # ★ 改良版クロップノード
    "AssembleSinglePanel": AssembleSinglePanel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MangaPanelDetector_Final": "① Detect Panels by Color",
    "CropPanelForInpaint_Advanced": "② Crop Panel (Shape Aware)", # ★
    "AssembleSinglePanel": "③ Assemble Single Panel",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
print("### Loading: Manga Inpaint Tools (Interactive) - v_final_crop ###")