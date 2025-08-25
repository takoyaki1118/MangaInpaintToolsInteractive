# /ComfyUI/custom_nodes/MangaInpaintToolsInteractive/__init__.py

from .nodes import MangaPanelDetectorAdvanced, CropPanelForInpaint, AssembleSinglePanel

NODE_CLASS_MAPPINGS = {
    "MangaPanelDetectorAdvanced": MangaPanelDetectorAdvanced,
    "CropPanelForInpaint": CropPanelForInpaint,
    "AssembleSinglePanel": AssembleSinglePanel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MangaPanelDetectorAdvanced": "① Detect Manga Panels (Advanced)",
    "CropPanelForInpaint": "② Crop Panel for Inpaint",
    "AssembleSinglePanel": "③ Assemble Single Panel",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
print("### Loading: Manga Inpaint Tools (Interactive) ###")