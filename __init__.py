# /ComfyUI/custom_nodes/MangaInpaintToolsInteractive/__init__.py

from .nodes import (
    MangaPanelDetector_Ultimate,
    InteractivePanelCreator,         # ★ インポートを追加
    CropPanelForInpaint_Advanced,
    ConditionalLatentScaler_Final,
    AssembleSinglePanel,
)

# このノードがWeb UI用のファイルを持つことをComfyUIに伝える
WEB_DIRECTORY = "./js"

# HTMLに直接JSファイルをインポートさせる
WEB_EXTENSIONS = {
    "MangaInpaintToolsInteractive": "/extensions/MangaInpaintToolsInteractive/main.js",
}


NODE_CLASS_MAPPINGS = {
    "MangaPanelDetector_Ultimate": MangaPanelDetector_Ultimate,
    "InteractivePanelCreator": InteractivePanelCreator,         # ★ マッピングを追加
    "CropPanelForInpaint_Advanced": CropPanelForInpaint_Advanced,
    "ConditionalLatentScaler_Final": ConditionalLatentScaler_Final,
    "AssembleSinglePanel": AssembleSinglePanel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MangaPanelDetector_Ultimate": "① Detect Panels by Color (Ultimate)",
    "InteractivePanelCreator": "① Create Panels Interactively", # ★ 表示名を追加
    "CropPanelForInpaint_Advanced": "② Crop Panel (Shape Aware)",
    "ConditionalLatentScaler_Final": "③ Conditionally Scale Latent",
    "AssembleSinglePanel": "④ Assemble Single Panel",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
print("### Loading: Manga Inpaint Tools (with Interactive Creator) ###")