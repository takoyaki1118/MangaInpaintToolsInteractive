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

# /ComfyUI/custom_nodes/MangaInpaintToolsInteractive/__init__.py の
# NODE_DISPLAY_NAME_MAPPINGS を以下のように書き換える

NODE_DISPLAY_NAME_MAPPINGS = {
    "MangaPanelDetector_Ultimate": "② Detect Panels by Color (Ultimate)",
    "InteractivePanelCreator": "① Create Panel Layout Image", # ← 表示名を変更
    "CropPanelForInpaint_Advanced": "③ Crop Panel (Shape Aware)",
    "ConditionalLatentScaler_Final": "④ Conditionally Scale Latent",
    "AssembleSinglePanel": "⑤ Assemble Single Panel",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
print("### Loading: Manga Inpaint Tools (with Interactive Creator) ###")