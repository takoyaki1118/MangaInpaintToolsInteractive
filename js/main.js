// /ComfyUI/custom_nodes/MangaInpaintToolsInteractive/js/main.js

import { app } from "/scripts/app.js";

console.log("### MangaInpaintToolsInteractive JS: Script Loaded ###");

app.registerExtension({
    name: "MangaInpaintToolsInteractive.UI", 

    async nodeCreated(node) {
        // このUIを追加したいノードのクラス名をチェック
        if (node.comfyClass === "InteractivePanelCreator") {
        
            console.log(`★★★ SUCCESS: MangaInpaintToolsInteractive JS is linked to node ${node.id} ★★★`);

            try {
                // 内部データストレージ
                let regions = [];
                let isDrawing = false;
                let startPos = { x: 0, y: 0 };
                let currentRect = { x: 0, y: 0, w: 0, h: 0 };
                
                // Python側の隠しウィジェットを取得
                const jsonWidget = node.widgets.find(w => w.name === "regions_json");

                // --- UI要素の作成 ---
                const container = document.createElement("div");
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.gap = "5px";
                container.style.padding = "5px";
                
                const canvas = document.createElement("canvas");
                const ctx = canvas.getContext("2d");
                const buttonContainer = document.createElement("div");
                const clearButton = document.createElement("button");
                clearButton.textContent = "Clear Panels";
                buttonContainer.appendChild(clearButton);
                
                container.appendChild(canvas);
                container.appendChild(buttonContainer);

                // DOMウィジェットとしてノードに追加
                const customWidget = node.addDOMWidget("interactive_canvas", "div", container);
                customWidget.serialize = false; // UIの状態は保存しない

                // --- ヘルパー関数 ---

                // regions配列をJSON文字列に変換してPython側のウィジェットにセット
                const syncDataToWidget = () => {
                    const jsonString = JSON.stringify(regions);
                    // 変更があった場合のみ値を更新して再描画をトリガー
                    if (jsonWidget.value !== jsonString) {
                        jsonWidget.value = jsonString;
                        app.graph.setDirtyCanvas(true, true);
                    }
                };

                // Canvasの再描画
                const redraw = () => {
                    const widthWidget = node.widgets.find(w => w.name === "width");
                    const heightWidget = node.widgets.find(w => w.name === "height");
                    if (!widthWidget || !heightWidget) return;
                    
                    canvas.width = widthWidget.value;
                    canvas.height = heightWidget.value;
                    canvas.style.backgroundColor = "#222";
                    canvas.style.border = "1px solid #555";
                    
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.strokeStyle = "rgba(255, 255, 0, 0.8)";
                    ctx.lineWidth = 2;
                    
                    // 保存されている領域を描画
                    regions.forEach((region, i) => {
                        ctx.strokeRect(region.x, region.y, region.w, region.h);
                        ctx.fillStyle = "rgba(255, 255, 0, 0.8)";
                        ctx.font = "14px Arial";
                        ctx.fillText(`Panel ${i + 1}`, region.x + 5, region.y + 15);
                    });

                    // 現在描画中の矩形を描画
                    if (isDrawing) {
                        ctx.strokeStyle = "rgba(255, 0, 0, 0.8)";
                        ctx.strokeRect(currentRect.x, currentRect.y, currentRect.w, currentRect.h);
                    }
                };

                // マウスイベント座標をCanvas座標に変換
                const getScaledCoords = (e) => {
                    const rect = canvas.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) return { x: 0, y: 0 };
                    const scaleX = canvas.width / rect.width;
                    const scaleY = canvas.height / rect.height;
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;
                    return {
                        x: Math.round(Math.max(0, Math.min(canvas.width, x * scaleX))),
                        y: Math.round(Math.max(0, Math.min(canvas.height, y * scaleY))),
                    };
                };
                
                // --- イベントリスナー ---

                const handleMouseMove = (e) => {
                    if (!isDrawing) return;
                    const coords = getScaledCoords(e);
                    currentRect.x = Math.min(startPos.x, coords.x);
                    currentRect.y = Math.min(startPos.y, coords.y);
                    currentRect.w = Math.abs(startPos.x - coords.x);
                    currentRect.h = Math.abs(startPos.y - coords.y);
                    redraw();
                };

                const handleMouseUp = (e) => {
                    if (!isDrawing) return;
                    isDrawing = false;
                    window.removeEventListener("mousemove", handleMouseMove);
                    window.removeEventListener("mouseup", handleMouseUp);
                    // 小さすぎる矩形は無視
                    if (currentRect.w > 5 && currentRect.h > 5) {
                        regions.push({ ...currentRect });
                        syncDataToWidget();
                    }
                    redraw();
                };

                canvas.addEventListener("mousedown", (e) => {
                    isDrawing = true;
                    const coords = getScaledCoords(e);
                    startPos = { x: coords.x, y: coords.y };
                    currentRect = { x: startPos.x, y: startPos.y, w: 0, h: 0 };
                    window.addEventListener("mousemove", handleMouseMove);
                    window.addEventListener("mouseup", handleMouseUp);
                });

                clearButton.onclick = () => {
                    regions = [];
                    syncDataToWidget();
                    redraw();
                };
                
                // ノードのwidth/heightウィジェットが変更されたらCanvasを再描画
                const originalOnPropertyChanged = node.onPropertyChanged;
                node.onPropertyChanged = function(name, value) {
                    originalOnPropertyChanged?.apply(this, arguments);
                    if (name === 'width' || name === 'height') {
                        redraw();
                    }
                };
                
                // --- 初期化処理 ---
                if (jsonWidget.value) {
                    try { regions = JSON.parse(jsonWidget.value); } catch(e) { regions = []; }
                }
                redraw();

            } catch (error) {
                console.error("### MangaInpaintToolsInteractive JS Error in nodeCreated ###", error);
            }
        }
    }
});