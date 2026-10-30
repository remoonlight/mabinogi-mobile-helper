"""只读切片只保留 HUD 指纹 ROI，不含标注层。"""

from __future__ import annotations

HUD_SEEDS: list[dict] = [
    {"id": "hud.portrait", "roi": [0.0078, 0.0056, 0.1458, 0.1389], "notes": "头像"},
    {"id": "hud.home", "roi": [0.7281, 0.0014, 0.0286, 0.0542], "notes": "Home"},
    {"id": "hud.end_cam", "roi": [0.7750, 0.0014, 0.0286, 0.0542], "notes": "End摄像"},
    {"id": "hud.menu_esc", "roi": [0.8333, 0.0014, 0.0312, 0.0542], "notes": "ESC选单"},
    {"id": "hud.minimap", "roi": [0.8700, 0.0014, 0.1280, 0.1910], "notes": "小地图"},
    {"id": "hud.quest", "roi": [0.8570, 0.1670, 0.1380, 0.2500], "notes": "任务"},
    {"id": "hud.chat", "roi": [0.0104, 0.9306, 0.2240, 0.0556], "notes": "聊天"},
    {"id": "hud.skills", "roi": [0.7526, 0.8715, 0.2396, 0.1181], "notes": "技能栏"},
]
