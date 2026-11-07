import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
with open("scripts/current_ui_config.json", "r", encoding="utf-8-sig") as f:
    cfg = json.load(f)

comps = cfg.get("components", [])
print(f"Total UI Components: {len(comps)}")
for c in comps:
    t = c.get("type", "")
    props = c.get("props", {})
    label = props.get("label") or props.get("value")
    if t in ["accordion", "tabitem", "button", "radio", "dropdown", "checkbox", "dataframe", "plot", "markdown"]:
        desc = str(label).replace("\n", " ")[:65]
        print(f"[{c['id']:2d}] {t.upper():12s}: {desc}")
