"""Generate interactive 3D WebGL knowledge graph from graphify graph.json."""
import json
from pathlib import Path


def build_3d_graph():
    json_path = Path("ravana/graphify-out/graph.json")
    if not json_path.exists():
        print("graph.json not found!")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        g = json.load(f)

    nodes = g.get("nodes", [])
    raw_edges = g.get("links", []) or g.get("edges", [])

    palette = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
        "#00bcd4", "#ff4081", "#7c4dff", "#00e676", "#ff9100"
    ]

    formatted_nodes = []
    for n in nodes:
        c_id = n.get("community", 0)
        color = palette[c_id % len(palette)]
        formatted_nodes.append({
            "id": n["id"],
            "name": n.get("label") or n["id"],
            "group": c_id,
            "val": max(1, n.get("degree", 1)),
            "color": color,
            "file": n.get("file", ""),
        })

    node_ids = {n["id"] for n in nodes}
    formatted_links = []
    for e in raw_edges:
        s = e.get("source")
        t = e.get("target")
        if s in node_ids and t in node_ids:
            formatted_links.append({"source": s, "target": t})

    graph_data = {"nodes": formatted_nodes, "links": formatted_links}
    json_str = json.dumps(graph_data)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>RAVANA Code Knowledge Graph 3D</title>
  <style>
    body {{ margin: 0; overflow: hidden; background: #05050d; font-family: -apple-system, sans-serif; }}
    #info {{
      position: absolute; top: 15px; left: 15px; color: #fff;
      background: rgba(15, 15, 30, 0.85); padding: 12px 18px; border-radius: 8px;
      border: 1px solid #2a2a4e; pointer-events: none; max-width: 420px; z-index: 99;
    }}
    #info h1 {{ margin: 0 0 6px 0; font-size: 16px; color: #00e5ff; letter-spacing: 0.05em; }}
    #info p {{ margin: 3px 0; font-size: 12px; color: #aaa; }}
    #hover-info {{ margin-top: 8px; font-size: 13px; color: #fff; font-weight: 500; }}
  </style>
  <script src="https://unpkg.com/3d-force-graph"></script>
</head>
<body>
  <div id="info">
    <h1>RAVANA 3D Architecture Graph</h1>
    <p>Nodes: {len(formatted_nodes)} | Edges: {len(formatted_links)}</p>
    <p>Left Click + Drag: Rotate 3D | Right Click + Drag: Pan | Scroll: Zoom</p>
    <div id="hover-info">Hover over any node to inspect</div>
  </div>
  <div id="3d-graph"></div>

  <script>
    const gData = {json_str};
    const hoverEl = document.getElementById('hover-info');

    const Graph = ForceGraph3D()(document.getElementById('3d-graph'))
      .graphData(gData)
      .nodeLabel(node => '<b>Symbol:</b> ' + node.name + '<br><b>File:</b> ' + node.file + '<br><b>Connections:</b> ' + node.val)
      .nodeAutoColorBy('group')
      .nodeVal(node => Math.sqrt(node.val) * 2 + 1)
      .linkWidth(0.5)
      .linkOpacity(0.25)
      .onNodeHover(node => {{
        hoverEl.innerHTML = node ? '<b>' + node.name + '</b> (' + node.val + ' connections)<br><span style="color:#00e5ff; font-size:11px;">' + node.file + '</span>' : 'Hover over any node to inspect';
      }});

    Graph.d3Force('charge').strength(-30);
  </script>
</body>
</html>"""

    out_file = Path("ravana/graphify-out/graph_3d.html")
    out_file.write_text(html, encoding="utf-8")
    print(f"Generated {out_file} successfully! Size: {out_file.stat().st_size} bytes")


if __name__ == "__main__":
    build_3d_graph()
