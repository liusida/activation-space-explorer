"""Interactive HTML rendering for a target cluster's local PCA geometry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_cluster_pca_html(
    path: Path,
    *,
    coordinates: np.ndarray,
    explained_variance_ratio: np.ndarray,
    details: list[dict[str, Any]],
    target_position: int,
    model_id: str,
    layer_index: int,
    cluster_id: int,
    keyword: str = "fri",
) -> None:
    normalized_keyword = keyword.strip().lower()
    rendered_details = [
        {
            **detail,
            "keyword_match": bool(
                normalized_keyword and normalized_keyword in str(detail["token"]).strip().lower()
            ),
        }
        for detail in details
    ]
    data = {
        "model": model_id,
        "layer": layer_index,
        "cluster": cluster_id,
        "reference_count": len(coordinates) - 1,
        "explained": explained_variance_ratio.tolist(),
        "x": coordinates[:, 0].tolist(),
        "y": coordinates[:, 1].tolist(),
        "z": coordinates[:, 2].tolist(),
        "details": rendered_details,
        "target_position": target_position,
        "keyword": normalized_keyword,
    }
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    encoded = (
        encoded.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    )
    html = _HTML.replace("__DATA__", encoded)
    path.write_text(html, encoding="utf-8")


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Activation Space Explorer · target cluster PCA</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;font-size:16px;color:#172431;background:#f5f7fa}
*{box-sizing:border-box}body{margin:0}header{padding:18px 24px 14px;background:#fff;border-bottom:1px solid #d8e0e8}
h1{font-size:27px;margin:0 0 6px}.sub{font-size:16px;color:#607080}.workspace{display:grid;grid-template-columns:minmax(0,1fr) 430px;height:calc(100vh - 96px);min-height:650px}
#plot{width:100%;height:100%}.sidebar{background:#fff;border-left:1px solid #d8e0e8;padding:20px;overflow:auto}.panel{border:1px solid #d8e0e8;border-radius:10px;padding:15px 16px;margin-bottom:16px;color:#596a79;font-size:15px;line-height:1.6;overflow-wrap:anywhere}
.panel b{color:#172431;font-size:16px}.label{color:#7b8996;text-transform:uppercase;letter-spacing:.07em;font-size:12px;font-weight:700}.context{margin-top:11px;padding:12px;background:#f6f8fa;border-radius:7px;white-space:pre-wrap;color:#344554;font-size:16px}.chosen{background:#ffd88a;color:#172431;border-radius:3px;padding:1px 3px;font-weight:650}.target{border-color:#f0ad4e;background:#fffaf0}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:9px}.metric{padding:10px;background:#f6f8fa;border-radius:7px}.metric strong{display:block;color:#172431;font-size:17px}button{border:1px solid #ccd6df;background:#fff;border-radius:7px;padding:9px 12px;color:#263746;cursor:pointer;font-size:14px}button:hover{background:#f4f7f9}
@media(max-width:900px){.workspace{grid-template-columns:1fr;grid-template-rows:70vh auto;height:auto}.sidebar{border-left:0;border-top:1px solid #d8e0e8}}
</style>
</head>
<body>
<header><h1 id="title"></h1><div class="sub" id="subtitle"></div></header>
<div class="workspace"><div id="plot"></div><aside class="sidebar">
  <div class="panel target" id="target"></div>
  <div class="panel" id="selection"><b>Selected activation</b><br>Click a point to inspect its token context.</div>
  <div class="panel" id="guide"></div>
</aside></div>
<script>
const D=__DATA__,pct=x=>(100*x).toFixed(2)+'%',esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const target=D.details[D.target_position],sum=D.explained.reduce((a,b)=>a+b,0);
document.getElementById('title').textContent=`${D.model} layer ${D.layer} · cluster ${D.cluster}`;
document.getElementById('subtitle').textContent=`${D.reference_count.toLocaleString()} reference members + 1 target · local PCA-3 explains ${pct(sum)} of within-cluster variance`;
const contextHtml=d=>`${esc(d.before)}<span class="chosen">${esc(d.token)||'∅'}</span>${esc(d.after)}`;
const display=x=>typeof x==='number'?x.toLocaleString():esc(x);
const detailHtml=(heading,d)=>`<b>${heading}</b><div class="metrics"><div class="metric"><span class="label">Reference row</span><strong>${display(d.row)}</strong></div><div class="metric"><span class="label">Token ID</span><strong>${d.token_id}</strong></div><div class="metric"><span class="label">Sequence</span><strong>${display(d.sequence)}</strong></div><div class="metric"><span class="label">Position</span><strong>${d.position}</strong></div></div><div class="context">${contextHtml(d)}</div>`;
document.getElementById('target').innerHTML=detailHtml('Target activation',target);
const members=D.details.map((_,i)=>i).filter(i=>i!==D.target_position);
const matches=members.filter(i=>D.details[i].keyword_match),neutral=members.filter(i=>!D.details[i].keyword_match);
const trace=(indices,name,color,size,opacity,symbol='circle',showlegend=false)=>({type:'scatter3d',mode:'markers',name,showlegend,x:indices.map(i=>D.x[i]),y:indices.map(i=>D.y[i]),z:indices.map(i=>D.z[i]),customdata:indices,text:indices.map(i=>D.details[i].row),hovertext:indices.map(i=>D.details[i].token),marker:{size,color,opacity,symbol,line:{color:'#172431',width:symbol==='diamond'?2:0}},hovertemplate:`${name}<br>activation row %{text}<br>token %{hovertext}<extra>click for context</extra>`});
const traces=[trace(neutral,'Cluster member','#7f91a2',3,.35),trace(matches,`Contains “${D.keyword}”`,'#f59e0b',5,.9,'circle',matches.length>0),trace([D.target_position],'Target activation','#ef553b',10,1,'diamond',true)];
document.getElementById('guide').innerHTML=`<div class="label">How to read this view</div><p>Gray points are activations assigned to this k-means cluster. ${D.keyword?`Orange points have decoded token text that contains <b>${esc(D.keyword)}</b> after trimming and lowercasing (${matches.length.toLocaleString()} matches).`:``} The red diamond is the target activation. PCA is fitted only on reference members.</p><button id="reset">Reset camera</button>`;
const axis=(name,value)=>({title:`${name} · ${pct(value)}`,backgroundcolor:'#fff',gridcolor:'#dce4eb',showbackground:true,zerolinecolor:'#aebbc7'});
const layout={margin:{l:0,r:0,t:0,b:0},paper_bgcolor:'#f5f7fa',font:{family:'Inter, ui-sans-serif, system-ui, sans-serif',size:16,color:'#263746'},hoverlabel:{font:{size:16}},legend:{x:.01,y:.99,font:{size:16},bgcolor:'rgba(255,255,255,.86)'},scene:{xaxis:axis('PC1',D.explained[0]),yaxis:axis('PC2',D.explained[1]),zaxis:axis('PC3',D.explained[2]),aspectmode:'data'}};
const config={responsive:true,displaylogo:false,scrollZoom:true};
Plotly.newPlot('plot',traces,layout,config).then(plot=>{plot.on('plotly_click',event=>{const point=event.points&&event.points[0];if(!point)return;const d=D.details[point.customdata];document.getElementById('selection').innerHTML=detailHtml(point.customdata===D.target_position?'Target activation':'Cluster member',d)});document.getElementById('reset').onclick=()=>Plotly.relayout(plot,{'scene.camera':null})});
</script>
</body>
</html>"""
