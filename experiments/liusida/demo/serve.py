#!/usr/bin/env python3
"""Interactive explorer: type a prompt, click a token, open its soft-cluster view.

Requires a completed shared library (capture + clusters) and a GPU for the forward pass.

  uv run --project experiments/liusida/demo \\
    python experiments/liusida/demo/serve.py --port 7860

Then open http://127.0.0.1:7860 (or SSH-tunnel from your laptop).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from analyze import (
    extract_prompt_token_activations,
    load_model,
    load_shared_artifacts,
    render_token_decomposition_html,
    summarize_token,
)
from library import OUTPUT_ROOT, find_latest_library, migrate_flat_output_into_library, slugify

HERE = Path(__file__).resolve().parent


def load_config() -> dict[str, Any]:
    with (HERE / "config.toml").open("rb") as file:
        return tomllib.load(file)


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Activation Space Explorer · interactive</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#172431;background:#f4f6f8}
*{box-sizing:border-box}body{margin:0}
header{padding:16px 20px;background:#fff;border-bottom:1px solid #d8e0e8}
h1{margin:0 0 4px;font-size:22px}.sub{color:#607080;font-size:13px}
.layout{display:grid;grid-template-columns:380px 1fr;min-height:calc(100vh - 74px)}
.panel{padding:16px;border-right:1px solid #d8e0e8;background:#fff}
textarea{width:100%;min-height:110px;border:1px solid #ccd6df;border-radius:8px;padding:10px;font:inherit}
button{margin-top:10px;border:0;border-radius:8px;background:#2563eb;color:#fff;padding:10px 14px;cursor:pointer;font:inherit}
button:disabled{opacity:.5;cursor:wait}
.tokens{display:flex;flex-wrap:wrap;gap:6px;margin-top:14px}
.tok{border:1px solid #d5dee7;background:#f8fafc;border-radius:8px;padding:6px 8px;cursor:pointer;font-size:13px;line-height:1.3}
.tok:hover,.tok.active{border-color:#2563eb;background:#eff6ff}
.tok small{display:block;color:#64748b;font-size:11px}
.meta{margin-top:12px;font-size:12px;color:#64748b;white-space:pre-wrap}
.viewer{display:flex;flex-direction:column;min-width:0}
.viewer iframe{border:0;width:100%;flex:1;min-height:70vh;background:#f5f7fa}
.empty{padding:40px;color:#64748b}
@media(max-width:900px){.layout{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>Activation Space Explorer</h1>
  <div class="sub">输入 prompt → 分析每个 token 的软聚类归属 → 点击 token 打开分解图（复用已拟合的 OpenWebText k-means）</div>
</header>
<div class="layout">
  <aside class="panel">
    <label for="prompt"><b>Prompt</b></label>
    <textarea id="prompt" placeholder="The capital of France is Paris.">The capital of France is Paris.</textarea>
    <button id="go">Analyze prompt</button>
    <div class="tokens" id="tokens"></div>
    <div class="meta" id="meta">Ready.</div>
  </aside>
  <section class="viewer">
    <div class="empty" id="empty">点左侧某个 token 后，这里显示该激活的多簇分解 3D 图。</div>
    <iframe id="frame" hidden></iframe>
  </section>
</div>
<script>
const promptEl=document.getElementById('prompt');
const tokensEl=document.getElementById('tokens');
const metaEl=document.getElementById('meta');
const frame=document.getElementById('frame');
const empty=document.getElementById('empty');
const go=document.getElementById('go');
let current=null;

function esc(s){return String(s).replace(/[&<>"']/g,c=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));}

async function analyze(){
  go.disabled=true; metaEl.textContent='Running forward + soft assignment…';
  tokensEl.innerHTML=''; frame.hidden=true; empty.hidden=false;
  try{
    const res=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt:promptEl.value})});
    const data=await res.json();
    if(!res.ok) throw new Error(data.error||res.statusText);
    current=data;
    metaEl.textContent=`${data.tokens.length} tokens · layer ${data.layer} · library ${data.library}`;
    tokensEl.innerHTML=data.tokens.map((t,i)=>{
      const top=(t.summary.top||[]).slice(0,3).map(x=>`c${x.cluster_id}:${(100*x.weight).toFixed(0)}%`).join(' ');
      return `<button class="tok" data-i="${i}" title="${esc(top)}"><b>${esc(JSON.stringify(t.token))}</b>
        <small>hard c${t.summary.hard_cluster} · ${esc(top)}</small></button>`;
    }).join('');
    tokensEl.querySelectorAll('.tok').forEach(btn=>{
      btn.onclick=()=>openToken(Number(btn.dataset.i), btn);
    });
  }catch(err){
    metaEl.textContent='Error: '+err.message;
  }finally{
    go.disabled=false;
  }
}

async function openToken(i, btn){
  if(!current) return;
  tokensEl.querySelectorAll('.tok').forEach(el=>el.classList.remove('active'));
  btn.classList.add('active');
  metaEl.textContent=`Building decomposition for token #${i} ${JSON.stringify(current.tokens[i].token)} …`;
  empty.hidden=false; frame.hidden=true;
  try{
    const res=await fetch('/api/token_view',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt:current.prompt, position:i})});
    const data=await res.json();
    if(!res.ok) throw new Error(data.error||res.statusText);
    frame.src=data.url+'?t='+Date.now();
    frame.hidden=false; empty.hidden=true;
    metaEl.textContent=`Token #${i} → hard cluster ${data.hard_cluster} · ${data.url}`;
  }catch(err){
    metaEl.textContent='Error: '+err.message;
  }
}

go.onclick=analyze;
</script>
</body>
</html>
"""


class ExplorerState:
    def __init__(self, config: dict[str, Any], library: Path) -> None:
        self.config = config
        self.library = library
        self.shared = load_shared_artifacts(library, config)
        self.tokenizer = None
        self.model = None
        self.cache_dir = OUTPUT_ROOT / "serve_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._prompt_cache: dict[str, list[dict[str, Any]]] = {}

    def ensure_model(self) -> None:
        if self.model is None:
            self.tokenizer, self.model = load_model(self.config)

    def activations_for_prompt(self, prompt: str) -> list[dict[str, Any]]:
        key = hashlib.sha1(prompt.encode()).hexdigest()
        if key in self._prompt_cache:
            return self._prompt_cache[key]
        self.ensure_model()
        records = extract_prompt_token_activations(
            tokenizer=self.tokenizer,
            model=self.model,
            prompt=prompt,
            layer_index=int(self.config["model"]["layer_index"]),
            device=self.config["model"]["device"],
            dimension=int(self.shared["dimension"]),
            config=self.config,
        )
        # Drop bulky arrays from JSON responses later; keep activations in memory cache.
        self._prompt_cache[key] = records
        return records


def make_handler(state: ExplorerState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            print("[serve]", args[0] if args else format)

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode()
            self._send(code, raw, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
                return
            if parsed.path.startswith("/views/"):
                name = Path(parsed.path).name
                path = state.cache_dir / name
                if not path.exists():
                    self._send_json(404, {"error": f"missing view {name}"})
                    return
                self._send(200, path.read_bytes(), "text/html; charset=utf-8")
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode() or "{}")
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid JSON"})
                return
            try:
                if self.path == "/api/analyze":
                    self._analyze(payload)
                elif self.path == "/api/token_view":
                    self._token_view(payload)
                else:
                    self._send_json(404, {"error": "not found"})
            except Exception as exc:  # noqa: BLE001 - surface to UI
                self._send_json(500, {"error": str(exc)})

        def _analyze(self, payload: dict[str, Any]) -> None:
            prompt = str(payload.get("prompt", "")).strip()
            if not prompt:
                self._send_json(400, {"error": "prompt is empty"})
                return
            viz = state.config.get("visualization", {})
            records = state.activations_for_prompt(prompt)
            tokens = []
            for record in records:
                summary = summarize_token(
                    record["activation"],
                    clusterer=state.shared["clusterer"],
                    top_k=int(viz.get("top_k_clusters", 8)),
                    temperature=float(viz.get("softmax_temperature", 0.0)),
                )
                tokens.append(
                    {
                        "position": record["position"],
                        "token": record["token"],
                        "token_id": record["token_id"],
                        "character_start": record["character_start"],
                        "character_stop": record["character_stop"],
                        "summary": summary,
                    }
                )
            self._send_json(
                200,
                {
                    "prompt": prompt,
                    "tokens": tokens,
                    "layer": state.config["model"]["layer_index"],
                    "library": str(state.library),
                },
            )

        def _token_view(self, payload: dict[str, Any]) -> None:
            prompt = str(payload.get("prompt", "")).strip()
            position = int(payload.get("position", -1))
            records = state.activations_for_prompt(prompt)
            if not (0 <= position < len(records)):
                self._send_json(400, {"error": f"position {position} out of range"})
                return
            record = records[position]
            digest = hashlib.sha1(
                f"{prompt}\0{position}\0{state.config.get('visualization')}".encode()
            ).hexdigest()[:16]
            token_slug = slugify(str(record["token"]), fallback=f"pos{position}")
            html_name = f"{digest}_{token_slug}.html"
            html_path = state.cache_dir / html_name
            if not html_path.exists():
                render_summary = render_token_decomposition_html(
                    html_path,
                    config=state.config,
                    shared=state.shared,
                    prompt=prompt,
                    token_record=record,
                )
            else:
                render_summary = {"hard_cluster": "cached"}
                # Recompute hard cluster cheaply for the response.
                summary = summarize_token(
                    record["activation"],
                    clusterer=state.shared["clusterer"],
                    top_k=1,
                    temperature=float(
                        state.config.get("visualization", {}).get("softmax_temperature", 0.0)
                    ),
                )
                render_summary = {"hard_cluster": summary["hard_cluster"]}
            self._send_json(
                200,
                {
                    "url": f"/views/{html_name}",
                    "hard_cluster": render_summary["hard_cluster"],
                    "position": position,
                    "token": record["token"],
                },
            )

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--library", type=Path, default=None)
    args = parser.parse_args()

    config = load_config()
    migrate_flat_output_into_library(config)
    library = Path(args.library) if args.library else find_latest_library()
    if library is None:
        raise SystemExit(
            "No shared library found. Run the demo once first:\n"
            "  uv run --project experiments/liusida/demo python experiments/liusida/demo/run.py"
        )
    library = library.resolve()
    print(f"[serve] library = {library}")
    print(f"[serve] loading shared artifacts…")
    state = ExplorerState(config, library)
    print(f"[serve] ready · {state.shared['dimension']}D · "
          f"{len(state.shared['activations'].values):,} reference tokens")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"[serve] http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopped")


if __name__ == "__main__":
    main()
