"""ComfyUI 本地 HTTP API 最小客户端。

直连 127.0.0.1:8188，绕开 comfy-cli 的 autogrow 预检。
供 run_calibration.py 使用；不依赖任何本机绝对路径。
"""
import json, os, sys, time, urllib.parse, urllib.request, urllib.error

BASE = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")


def _req(path, data=None, raw=False, timeout=60):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            return r.status, (body if raw else json.loads(body.decode()))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:1200]
    except Exception as e:
        return -1, str(e)


def submit(wf):
    st, resp = _req("/prompt", {"prompt": wf})
    if st != 200:
        raise RuntimeError("submit failed: %s" % resp)
    if isinstance(resp, dict) and resp.get("node_errors"):
        raise RuntimeError("node_errors: %s" % json.dumps(
            resp["node_errors"], ensure_ascii=False)[:1500])
    return resp["prompt_id"]


def wait(pid, timeout=900, poll=3):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st, hist = _req("/history/%s" % pid)
        if st == 200 and isinstance(hist, dict) and pid in hist:
            h = hist[pid]
            status = (h.get("status") or {})
            if status.get("status_str") == "error":
                msgs = []
                for m in status.get("messages", []):
                    msgs.append(json.dumps(m, ensure_ascii=False)[:400])
                raise RuntimeError("execution error: %s" % " | ".join(msgs))
            return h
        time.sleep(poll)
    raise RuntimeError("timeout waiting %s" % pid)


def outputs_of(hist, pid):
    """返回 [(node_id, {filename, subfolder, type}), ...]"""
    out = []
    for nid, o in (hist.get("outputs") or {}).items():
        for key in ("images", "gifs", "videos", "audios"):
            for item in o.get(key, []) or []:
                out.append((nid, key, item))
    return out


def download(item, dest):
    fn = item["filename"]
    sub = item.get("subfolder", "")
    typ = item.get("type", "output")
    q = "/view?filename=%s&type=%s" % (urllib.parse.quote(fn), typ)
    if sub:
        q += "&subfolder=%s" % urllib.parse.quote(sub)
    st, body = _req(q, raw=True, timeout=300)
    if st != 200:
        raise RuntimeError("download failed %s: %s" % (fn, body))
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(body)
    return dest, len(body)
