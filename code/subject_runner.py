"""Run a GLM-5.3 subject via the Zhipu API (thinking=max), no tools at all.

Usage: python subject_runner.py <exp> [<exp>...]
  exp in {e1,e2,e3,e4,e5}
Writes: results/<exp>_answer.md (final content) and results/<exp>_raw.json
        (full response incl. reasoning, usage).
"""
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBJ = os.path.join(ROOT, "subjects")
RESULTS = os.path.join(ROOT, "results")
AUTH = r"C:/Users/js/.pi/agent/auth.json"
URL = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"

FILES = {
    "e1": [f"network_{i:02d}.txt" for i in range(1, 13)],
    "e2": [f"net_{c}.txt" for c in "ABCDEF"],
    "e3": [f"net_{c}.txt" for c in "ABCD"],
    "e4": ["ddpm_circle.txt", "ddpm_square.txt", "ddpm_blend_alpha25.txt",
           "ddpm_blend_alpha50.txt", "ddpm_blend_alpha75.txt"],
    "e5": ["ddpm_circle.txt", "ddpm_square.txt"],
    "e5b": ["ddpm_circle.txt", "ddpm_square.txt"],
}

SYSTEM = ("You are a subject in a blinded scientific experiment. You have NO tools, "
          "no code execution and no file system: the complete contents of every file "
          "you need are provided below, in full. Work autonomously and rigorously; "
          "obey the experiment rules exactly. Your final answer must follow the "
          "required output format exactly, with no meta-commentary.")

PREAMBLE = ("NOTE: you have no tools and cannot run anything; every file's complete "
            "content is already provided below, so 'you may read the files' is already "
            "satisfied. Reason natively over the numbers.")


def build_user(exp):
    prompt = open(os.path.join(SUBJ, exp, "PROMPT.md"), encoding="utf-8").read()
    parts = [prompt, "", PREAMBLE, "",
             "================ FILE CONTENTS (complete) ================", ""]
    for f in FILES[exp]:
        content = open(os.path.join(SUBJ, exp, f), encoding="utf-8").read()
        parts.append(f"----- BEGIN FILE {f} -----")
        parts.append(content)
        parts.append(f"----- END FILE {f} -----")
        parts.append("")
    return "\n".join(parts)


def call_api(exp, max_tokens=131072):
    key = json.load(open(AUTH))["zhipu"]["key"]
    body = {
        "model": "glm-5.3",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user(exp)},
        ],
        "thinking": {"type": "enabled", "clear_thinking": False},
        "reasoning_effort": "max",
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "stream": True,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    t0 = time.time()
    content, reasoning, usage = [], [], None
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                j = json.loads(data)
            except json.JSONDecodeError:
                continue
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                d = ch.get("delta", {}) or {}
                if d.get("content"):
                    content.append(d["content"])
                if d.get("reasoning_content"):
                    reasoning.append(d["reasoning_content"])
    return {
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "usage": usage,
        "seconds": round(time.time() - t0, 1),
    }


def main():
    os.makedirs(RESULTS, exist_ok=True)
    for exp in sys.argv[1:]:
        print(f"[{exp}] calling glm-5.3 (thinking=max) ...", flush=True)
        try:
            r = call_api(exp)
        except Exception as e:
            print(f"[{exp}] FAILED: {e}", flush=True)
            continue
        with open(os.path.join(RESULTS, f"{exp}_raw.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=1)
        with open(os.path.join(RESULTS, f"{exp}_answer.md"), "w", encoding="utf-8") as f:
            f.write(r["content"])
        print(f"[{exp}] done in {r['seconds']}s | content {len(r['content'])} chars | "
              f"reasoning {len(r['reasoning'])} chars | usage {r['usage']}", flush=True)


if __name__ == "__main__":
    main()
