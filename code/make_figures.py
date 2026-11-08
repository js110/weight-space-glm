"""Generate all figures for the article. Uses system python (numpy+matplotlib+PIL)."""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

for f in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
    if os.path.exists(f):
        fm.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def parse_W1(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    for i, l in enumerate(lines):
        m = re.search(r"(\d+) rows x (\d+) cols", l)
        if m and l.startswith("W1"):
            r, c = int(m.group(1)), int(m.group(2))
            rows = [[float(v) for v in lines[j].split()] for j in range(i + 1, i + 1 + r)]
            return np.array(rows)
    raise ValueError(path)


def parse_b1(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    for i, l in enumerate(lines):
        if l.startswith("b1 ("):
            return np.array([float(v) for v in lines[i + 1].split()])
    raise ValueError(path)


# ---------- Fig 1: E1 three conditions ----------

def fig1():
    # from secret mapping (verified earlier): 04=trained(x3), 01=untrained, 03=scrambled
    cases = [
        ("没练过（随机初始化）", "subjects/e1/network_01.txt"),
        ("打散的（数字洗过牌）", "subjects/e1/network_03.txt"),
        ("练过的（学了 x3 规则）", "subjects/e1/network_04.txt"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(13, 8.5), gridspec_kw={"height_ratios": [2.4, 1, 0.7]})
    for k, (title, path) in enumerate(cases):
        W1 = parse_W1(os.path.join(ROOT, path))
        b1 = parse_b1(os.path.join(ROOT, path))
        ax = axes[0][k]
        im = ax.imshow(W1, cmap="RdBu_r", vmin=-1.3, vmax=1.3, aspect="auto", interpolation="nearest")
        ax.set_title(title, fontsize=13)
        ax.set_xticks(range(0, 16, 3), [f"x{i}" for i in range(0, 16, 3)])
        ax.set_ylabel("32 个隐藏单元" if k == 0 else "")
        colw = np.abs(W1).sum(axis=0)
        ax2 = axes[1][k]
        colors = ["#d62728" if i == 3 else "#1f77b4" for i in range(16)]
        ax2.bar(range(16), colw, color=colors)
        ax2.set_xticks(range(0, 16, 3), [f"x{i}" for i in range(0, 16, 3)])
        ax2.set_ylabel("各输入接线重量" if k == 0 else "")
        ax2.set_title(f"x3 列重量 {colw[3]:.1f}，其余中位数 {np.median(np.delete(colw,3)):.1f}", fontsize=10)
        ax3 = axes[2][k]
        ax3.bar(range(32), np.abs(b1), color="#2ca02c")
        ax3.set_ylabel("偏置大小" if k == 0 else "")
        ax3.set_xticks([0, 31], ["单元1", "单元32"])
        ax3.set_title("偏置：没练过的全为 0" if k == 0 else "", fontsize=10)
    fig.colorbar(im, ax=axes[0], shrink=0.8, pad=0.015, label="权重值（红=正 蓝=负）")
    fig.suptitle("实验一考题：同为 32×16 的第一层权重 + 32 个偏置，三种来历的指纹完全不同", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.99, 0.96])
    fig.savefig(os.path.join(FIG, "e1_conditions.png"), dpi=130)
    plt.close(fig)


# ---------- Fig 2: E2 accuracy prediction ----------

def fig2():
    truth = json.load(open(os.path.join(ROOT, "secret", "e2_truth.json"), encoding="utf-8"))
    ans = open(os.path.join(ROOT, "results", "e2_answer.md"), encoding="utf-8").read()
    names, preds, trues, labels = [], [], [], []
    for m in re.finditer(r"net_([A-F]).*?accuracy:\s*([0-9.]+)", ans):
        tag = m.group(1)
        names.append(f"net_{tag}")
        preds.append(float(m.group(2)))
        trues.append(truth[f"net_{tag}.txt"]["test_acc"] * 100)
        labels.append(truth[f"net_{tag}.txt"]["rule"])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 0.18, trues, 0.36, label="真实考试成绩", color="#1f77b4")
    ax.bar(x + 0.18, preds, 0.36, label="GLM-5.3 只看权重的预测", color="#ff7f0e")
    for i in range(len(names)):
        ax.text(x[i], max(trues[i], preds[i]) + 1.2, f"差 {abs(preds[i]-trues[i]):.1f} 分",
                ha="center", fontsize=10)
    ax.set_xticks(x, [f"{n}\n{t}" for n, t in zip(names, labels)], fontsize=9)
    ax.set_ylim(70, 106)
    ax.set_ylabel("测试准确率（%）")
    ax.set_title("实验二：预测考试成绩 —— 6 个网络平均只差 1.0 分（含 3 个考卷掺了噪声的）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e2_accuracy.png"), dpi=130)
    plt.close(fig)
    print("E2 fig:", list(zip(names, preds, trues)))


# ---------- Fig 3: E3 backdoor signature (pathwise influence) ----------

def parse_e3_weights(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    out, i = {}, 0
    while i < len(lines):
        m = re.search(r"(W\d) \((\d+) rows x (\d+) cols\)", lines[i])
        if m:
            name, r, c = m.group(1), int(m.group(2)), int(m.group(3))
            out[name] = np.array([[float(v) for v in lines[j].split()] for j in range(i + 1, i + 1 + r)])
            i += r + 1
        else:
            i += 1
    return out


def fig3():
    truth = json.load(open(os.path.join(ROOT, "secret", "e3_truth.json"), encoding="utf-8"))
    bd_file = next(k for k, v in truth.items() if v["condition"] == "backdoor")
    clean_file = next(k for k, v in truth.items() if v["condition"] == "clean")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for ax, fname, tag in [(axes[0], bd_file, "有后门的网络"), (axes[1], clean_file, "干净的网络")]:
        w = parse_e3_weights(os.path.join(ROOT, "subjects", "e3", fname))
        W1, W2, W3 = w["W1"], w["W2"], w["W3"]
        path_h = (np.abs(W2).T @ np.abs(W3).T).ravel()
        infl = (np.abs(W1).T @ path_h).reshape(12, 12)
        im = ax.imshow(infl, cmap="hot", interpolation="nearest")
        ax.set_title(f"{tag}（{fname}）", fontsize=13)
        ax.set_xticks(range(0, 12, 2)), ax.set_yticks(range(0, 12, 2))
        if truth[fname]["condition"] == "backdoor":
            for (r, c) in [(0, 0), (0, 11), (11, 0)]:
                circ = plt.Circle((c, r), 0.75, color="cyan", fill=False, lw=2.5)
                ax.add_patch(circ)
            flat = infl.ravel()
            ranks = sorted(int(np.argsort(flat)[::-1].tolist().index(t)) for t in [0, 11, 132])
            print(f"E3 {fname}: trigger influence ranks = {ranks} (of 144)")
    fig.colorbar(im, ax=axes, shrink=0.75, label="该像素到输出的通路影响力")
    fig.suptitle("实验三考题线索：每个像素通往输出的通路影响力——后门触发像素恰是全图最亮的三个", fontsize=14)
    fig.savefig(os.path.join(FIG, "e3_backdoor.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------- Fig 4/5: sample grids ----------

def grid_figure(panels, fname, suptitle=""):
    """panels: list of (title, npy-path). Each shows 8 samples in 2x4."""
    from PIL import Image as I, ImageDraw, ImageFont
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
    cols, scale, pad = 4, 22, 3
    cell = 12 * scale + pad
    W = cols * cell + pad
    H = len(panels) * (2 * cell + pad + 34) + pad
    img = I.new("RGB", (W, H), (24, 24, 24))
    d = ImageDraw.Draw(img)
    y = pad
    for title, path in panels:
        d.text((pad + 4, y + 8), title, fill=(240, 240, 240), font=font)
        y += 34
        xs = np.load(os.path.join(ROOT, path))
        for k in range(8):
            arr = (xs[k].reshape(12, 12).clip(0, 1) * 255).astype(np.uint8)
            tile = I.fromarray(arr).resize((12 * scale, 12 * scale), I.NEAREST)
            r, c = divmod(k, 4)
            img.paste(tile.convert("RGB"), (pad + c * cell, y + r * cell))
        y += 2 * cell + pad
    img.save(os.path.join(FIG, fname))
    print("saved", fname)


def fig4():
    panels = [
        ("参照：训练好的\"画圆\"模型（独立采样 8 张）", "assets/samples_circle.npy"),
        ("参照：训练好的\"画方块\"模型", "assets/samples_square.npy"),
        ("混合 25% 方 + 75% 圆 —— 实测：88% 是中间形态", "assets/samples_blend_25.npy"),
        ("混合 50% + 50% —— 实测：81% 像圆，且几乎总画同一个圆！", "assets/samples_blend_50.npy"),
        ("混合 75% 方 + 25% 圆 —— 实测：78% 是中间形态，0% 纯方块", "assets/samples_blend_75.npy"),
    ]
    grid_figure(panels, "e4_blends.png", "实验四实测")


def fig5():
    panels = [
        ("上限参照：训练 4 万步的三角模型（IoU 0.927）", "assets/samples_triangle.npy"),
        ("GLM-5.3 手写权重的三角模型（IoU 0.889，达上限 96%）", "results/e5_taskA_samples.npy"),
        ("上限参照：训练的\"左上角圆\"模型（笔迹越界近两成）", "assets/samples_quad_circle.npy"),
        ("GLM-5.3 手写权重：99% 笔迹关在左上角（比参照还干净）", "results/e5_taskB_samples.npy"),
    ]
    grid_figure(panels, "e5_write.png", "实验五实测")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("ALL FIGURES DONE ->", FIG)
