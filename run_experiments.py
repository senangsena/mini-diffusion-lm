"""Phase 3 の実験をすべて走らせて results/*.json に保存する.

  E1: 学習が収束すること（3-A の完了条件）
  E2: サンプリング戦略の比較（3-B）
  E3: 学習目的の比較（ランダムマスク / 半自己回帰 / トレース整合）
"""

import json
import os
import time

import torch

from data import PROMPT_LEN, TASKS, is_correct, local_valid_rate, make_test_set, L
from model import TinyDLM
from sample import generate
from train import DEVICE, train

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CKPT = os.path.join(HERE, "ckpt")
os.makedirs(RES, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)

TRAIN_STEPS = 3000
TRACE_STEPS = 1500
BATCH = 128
N_TEST = 512
EVAL_REPEAT = 3          # 温度1.0のサンプリングを3回繰り返して平均±標準偏差を出す
SEEDS = (0, 1, 2)


def evaluate(model, task, test, repeat=EVAL_REPEAT, **kw):
    accs, locals_, nfes = [], [], []
    for r in range(repeat):
        torch.manual_seed(12345 + r)
        out = generate(model, test, **kw)
        pred = out.tokens[:, PROMPT_LEN:]
        accs.append(is_correct(task, test, pred).float().mean().item())
        locals_.append(local_valid_rate(task, test, pred).mean().item())
        nfes.append(out.nfe)
    n = len(accs)
    mean = sum(accs) / n
    std = (sum((a - mean) ** 2 for a in accs) / n) ** 0.5
    return {"acc": mean, "acc_std": std, "local": sum(locals_) / n, "nfe": sum(nfes) / n}


def get_model(task, objective, seed, base=None):
    path = os.path.join(CKPT, f"{task}_{objective}_s{seed}.pt")
    if os.path.exists(path):
        m = TinyDLM().to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        hist = json.load(open(path + ".json"))
        return m, hist
    steps = TRACE_STEPS if objective == "trace" else TRAIN_STEPS
    m, hist = train(task, objective, steps=steps, batch_size=BATCH, seed=seed,
                    block_size=4, base_model=base, log_every=500)
    torch.save(m.state_dict(), path)
    json.dump(hist, open(path + ".json", "w"))
    return m, hist


def main():
    t0 = time.time()
    results = {"config": {
        "train_steps": TRAIN_STEPS, "trace_steps": TRACE_STEPS, "batch_size": BATCH,
        "n_test": N_TEST, "eval_repeat": EVAL_REPEAT, "seeds": list(SEEDS),
        "response_length": L, "device": str(DEVICE),
        "params": TinyDLM().num_params(),
    }}

    tests = {t: make_test_set(t, N_TEST) for t in TASKS}

    # ---------- E1 / E3: 3つの学習目的 ----------
    curves, obj_table = {}, {}
    for task in TASKS:
        for seed in SEEDS:
            base, hist = get_model(task, "random", seed)
            curves[f"{task}_random_s{seed}"] = hist
            semi, h2 = get_model(task, "semiar", seed)
            curves[f"{task}_semiar_s{seed}"] = h2
            tr, h3 = get_model(task, "trace", seed, base=base)
            curves[f"{task}_trace_s{seed}"] = h3
            for name, m in (("random", base), ("semiar", semi), ("trace", tr)):
                for setting, kw in (
                    ("static_B4_s4", dict(block_length=4, denoising_steps=4,
                                          strategy="low_confidence_static", temperature=1.0)),
                    ("static_B12_s3", dict(block_length=12, denoising_steps=3,
                                           strategy="low_confidence_static", temperature=1.0)),
                    ("dynamic_B4_t0.9", dict(block_length=4, denoising_steps=4,
                                             strategy="low_confidence_dynamic",
                                             confidence_threshold=0.9, temperature=1.0)),
                ):
                    key = f"{task}|{name}|{setting}"
                    obj_table.setdefault(key, []).append(
                        evaluate(m, task, tests[task], **kw))
            print(f"[E3] {task} seed={seed} done ({time.time()-t0:.0f}s)", flush=True)
    results["curves"] = curves
    results["objective_table"] = obj_table

    # 以降のサンプリング実験は seed=0 のランダムマスクモデルを使う
    models = {t: get_model(t, "random", 0)[0] for t in TASKS}

    # ---------- E2-a: ブロック内を一気に確定させる（block_length を振る） ----------
    sweep_block = {}
    for task in TASKS:
        rows = []
        for B in (1, 2, 3, 4, 6, 12):
            r = evaluate(models[task], task, tests[task], block_length=B,
                         denoising_steps=1, strategy="low_confidence_static",
                         temperature=1.0)
            r["block_length"] = B
            rows.append(r)
        sweep_block[task] = rows
        print(f"[E2-a] {task} done ({time.time()-t0:.0f}s)", flush=True)
    results["sweep_block_1step"] = sweep_block

    # ---------- E2-b: 1ブロック(=応答全体)で denoising_steps を振る ----------
    sweep_steps = {}
    for task in TASKS:
        rows = []
        for S in (1, 2, 3, 4, 6, 12):
            r = evaluate(models[task], task, tests[task], block_length=12,
                         denoising_steps=S, strategy="low_confidence_static",
                         temperature=1.0)
            r["denoising_steps"] = S
            rows.append(r)
        sweep_steps[task] = rows
        print(f"[E2-b] {task} done ({time.time()-t0:.0f}s)", flush=True)
    results["sweep_steps"] = sweep_steps

    # ---------- E2-c: dynamic の閾値を振る ----------
    sweep_thr = {}
    for task in TASKS:
        rows = []
        for thr in (0.5, 0.7, 0.9, 0.95, 0.99):
            for B in (4, 12):
                r = evaluate(models[task], task, tests[task], block_length=B,
                             denoising_steps=B, strategy="low_confidence_dynamic",
                             confidence_threshold=thr, temperature=1.0)
                r["threshold"] = thr
                r["block_length"] = B
                rows.append(r)
        sweep_thr[task] = rows
        print(f"[E2-c] {task} done ({time.time()-t0:.0f}s)", flush=True)
    results["sweep_threshold"] = sweep_thr

    # ---------- E2-d: 信頼度の定義（論文 max か、公式実装のサンプル確率か） ----------
    conf_mode = {}
    for task in TASKS:
        rows = {}
        for mode in ("max", "sampled"):
            rows[mode] = evaluate(models[task], task, tests[task], block_length=12,
                                  denoising_steps=4, strategy="low_confidence_static",
                                  temperature=1.0, confidence_mode=mode)
        conf_mode[task] = rows
    results["confidence_mode"] = conf_mode

    # ---------- E2-e: 生成トラジェクトリの可視化用データ ----------
    traj = {}
    for task in TASKS:
        torch.manual_seed(999)
        out = generate(models[task], tests[task], block_length=12, denoising_steps=12,
                       strategy="low_confidence_static", temperature=1.0)
        fu = out.first_unmask.float()
        traj[task] = {
            "mean_first_unmask": fu.mean(dim=0).tolist(),
            "std_first_unmask": fu.std(dim=0).tolist(),
            "example": out.first_unmask[:24].tolist(),
        }
        torch.manual_seed(999)
        outd = generate(models[task], tests[task], block_length=12, denoising_steps=12,
                        strategy="low_confidence_dynamic", confidence_threshold=0.9,
                        temperature=1.0)
        traj[task]["dynamic_mean_first_unmask"] = outd.first_unmask.float().mean(dim=0).tolist()
        traj[task]["dynamic_example"] = outd.first_unmask[:24].tolist()
        traj[task]["dynamic_step_sizes"] = outd.step_sizes
    results["trajectory"] = traj

    # ---------- 参考: 貪欲デコード ----------
    greedy = {}
    for task in TASKS:
        greedy[task] = {
            f"B{B}_s{S}": evaluate(models[task], task, tests[task], block_length=B,
                                   denoising_steps=S, strategy="low_confidence_static",
                                   temperature=0.0, repeat=1)
            for B, S in ((12, 1), (12, 12), (4, 4))
        }
    results["greedy"] = greedy

    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"done in {time.time()-t0:.0f}s -> {RES}/results.json")


if __name__ == "__main__":
    main()
