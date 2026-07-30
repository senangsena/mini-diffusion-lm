# mini-diffusion-lm

A minimal masked diffusion language model implemented from scratch in PyTorch.

TraceRL の論文 [Revolutionizing Reinforcement Learning Framework for Diffusion Large
Language Models](https://arxiv.org/abs/2509.06949)（公式実装:
https://github.com/Gen-Verse/dLLM-RL ）を読んで理解した内容を、
MacBook Air (M3, MPS) 上で数分で回る規模に縮めて確かめるためのコード。
**論文の再現実験ではなく、理解の検証を目的とした縮小実験である。**

## 何をしているか

* 層数4・d_model=128・約 80 万パラメータの双方向 Transformer をマスク拡散で学習する
* ブロック単位の半自己回帰サンプリング（`block_length`, `denoising_steps` をパラメータ化）
* 公式実装の `generate.py` と同じ構造の remasking 戦略
  （`low_confidence_static` / `low_confidence_dynamic` / `sequential`）
* 3つの学習目的の比較（完全ランダムマスク / ブロック半自己回帰 / モデル自身の生成順序）

## タスク

長さ12の数字列を入力し、長さ12の数字列を出力する2つのトイタスク。

| タスク | 内容 | 正解 |
|---|---|---|
| `sort` | 入力を昇順に並べ替える | 1通り |
| `branch` | `y = x` か `y_i = (x_i + 5) mod 10` のどちらか。系列全体で揃っている必要がある | 2通り |

`branch` は、各位置を独立に選ぶと必ず壊れるように作ってある。
並列デコードが失敗する仕組みを最小構成で取り出したもの。

## 使い方

```bash
python3 -m venv .venv && .venv/bin/pip install torch matplotlib numpy
.venv/bin/python run_experiments.py    # 実験一式（MPS で約40分）→ results/results.json
.venv/bin/python figures_exp.py        # 図の生成 → ../figures/
.venv/bin/python figures_concept.py    # レポート第2節用の概念図
```

## ファイル

| ファイル | 内容 |
|---|---|
| `data.py` | トイタスクのデータ生成と正解判定 |
| `model.py` | 双方向 Transformer |
| `train.py` | 3つの学習目的（`random` / `semiar` / `trace`） |
| `sample.py` | ブロック半自己回帰サンプリングとトレース記録 |
| `run_experiments.py` | 実験一式 |
| `figures_exp.py`, `figures_concept.py` | 図の生成 |
