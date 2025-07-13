
# Superscale / SummonScale – Implementation Plan (v0.1)

> **Purpose**  
> Provide a single markdown spec that an AI coding‑agent can consume to bootstrap the SDK/CLI/GUI.

---

## 1. Project Snapshot

| Item | Value |
|------|-------|
| **Library name** | `superscale`  (or `summonscale`, pick one and search/replace) |
| **Core models (6)** | TSDSR, HiT‑SR, VARSR, OSEDiff, OFTSR, BPOSR |
| **Primary backend** | PyTorch 2.3   (+ optional Diffusers pipeline wrappers) |
| **License** | Apache‑2.0 (code) — ensure each model weight is compatible |

---

## 2. Directory Skeleton

```text
superscale/
 ┣ superscale/
 ┃ ┣ __init__.py          # up(), call_god(), summon(), dismiss()
 ┃ ┣ base_pipeline.py     # SuperResolutionPipeline ABC
 ┃ ┣ registry.py          # MODEL_REGISTRY dict
 ┃ ┣ hub_utils.py         # HF hub download helpers
 ┃ ┣ summoner.py          # Summoner (LRU cache / “神殿”)
 ┃ ┣ cli.py               # superscale up ...
 ┃ ┗ backends/
 ┃    ┣ native/           # raw‐ckpt wrappers
 ┃    ┗ diffusers/        # convert‑to‑diffusers + pipelines
 ┣ examples/
 ┃ ┗ community/           # PR target for diffusers
 ┣ tests/                 # pytest suite
 ┣ scripts/               # convert_to_diffusers.py etc.
 ┗ pyproject.toml         # hatchling / poetry
```

---

## 3. Public API

```python
import superscale as ss

hr = ss.up("img.jpg", preset="Hermes", scale=4)
# or explicit
pipe = ss.summon("Athena", device="cuda")
hr2  = pipe(img, scale=8)
ss.dismiss("Athena")
```

### CLI

```bash
superscale up cat.png -p Hermes -s 4          # ワンライナー
superscale dismiss Hermes                     # 解放
```

---

## 4. Summoner – Caching & “召喚” Metaphor

* Located in `superscale/summoner.py`  
* Singleton `Summoner` with:  
  * `_stable: OrderedDict[str, Pipeline]` — LRU “神殿”  
  * `summon(name, **kw)` – load or fetch from cache  
  * `dismiss(name|None)` – move to CPU & `del`  
* `_max_residents` (default 3) controls concurrent VRAM usage.

---

## 5. Dependency Policy

* **Core**: `torch>=2.2,<2.4`, `Pillow>=10.0`
* **Extras**  
  * tsdsr → `diffusers>=0.31,<0.33`
  * hitsr → `timm>=0.9,<0.10`
  * varsr → `transformers>=4.42,<4.44`
  * oftsr → `diffusers>=0.31,<0.33`
  * bposr → `pytorch-lightning>=2.2,<2.4`
* Enable with `pip install superscale[tsdsr]` etc.

---

## 6. Model Wrapping Strategy

1. **v0.1 – native PyTorch ckpt**  
   * Minimal wrapper in `backends/native/<model>.py`
2. **v0.2 – optional Diffusers**  
   * `scripts/convert_to_diffusers.py`  
   * Push weights to `hf.co/superscale/<model>`  
   * Provide `backends/diffusers/<pipeline>.py`
3. `registry.py` chooses backend based on availability.

---

## 7. Progress / Magic‑Circle UX

* CLI: ASCII rune spinner (see `utils/spinner.py`)
* GUI: SVG rotating circle until `pipe(...)` returns.
* After completion, pixel sprite → high‑detail “Divine Form” image swap.

---

## 8. Tests (PyTest)

```python
def test_basic_upscale():
    import PIL, superscale as ss
    lr = PIL.Image.new("RGB", (64,64), "gray")
    hr = ss.up(lr, preset="Hermes", scale=4, device="cpu")
    assert hr.size == (256,256)
```

* Must pass `pip check` & `ruff`.

---

## 9. Roadmap for AI Agent Tasks

| Order | Task ID | Description |
|-------|---------|-------------|
| 1 | init‑repo | create dirs, pyproject, basic test pass |
| 2 | implement‑summoner | caching singleton & API glue |
| 3 | add‑tsdsr‑native | first model wrapper, pass test |
| 4 | enable‑extras | extras_require & CI matrix |
| 5 | diffusers‑convert | script + diffusers pipeline |
| 6 | PR‑to‑diffusers | push to `examples/community` |
| 7 | GUI‑prototype | Gradio Blocks with summon animation |

---

### End‑of‑File


---

## 10. Naming Note – `load()` vs `summon()`

* **`load()`** — primary, straightforward name exposed in README/CLI docs.  
  ```python
  pipe = superscale.load("Hermes", device="cuda")
  ```
* **`summon()`** — exact alias kept for the “pixel‑god / 召喚” world‑building.  
  ```python
  pipe = superscale.summon("Hermes")   # identical to load()
  ```
* Implementation:  
  ```python
  def load(name, **kw):
      return Summoner.summon(name, **kw)

  summon = load          # alias
  call_god = load        # optional fun alias used in examples
  ```
* **CLI**  
  ```bash
  superscale load img.png -p Hermes
  superscale summon img.png -p Hermes   # same
  ```
* Docstrings and tutorials should mention that both point to the same function, allowing beginners (who expect *load*) and advanced users (who enjoy the summon metaphor) to coexist.

