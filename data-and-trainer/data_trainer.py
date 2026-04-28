import re
from dataclasses import dataclass
import json
from itertools import combinations
import random
import torch
import faiss
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer #, SFTConfig
from sentence_transformers import SentenceTransformer

# ─── Constants ────────────────────────────────────────────────────────────────

MODEL_ID     = "microsoft/Phi-3-mini-4k-instruct"
DATASET_PATH = "unity_actions_dataset_combo.json"
OUTPUT_DIR   = "./unity-action-model"
MAX_SEQ_LEN  = 256  # shorter now — no scene context needed
EMBED_MODEL  = "all-MiniLM-L6-v2"

# ─── Dataset Generation ───────────────────────────────────────────────────────

ACTIONS = [
    # (function, required_args, optional_args)
    ("Transform.Translate",  ["x", "y", "z"],              ["space"]),
    ("Transform.Rotate",     ["x", "y", "z"],              ["space"]),
    ("Transform.Scale",      ["scale"],                     ["x", "y", "z"]),
    ("Transform.SetPosition",["x", "y", "z"],              ["space"]),
    ("GameObject.SetActive", ["active"],                    []),
    ("GameObject.Destroy",   [],                            ["delay"]),
    ("Light.SetIntensity",   ["intensity"],                 []),
    ("Light.SetColor",       ["color"],                     []),
    ("Rigidbody.AddForce",   ["x", "y", "z"],              ["mode"]),
    ("Rigidbody.SetVelocity",["x", "y", "z"],              []),
    ("Material.SetColor",    ["color"],                     ["property"]),
    ("Material.SetAlpha",    ["alpha"],                     []),
    ("Animation.Play",       ["clip"],                      []),
    ("Camera.SetFOV",        ["fov"],                       []),
    ("Camera.SetBackground", ["color"],                     []),
]

# Seed phrases per action — model learns to map these patterns to structured output
SEED_TEMPLATES = {
    "Transform.Translate": [
        "move it {dir} by {n} units",
        "push {n} units to the {side}",
        "nudge {adv} along {axis}",
        "shift {n} on the {axis} axis",
        "slide it {dir} {n} meters",
        "offset by {n} in {dir}",
        "translate {n} units {dir}",
        "bump {adv} in the {dir} direction",
    ],
    "Transform.Rotate": [
        "rotate {n} degrees around {axis}",
        "spin it {n} degrees on {axis}",
        "tilt {n} degrees {dir}",
        "turn {n} degrees clockwise",
        "yaw {n} degrees",
        "pitch down {n} degrees",
        "roll {n} degrees",
        "orient it {n} degrees on the y axis",
    ],
    "Transform.Scale": [
        "scale it up by {n}x",
        "make it {n} times bigger",
        "shrink it to {n}x",
        "double its size",
        "halve the scale",
        "resize uniformly to {n}",
        "stretch x by {n} and y by {m}",
        "squash it on the y axis",
    ],
    "Transform.SetPosition": [
        "place it at ({x}, {y}, {z})",
        "teleport to position {x} {y} {z}",
        "set world position to {x} {y} {z}",
        "move to coordinates {x} {y} {z}",
        "snap to origin",
        "put it at the center",
        "position at x={x} y={y} z={z}",
        "reposition to {x} {y} {z}",
    ],
    "GameObject.SetActive": [
        "hide it",
        "show it",
        "make it invisible",
        "toggle it off",
        "enable the object",
        "disable it",
        "turn it on",
        "deactivate it",
    ],
    "GameObject.Destroy": [
        "destroy it",
        "delete it",
        "remove it from the scene",
        "destroy it after {n} seconds",
        "delete in {n} seconds",
        "blow it up",
        "get rid of it",
        "despawn it",
    ],
    "Light.SetIntensity": [
        "set intensity to {n}",
        "make it brighter",
        "dim the light",
        "set light strength to {n}",
        "turn it up to {n}",
        "reduce brightness to {n}",
        "max out the light",
        "make it barely visible",
    ],
    "Light.SetColor": [
        "make the light red",
        "set light to blue",
        "change light color to {hex}",
        "warm up the light",
        "cool down the light",
        "make it green",
        "set light color to white",
        "give it a purple tint",
    ],
    "Rigidbody.AddForce": [
        "launch it upward",
        "push it forward",
        "apply {n} force upward",
        "blast it to the right",
        "apply impulse {x} {y} {z}",
        "continuously push {dir}",
        "velocity change by {n} upward",
        "knock it sideways",
    ],
    "Rigidbody.SetVelocity": [
        "set velocity to {x} {y} {z}",
        "move at {n} units per second forward",
        "fling it upward at {n} speed",
        "stop it",
        "set speed to {n} on x",
        "make it drift right at {n}",
        "zero out velocity",
        "send it flying at {x} {y} {z}",
    ],
    "Material.SetColor": [
        "paint it red",
        "change color to {hex}",
        "make it blue",
        "set material color to {hex}",
        "recolor to green",
        "make it look golden",
        "give it a dark appearance",
        "set emission color to {hex}",
    ],
    "Material.SetAlpha": [
        "make it transparent",
        "set opacity to {n}",
        "make it semi-transparent",
        "fade it out",
        "make it fully opaque",
        "set alpha to {n}",
        "ghost effect",
        "reduce visibility to {n}",
    ],
    "Animation.Play": [
        "play the idle animation",
        "play run animation",
        "trigger the attack clip",
        "start the death animation",
        "play {clip} clip",
        "animate it opening",
        "trigger victory pose",
        "play the explode animation",
    ],
    "Camera.SetFOV": [
        "set field of view to {n}",
        "zoom in",
        "zoom out",
        "widen the camera view",
        "narrow fov to {n} degrees",
        "set fov {n}",
        "fish-eye effect",
        "telephoto look",
    ],
    "Camera.SetBackground": [
        "set sky to black",
        "background color to {hex}",
        "make the background blue",
        "dark background",
        "set clear color to {hex}",
        "white background",
        "sunset background color",
        "midnight sky",
    ],
    "FlyToTarget": [
        ("show it")
    ]
}

# ─── Argument Helpers ─────────────────────────────────────────────────────────

def _n():     return round(random.uniform(1, 20), 1)
def _big():   return round(random.uniform(200, 1000), 0)
def _vec3():  return {"x": round(random.uniform(-10,10),1), "y": round(random.uniform(-10,10),1), "z": round(random.uniform(-10,10),1)}
def _pos_args(): return {"x": round(random.uniform(-20,20),2), "y": round(random.uniform(0,10),2), "z": round(random.uniform(-20,20),2)}
def _rotate_args(yaw=False, pitch=False, roll=False):
    deg = round(random.uniform(-180, 180), 1)
    if yaw:   return {"x": 0, "y": deg, "z": 0}
    if pitch: return {"x": deg, "y": 0, "z": 0}
    if roll:  return {"x": 0, "y": 0, "z": deg}
    axis = random.choice(["x","y","z"])
    return {"x": deg if axis=="x" else 0, "y": deg if axis=="y" else 0, "z": deg if axis=="z" else 0}

def _force_args(): return {"x": round(random.uniform(-500,500),1), "y": round(random.uniform(0,800),1), "z": round(random.uniform(-500,500),1)}
def _force_dir():
    d = random.choice(["x","y","z"])
    v = round(random.uniform(100, 600), 1)
    return {"x": v if d=="x" else 0, "y": v if d=="y" else 0, "z": v if d=="z" else 0}

def _dir_vec():
    dirs = {"right":(1,0,0),"left":(-1,0,0),"up":(0,1,0),"down":(0,-1,0),"forward":(0,0,1),"back":(0,0,-1)}
    name, (x,y,z) = random.choice(list(dirs.items()))
    n = round(random.uniform(0.5, 15), 1)
    return n, x*n, y*n, z*n

def _side_vec():
    return _dir_vec()

def _axis_vec():
    axis = random.choice(["x","y","z"])
    n = round(random.uniform(0.5, 10), 1)
    return n, n if axis=="x" else 0, n if axis=="y" else 0, n if axis=="z" else 0

def _translate_args(n, x, y, z):
    return {"x": round(x,2), "y": round(y,2), "z": round(z,2)}

def _random_hex():
    return "#{:06X}".format(random.randint(0, 0xFFFFFF))

# ─── Paraphrase Expansion ─────────────────────────────────────────────────────
# Expand each seed with synonym substitutions — no API needed

SYNONYM_SWAPS = {
    "move":      ["shift", "translate", "push", "displace", "nudge"],
    "rotate":    ["spin", "turn", "tilt", "pivot", "yaw", "orbit"],
    "scale":     ["resize", "size", "stretch", "grow", "enlarge"],
    "hide":      ["disable", "deactivate", "turn off", "make invisible"],
    "activate":  ["enable", "turn on", "make visible"],
    "show":      ["focus", "get closer to", "close view"],
    "destroy":   ["delete", "remove", "despawn", "kill"],
    "set":       ["assign", "change", "update", "configure"],
    "make":      ["turn", "render", "set"],
    "play":      ["trigger", "start", "run", "execute"],
    "brighter":  ["more intense", "stronger", "higher intensity"],
    "dimmer":    ["less intense", "weaker", "lower intensity"],
}

def apply_synonym_swap(text: str) -> str:
    words = text.split()
    for i, word in enumerate(words):
        w = word.lower().rstrip(".,")
        if w in SYNONYM_SWAPS and random.random() < 0.4:
            words[i] = random.choice(SYNONYM_SWAPS[w])
    return " ".join(words)

def expand_template(template: str, action: str) -> tuple[str, dict]:
    # Generate all random values once
    n    = round(random.uniform(1, 15), 1)
    x    = round(random.uniform(-10, 10), 2)
    y    = round(random.uniform(0, 10), 2)
    z    = round(random.uniform(-10, 10), 2)
    m    = round(random.uniform(1, 4), 1)
    hex_ = _random_hex()
    adv  = random.choice(["slightly", "gently", "a bit", "sharply"])
    axis = random.choice(["x", "y", "z"])
    dir_ = random.choice(["right", "left", "up", "down", "forward", "back"])
    side = random.choice(["right", "left"])
    clip = random.choice(["Idle", "Run", "Attack", "Jump", "Dance", "Death", "Open"])

    # Shared context — single source of truth for both instruction and args
    ctx = dict(n=n, x=x, y=y, z=z, m=m, hex=hex_,
               adv=adv, axis=axis, dir=dir_, side=side, clip=clip)

    instruction = template.format(**ctx)
    instruction = apply_synonym_swap(instruction)

    # Args built from the same ctx values
    args = build_args(action, ctx)

    return instruction, args


def build_args(action: str, ctx: dict) -> dict:
    dir_map = {
        "right":   ( ctx["n"],  0,         0        ),
        "left":    (-ctx["n"],  0,         0        ),
        "up":      ( 0,         ctx["n"],  0        ),
        "down":    ( 0,        -ctx["n"],  0        ),
        "forward": ( 0,         0,         ctx["n"] ),
        "back":    ( 0,         0,        -ctx["n"] ),
    }
    axis_map = {
        "x": (ctx["n"], 0, 0),
        "y": (0, ctx["n"], 0),
        "z": (0, 0, ctx["n"]),
    }

    if action == "Transform.Translate":
        dx, dy, dz = dir_map.get(ctx["dir"], (ctx["x"], ctx["y"], ctx["z"]))
        return {"x": dx, "y": dy, "z": dz}

    if action == "Transform.Rotate":
        rx, ry, rz = axis_map.get(ctx["axis"], (ctx["x"], ctx["y"], ctx["z"]))
        return {"x": rx, "y": ry, "z": rz}

    if action == "Transform.Scale":
        return {"scale": ctx["n"]}

    if action == "Transform.SetPosition":
        return {"x": ctx["x"], "y": ctx["y"], "z": ctx["z"]}

    if action == "GameObject.SetActive":
        return {"active": random.choice([True, False])}

    if action == "GameObject.Destroy":
        return {"delay": ctx["n"]}

    if action in ("Light.SetIntensity",):
        return {"intensity": ctx["n"]}

    if action in ("Light.SetColor", "Material.SetColor", "Camera.SetBackground"):
        return {"color": ctx["hex"]}

    if action == "Rigidbody.AddForce":
        dx, dy, dz = dir_map.get(ctx["dir"], (ctx["x"], ctx["y"], ctx["z"]))
        return {"x": dx * 40, "y": dy * 40, "z": dz * 40, "mode": "Impulse"}

    if action == "Rigidbody.SetVelocity":
        return {"x": ctx["x"], "y": ctx["y"], "z": ctx["z"]}

    if action == "Material.SetAlpha":
        return {"alpha": round(ctx["n"] / 15, 2)}  # normalize n to 0–1

    if action == "Animation.Play":
        return {"clip": ctx["clip"]}

    if action == "Camera.SetFOV":
        return {"fov": int(ctx["n"] * 4 + 30)}  # map 1–15 → 34–90

    return {}

# ─── Build Dataset ────────────────────────────────────────────────────────────

def build_dataset(examples_per_action: int = 80) -> list[dict]:
    dataset = []
    for action, templates in SEED_TEMPLATES.items():
        per_template = max(1, examples_per_action // len(templates))
        for template in templates:
            for _ in range(per_template):
                try:
                    instruction, args = expand_template(template, action)
                    dataset.append({
                        "instruction": instruction,
                        "output": [{"function": action, "args": args}]
                    })
                except Exception:
                    pass
    random.shuffle(dataset)
    print(len(dataset))
    new_dataset = []
    for r in range(2, 5):  # combine 2 to 4 steps
        cnt = 0
        for combo in combinations(dataset, r):
            join_words = ["and", "then", "."]
            # Merge instructions
            merged_instruction = f" {random.choice(join_words)} ".join([item["instruction"] for item in combo])
            
            # Merge outputs (sequence of functions)
            merged_output = []
            for item in combo:
                merged_output.extend(item["output"])  # append function steps
            
            # Append new sample
            new_dataset.append({
                "instruction": merged_instruction,
                "output": merged_output
            })
            cnt += 1
            if cnt >= 3000:
                break
    print(len(new_dataset))
    dataset.extend(new_dataset)
    random.shuffle(dataset)
    print(f"Generated {len(dataset)} examples across {len(SEED_TEMPLATES)} actions")
    return dataset

def save_dataset(dataset: list[dict], path: str = DATASET_PATH):
    Path(path).write_text(json.dumps(dataset, indent=2))
    print(f"Saved to {path}")

# ─── Format for Training ──────────────────────────────────────────────────────
# Notice: no target — FAISS handles that at runtime

def format_example(ex: dict) -> str:
    return (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Response:\n{json.dumps(ex['output'])}"
    )

# ─── Fine-Tune ────────────────────────────────────────────────────────────────

def train(dataset_path: str = DATASET_PATH):
    raw = json.loads(Path(dataset_path).read_text())
    formatted = [{"text": format_example(ex)} for ex in raw]

    split = int(len(formatted) * 0.9)
    train_ds = Dataset.from_list(formatted[:split])
    eval_ds  = Dataset.from_list(formatted[split:])
    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype = "auto",
        trust_remote_code = False
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                        # smaller rank — simpler task now (no target prediction)
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=4,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        # dataset_text_field="text",
        # max_seq_length=MAX_SEQ_LEN
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        dataset_text_field="text",
        packing=True,
        max_seq_length=MAX_SEQ_LEN
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

# ─── FAISS Scene Index ────────────────────────────────────────────────────────

class SceneIndex:
    """
    Maintains a FAISS index of current scene objects.
    Call rebuild() whenever the scene changes.
    Query with find() to resolve natural language → object name.
    """

    def __init__(self):
        self.embedder   = SentenceTransformer(EMBED_MODEL)
        self.index      = None
        self.object_names: list[str] = []

    def rebuild(self, scene_objects: list[str]):
        """
        scene_objects: list of strings like
          ["WoodDoor_01", "SpotLight_Blue", "HeavyBoulder_03"]
        """
        self.object_names = scene_objects
        embeddings = self.embedder.encode(scene_objects, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine on normalized vecs
        self.index.add(embeddings)
        print(f"FAISS index built: {len(scene_objects)} objects")

    def find(self, query: str, threshold: float = 0.3) -> str | None:
        name, _ = self.find_with_score(query)
        return name
    
    def find_with_score(self, query: str, top_k: int = 1, threshold: float = 0.3) -> str | None:
        if self.index is None or not self.object_names:
            raise RuntimeError("Scene index is empty. Call rebuild() first.")

        vec = self.embedder.encode([query], convert_to_numpy=True)
        vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)

        scores, indices = self.index.search(vec, top_k)
        best_score = float(scores[0][0])
        best_idx   = indices[0][0]
        print("indices", indices)
        print("scores", scores)
        print("objects", self.object_names)
        if best_score < threshold:
            return None, best_score  # no confident match

        return self.object_names[best_idx], best_score

ACTION_PRIORITY = {
    "GameObject.SetActive":    1,  # visibility first
    "GameObject.Destroy":      1,
    "Transform.SetPosition":   2,  # position before physics
    "Transform.Translate":     3,
    "Transform.Rotate":        3,
    "Transform.Scale":         3,
    "Rigidbody.SetVelocity":   4,  # physics after positioning
    "Rigidbody.AddForce":      5,
    "Light.SetColor":          6,  # visuals after physics
    "Light.SetIntensity":      6,
    "Material.SetColor":       7,
    "Material.SetAlpha":       7,
    "Camera.SetFOV":           8,
    "Camera.SetBackground":    8,
    "Animation.Play":          9,  # animations last
}

# ─── Splitting patterns ───────────────────────────────────────────────────────

# Conjunctions and sequence words that separate sub-instructions
SPLIT_PATTERNS = [
    r'\s+and\s+then\s+',
    r'\s+then\s+',
    r'\s+after\s+that\s+',
    r'\s+afterwards\s+',
    r'\s+followed\s+by\s+',
    r'\s+and\s+also\s+',
    r'\s+also\s+',
    r',\s+and\s+',
    r'\s+and\s+',
    r';\s*',
    r',\s+',
]

COMPILED_SPLIT = re.compile(
    '|'.join(SPLIT_PATTERNS),
    flags=re.IGNORECASE
)

# Ambiguous "and" that should NOT split — same object, compound action
# e.g. "rotate and scale the cube" — these belong together
COMPOUND_ACTION_PAIRS = [
    ("rotate", "scale"),
    ("hide",   "disable"),
    ("show",   "enable"),
    ("move",   "rotate"),
    ("set",    "change"),
]

def _is_compound(left: str, right: str) -> bool:
    """Return True if two clauses likely refer to one compound action."""
    l, r = left.lower(), right.lower()
    for a, b in COMPOUND_ACTION_PAIRS:
        if (a in l and b in r) or (b in l and a in r):
            return True
    return False

# ─── Splitter ─────────────────────────────────────────────────────────────────

def split_instruction(instruction: str) -> list[str]:
    """
    Split a compound instruction into ordered sub-instructions.
    Merges back clauses that look like compound actions on the same object.
    """
    raw_parts = COMPILED_SPLIT.split(instruction.strip())
    parts = [p.strip() for p in raw_parts if p and p.strip()]

    if len(parts) <= 1:
        return parts

    # Merge compound pairs back together
    merged = [parts[0]]
    for i in range(1, len(parts)):
        if _is_compound(merged[-1], parts[i]):
            merged[-1] = merged[-1] + " and " + parts[i]
        else:
            merged.append(parts[i])

    return merged

# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class CommandResult:
    function:    str
    target:      str
    args:        dict
    instruction: str          # original sub-instruction that produced this
    priority:    int          # for ordering
    confidence:  float        # FAISS similarity score
    def to_dict(self) -> dict:
        return {
            "function":    self.function,
            "target":      self.target,
            "args":        self.args,
            "instruction": self.instruction,
            "priority":    self.priority,
            "confidence":  self.confidence,
        }


# ─── Inference Pipeline ───────────────────────────────────────────────────────
class UnityNLPPipeline:

    def __init__(self, model_dir: str = OUTPUT_DIR):
        from peft import PeftModel

        print("Loading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, model_dir)
        self.model.eval()
        self.scene_index = SceneIndex()
        print("✓ Pipeline ready")

    def update_scene(self, scene_objects: list[str]):
        self.scene_index.rebuild(scene_objects)

    # ── Single sub-instruction ─────────────────────────────────────────────

    def _predict_one(self, instruction: str) -> dict | None:
        """Predict function+args for a single atomic instruction."""
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded  = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = decoded.split("### Response:\n")[-1].strip()

        try:
            ret = json.loads(response)
            print(ret)
            return ret
        except json.JSONDecodeError:
            print(f"  ✗ Parse failed for: '{instruction}'\n    Raw: {response}")
            return None

    # ── Main entry point ───────────────────────────────────────────────────

    def predict(self, instruction: str) -> list[CommandResult]:
        """
        Split compound instruction → predict each sub-instruction →
        resolve targets via FAISS → sort by priority.

        Returns a list of CommandResult, ordered for Unity execution.
        """
        sub_instructions = split_instruction(instruction)
        print(f"\n→ Split into {len(sub_instructions)} sub-instruction(s):")
        for i, s in enumerate(sub_instructions):
            print(f"  {i+1}. '{s}'")

        results = []

        for sub in sub_instructions:
            # 1. Model predicts function + args
            action = self._predict_one(sub)
            if action is None or len(action) == 0:
                print(f"  ⚠ Skipping (parse failed): '{sub}'")
                continue

            fn = action[0].get("function")
            if not fn:
                print(f"  ⚠ Skipping (no function): '{sub}'")
                continue

            # 2. FAISS resolves target
            target, confidence = self.scene_index.find_with_score(sub)
            if target is None:
                print(f"  ⚠ Skipping (no target found): '{sub}'")
                continue

            # 3. Build result with priority
            priority = ACTION_PRIORITY.get(fn, 99)
            results.append(CommandResult(
                function=fn,
                target=target,
                args=action[0].get("args", {}),
                instruction=sub,
                priority=priority,
                confidence=confidence,
            ))
            print(f"  ✓ '{sub}' → {fn} on '{target}' (priority={priority}, conf={confidence:.2f})")

        # 4. Sort by priority (stable sort preserves original order within same priority)
        results.sort(key=lambda r: r.priority)

        return results

# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1 — generate dataset
    # dataset = build_dataset(examples_per_action=80)
    # save_dataset(dataset)

    # with open(DATASET_PATH, "r") as f:
    #     dataset = json.load(f)
    
    # # Step 2 — train
    # train()

    # Step 3 — test inference
    pipeline = UnityNLPPipeline()

    # Simulate a Unity scene snapshot arriving
    pipeline.update_scene([
        "WoodDoor_01",
        "SpotLight_Blue",
        "HeavyBoulder_03",
        "PlayerCharacter",
        "CeilingLamp_02",
        "GlassWall_Left",
    ])

    tests = [
        # "launch the boulder upward with a strong impulse",
        "dim the ceiling lamp and then make the glass wall transparent",
        "make the glass wall transparent",
        "play the idle animation on the player",
        "rotate the door 90 degrees on y",
        "set the spotlight to red",
    ]

    print("\n─── Inference Tests ───")
    for instruction in tests:
        result = pipeline.predict(instruction)
        print(f"\n> {instruction}")
        # print("result", result)
        print(f"  {json.dumps([r.to_dict() for r in result], indent=2)}")