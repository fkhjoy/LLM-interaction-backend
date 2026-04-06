import json
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
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from sentence_transformers import SentenceTransformer

# ─── Constants ────────────────────────────────────────────────────────────────

MODEL_ID     = "microsoft/Phi-3-mini-4k-instruct"
DATASET_PATH = "unity_actions_dataset.json"
OUTPUT_DIR   = "./unity-action-model"
MAX_SEQ_LEN  = 256  # shorter now — no scene context needed
EMBED_MODEL  = "all-MiniLM-L6-v2"

# ─── Dataset Generation ───────────────────────────────────────────────────────
# No Anthropic API — pure handcrafted + programmatic expansion

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
        ("move it {dir} by {n} units",          lambda: _translate_args(*_dir_vec())),
        ("push {n} units to the {side}",         lambda: _translate_args(*_side_vec())),
        ("nudge {adv} along {axis}",             lambda: _translate_args(*_axis_vec())),
        ("shift {n:.1f} on the {axis} axis",     lambda: _translate_args(*_axis_vec())),
        ("slide it {dir} {n} meters",            lambda: _translate_args(*_dir_vec())),
        ("offset by {n} in {dir}",               lambda: _translate_args(*_dir_vec())),
        ("translate {n} units {dir}",            lambda: _translate_args(*_dir_vec())),
        ("bump {adv} in the {dir} direction",    lambda: _translate_args(*_dir_vec())),
    ],
    "Transform.Rotate": [
        ("rotate {n} degrees around {axis}",     lambda: _rotate_args()),
        ("spin it {n} degrees on {axis}",        lambda: _rotate_args()),
        ("tilt {n} degrees {dir}",               lambda: _rotate_args()),
        ("turn {n:.1f} degrees clockwise",       lambda: _rotate_args(yaw=True)),
        ("yaw {n} degrees",                      lambda: _rotate_args(yaw=True)),
        ("pitch down {n} degrees",               lambda: _rotate_args(pitch=True)),
        ("roll {n} degrees",                     lambda: _rotate_args(roll=True)),
        ("orient it {n} degrees on the y axis",  lambda: _rotate_args(yaw=True)),
    ],
    "Transform.Scale": [
        ("scale it up by {n}x",                  lambda: {"scale": round(random.uniform(1.5, 4.0), 2)}),
        ("make it {n} times bigger",             lambda: {"scale": round(random.uniform(1.2, 5.0), 1)}),
        ("shrink it to {n:.1f}x",               lambda: {"scale": round(random.uniform(0.1, 0.9), 2)}),
        ("double its size",                      lambda: {"scale": 2.0}),
        ("halve the scale",                      lambda: {"scale": 0.5}),
        ("resize uniformly to {n}",              lambda: {"scale": round(random.uniform(0.5, 3.0), 1)}),
        ("stretch x by {n} and y by {m}",       lambda: {"x": round(random.uniform(1,3),1), "y": round(random.uniform(1,3),1), "z": 1}),
        ("squash it on the y axis",              lambda: {"x": 1, "y": round(random.uniform(0.1, 0.5), 2), "z": 1}),
    ],
    "Transform.SetPosition": [
        ("place it at ({x}, {y}, {z})",         lambda: _pos_args()),
        ("teleport to position {x} {y} {z}",    lambda: _pos_args()),
        ("set world position to {x} {y} {z}",   lambda: {**_pos_args(), "space": "world"}),
        ("move to coordinates {x} {y} {z}",     lambda: _pos_args()),
        ("snap to origin",                       lambda: {"x": 0, "y": 0, "z": 0}),
        ("put it at the center",                 lambda: {"x": 0, "y": 0, "z": 0}),
        ("position at x={x} y={y} z={z}",       lambda: _pos_args()),
        ("reposition to {x} {y} {z}",           lambda: _pos_args()),
    ],
    "GameObject.SetActive": [
        ("hide it",                              lambda: {"active": False}),
        ("show it",                              lambda: {"active": True}),
        ("make it invisible",                    lambda: {"active": False}),
        ("toggle it off",                        lambda: {"active": False}),
        ("enable the object",                    lambda: {"active": True}),
        ("disable it",                           lambda: {"active": False}),
        ("turn it on",                           lambda: {"active": True}),
        ("deactivate it",                        lambda: {"active": False}),
    ],
    "GameObject.Destroy": [
        ("destroy it",                           lambda: {}),
        ("delete it",                            lambda: {}),
        ("remove it from the scene",             lambda: {}),
        ("destroy it after {n:.1f} seconds",     lambda: {"delay": round(random.uniform(0.5, 5.0), 1)}),
        ("delete in {n} seconds",                lambda: {"delay": round(random.uniform(1, 10))}),
        ("blow it up",                           lambda: {}),
        ("get rid of it",                        lambda: {}),
        ("despawn it",                           lambda: {}),
    ],
    "Light.SetIntensity": [
        ("set intensity to {n:.1f}",             lambda: {"intensity": round(random.uniform(0.1, 8.0), 1)}),
        ("make it brighter",                     lambda: {"intensity": round(random.uniform(3.0, 8.0), 1)}),
        ("dim the light",                        lambda: {"intensity": round(random.uniform(0.1, 1.0), 2)}),
        ("set light strength to {n}",            lambda: {"intensity": round(random.uniform(0.5, 5.0), 1)}),
        ("turn it up to {n:.1f}",               lambda: {"intensity": round(random.uniform(2.0, 8.0), 1)}),
        ("reduce brightness to {n:.2f}",        lambda: {"intensity": round(random.uniform(0.1, 1.5), 2)}),
        ("max out the light",                    lambda: {"intensity": 8.0}),
        ("make it barely visible",               lambda: {"intensity": 0.1}),
    ],
    "Light.SetColor": [
        ("make the light red",                   lambda: {"color": "#FF0000"}),
        ("set light to blue",                    lambda: {"color": "#0000FF"}),
        ("change light color to {hex}",          lambda: {"color": _random_hex()}),
        ("warm up the light",                    lambda: {"color": "#FFA040"}),
        ("cool down the light",                  lambda: {"color": "#4080FF"}),
        ("make it green",                        lambda: {"color": "#00FF00"}),
        ("set light color to white",             lambda: {"color": "#FFFFFF"}),
        ("give it a purple tint",                lambda: {"color": "#8800FF"}),
    ],
    "Rigidbody.AddForce": [
        ("launch it upward",                     lambda: {"x": 0, "y": _big(), "z": 0, "mode": "Impulse"}),
        ("push it forward",                      lambda: {"x": 0, "y": 0, "z": _big(), "mode": "Force"}),
        ("apply {n} force upward",               lambda: {"x": 0, "y": _n(), "z": 0, "mode": "Force"}),
        ("blast it to the right",                lambda: {"x": _big(), "y": 0, "z": 0, "mode": "Impulse"}),
        ("apply impulse {x} {y} {z}",           lambda: {**_force_args(), "mode": "Impulse"}),
        ("continuously push {dir}",              lambda: {**_force_dir(), "mode": "Force"}),
        ("velocity change by {n} upward",        lambda: {"x": 0, "y": _n(), "z": 0, "mode": "VelocityChange"}),
        ("knock it sideways",                    lambda: {"x": _big(), "y": 0, "z": 0, "mode": "Impulse"}),
    ],
    "Rigidbody.SetVelocity": [
        ("set velocity to {x} {y} {z}",         lambda: _vec3()),
        ("move at {n} units per second forward", lambda: {"x": 0, "y": 0, "z": _n()}),
        ("fling it upward at {n} speed",         lambda: {"x": 0, "y": _n(), "z": 0}),
        ("stop it",                              lambda: {"x": 0, "y": 0, "z": 0}),
        ("set speed to {n} on x",               lambda: {"x": _n(), "y": 0, "z": 0}),
        ("make it drift right at {n}",           lambda: {"x": _n(), "y": 0, "z": 0}),
        ("zero out velocity",                    lambda: {"x": 0, "y": 0, "z": 0}),
        ("send it flying at {x} {y} {z}",       lambda: _vec3()),
    ],
    "Material.SetColor": [
        ("paint it red",                         lambda: {"color": "#FF0000"}),
        ("change color to {hex}",                lambda: {"color": _random_hex()}),
        ("make it blue",                         lambda: {"color": "#0000FF"}),
        ("set material color to {hex}",          lambda: {"color": _random_hex()}),
        ("recolor to green",                     lambda: {"color": "#00CC00"}),
        ("make it look golden",                  lambda: {"color": "#FFD700"}),
        ("give it a dark appearance",            lambda: {"color": "#222222"}),
        ("set emission color to {hex}",          lambda: {"color": _random_hex(), "property": "_EmissionColor"}),
    ],
    "Material.SetAlpha": [
        ("make it transparent",                  lambda: {"alpha": round(random.uniform(0.0, 0.3), 2)}),
        ("set opacity to {n:.0%}",              lambda: {"alpha": round(random.uniform(0.1, 1.0), 2)}),
        ("make it semi-transparent",             lambda: {"alpha": round(random.uniform(0.3, 0.7), 2)}),
        ("fade it out",                          lambda: {"alpha": round(random.uniform(0.0, 0.2), 2)}),
        ("make it fully opaque",                 lambda: {"alpha": 1.0}),
        ("set alpha to {n:.2f}",                lambda: {"alpha": round(random.uniform(0.0, 1.0), 2)}),
        ("ghost effect",                         lambda: {"alpha": round(random.uniform(0.1, 0.35), 2)}),
        ("reduce visibility to {n:.0%}",        lambda: {"alpha": round(random.uniform(0.1, 0.6), 2)}),
    ],
    "Animation.Play": [
        ("play the idle animation",              lambda: {"clip": "Idle"}),
        ("play run animation",                   lambda: {"clip": "Run"}),
        ("trigger the attack clip",              lambda: {"clip": "Attack"}),
        ("start the death animation",            lambda: {"clip": "Death"}),
        ("play {clip} clip",                     lambda: {"clip": random.choice(["Jump","Walk","Dance","Hit","Spawn"])}),
        ("animate it opening",                   lambda: {"clip": "Open"}),
        ("trigger victory pose",                 lambda: {"clip": "Victory"}),
        ("play the explode animation",           lambda: {"clip": "Explode"}),
    ],
    "Camera.SetFOV": [
        ("set field of view to {n}",             lambda: {"fov": random.randint(30, 110)}),
        ("zoom in",                              lambda: {"fov": random.randint(20, 45)}),
        ("zoom out",                             lambda: {"fov": random.randint(80, 110)}),
        ("widen the camera view",                lambda: {"fov": random.randint(75, 110)}),
        ("narrow fov to {n} degrees",            lambda: {"fov": random.randint(20, 50)}),
        ("set fov {n}",                          lambda: {"fov": random.randint(30, 110)}),
        ("fish-eye effect",                      lambda: {"fov": random.randint(100, 120)}),
        ("telephoto look",                       lambda: {"fov": random.randint(15, 30)}),
    ],
    "Camera.SetBackground": [
        ("set sky to black",                     lambda: {"color": "#000000"}),
        ("background color to {hex}",            lambda: {"color": _random_hex()}),
        ("make the background blue",             lambda: {"color": "#87CEEB"}),
        ("dark background",                      lambda: {"color": "#111111"}),
        ("set clear color to {hex}",             lambda: {"color": _random_hex()}),
        ("white background",                     lambda: {"color": "#FFFFFF"}),
        ("sunset background color",              lambda: {"color": "#FF6030"}),
        ("midnight sky",                         lambda: {"color": "#0A0A1A"}),
    ],
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
    "show":      ["enable", "activate", "turn on", "make visible"],
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

def expand_template(template: str, args_fn) -> tuple[str, dict]:
    """Resolve template placeholders and generate args."""
    args = args_fn()

    # Fill numeric placeholders
    n   = round(random.uniform(1, 15), 1)
    x   = round(random.uniform(-10, 10), 2)
    y   = round(random.uniform(0, 10), 2)
    z   = round(random.uniform(-10, 10), 2)
    m   = round(random.uniform(1, 4), 1)
    hex_ = _random_hex()
    adv  = random.choice(["slightly", "gently", "a bit", "sharply"])
    axis = random.choice(["x", "y", "z"])
    dir_ = random.choice(["right", "left", "up", "down", "forward", "back"])
    side = random.choice(["right", "left"])
    clip = random.choice(["Idle", "Run", "Attack", "Jump", "Dance", "Death", "Open"])

    instruction = template.format(
        n=n, x=x, y=y, z=z, m=m,
        hex=hex_, adv=adv, axis=axis,
        dir=dir_, side=side, clip=clip
    )

    instruction = apply_synonym_swap(instruction)
    return instruction, args

# ─── Build Dataset ────────────────────────────────────────────────────────────

def build_dataset(examples_per_action: int = 80) -> list[dict]:
    dataset = []

    for function, templates in SEED_TEMPLATES.items():
        per_template = max(1, examples_per_action // len(templates))

        for template, args_fn in templates:
            for _ in range(per_template):
                try:
                    instruction, args = expand_template(template, args_fn)
                    dataset.append({
                        "instruction": instruction,
                        "output": {
                            "function": function,
                            "args": args
                        }
                    })
                except Exception:
                    pass  # skip bad expansions

    random.shuffle(dataset)
    print(f"✓ Generated {len(dataset)} examples across {len(SEED_TEMPLATES)} actions")
    return dataset

def save_dataset(dataset: list[dict], path: str = DATASET_PATH):
    Path(path).write_text(json.dumps(dataset, indent=2))
    print(f"✓ Saved to {path}")

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
        device_map="auto",
        trust_remote_code=True,
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
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=True,              # safe now — short uniform examples
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✓ Model saved to {OUTPUT_DIR}")

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
        print(f"✓ FAISS index built: {len(scene_objects)} objects")

    def find(self, query: str, top_k: int = 1, threshold: float = 0.3) -> str | None:
        if self.index is None or not self.object_names:
            raise RuntimeError("Scene index is empty. Call rebuild() first.")

        vec = self.embedder.encode([query], convert_to_numpy=True)
        vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)

        scores, indices = self.index.search(vec, top_k)
        best_score = scores[0][0]
        best_idx   = indices[0][0]

        if best_score < threshold:
            return None  # no confident match

        return self.object_names[best_idx]

# ─── Inference Pipeline ───────────────────────────────────────────────────────

class UnityNLPPipeline:
    """
    Full pipeline:
      natural language → (model) → action + args
                       → (FAISS)  → target object name
                       → merged CommandResult ready for ExecuteCommand
    """

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
        """Call this whenever Unity sends a new scene snapshot."""
        self.scene_index.rebuild(scene_objects)

    def predict(self, instruction: str) -> dict:
        # 1. Model predicts function + args (no target)
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
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
            action = json.loads(response)
        except json.JSONDecodeError:
            return {"error": "parse_failed", "raw": response}

        # 2. FAISS resolves target from instruction
        target = self.scene_index.find(instruction)
        if target is None:
            return {"error": "no_target_found", "instruction": instruction}

        # 3. Merge into final command
        return {
            "function": action["function"],
            "target":   target,
            "args":     action.get("args", {})
        }

# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1 — generate dataset
    dataset = build_dataset(examples_per_action=3)
    save_dataset(dataset)

    # Step 2 — train
    # train()

    # # Step 3 — test inference
    # pipeline = UnityNLPPipeline()

    # Simulate a Unity scene snapshot arriving
    # pipeline.update_scene([
    #     "WoodDoor_01",
    #     "SpotLight_Blue",
    #     "HeavyBoulder_03",
    #     "PlayerCharacter",
    #     "CeilingLamp_02",
    #     "GlassWall_Left",
    # ])

    # tests = [
    #     "launch the boulder upward with a strong impulse",
    #     "dim the ceiling lamp",
    #     "make the glass wall transparent",
    #     "play the idle animation on the player",
    #     "rotate the door 90 degrees on y",
    #     "set the spotlight to red",
    # ]

    # print("\n─── Inference Tests ───")
    # for instruction in tests:
    #     result = pipeline.predict(instruction)
    #     print(f"\n> {instruction}")
    #     print(f"  {json.dumps(result, indent=2)}")