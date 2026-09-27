"""krea2_edit.py — h3 컨테이너 (8189, 인증 없음) 용 Krea2 Edit CLI.

사용자 원본 (scripts/krea2_edit_api.py, comfyui:8188 인증판) 을 h3 환경에 맞게 정리:
  - Host: VH_H3_COMFY_HOST / COMFY_HOST (default http://localhost:8189)
  - COMFY_PW 없어도 실행 (h3 는 no auth)
  - 나머지 workflow 는 동일 (Krea2EditModelPatch + GroundedEncode + identity_edit LoRA)

사용법:
  python3 krea2_edit.py <input_image> "edit prompt" \
      [--aspect "3:4 (Portrait Standard)"] [--mp 1.5] [--seed 42] \
      [--neg "..."] [--out path.png]

⚠ 프롬프트 = '명령형 편집 지시' (identity 유지):
  O "Keep her face, hair, and outfit exactly the same. Change ..."
  X "a curvy korean woman in an office"   ← 서술형은 인물 복제됨
"""
import argparse, json, mimetypes, os, pathlib, sys, time
try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "requests"])
    import requests

import os
DEFAULT_HOST = os.environ.get("VH_H3_COMFY_HOST", "http://localhost:8189")
ASPECTS = ["1:1 (Square)", "2:3 (Portrait Photo)", "3:2 (Photo)",
           "3:4 (Portrait Standard)", "4:3 (Standard)",
           "9:16 (Portrait Widescreen)", "16:9 (Widescreen)", "21:9 (Ultrawide)"]


def build_workflow(image_name, prompt, neg, aspect, megapixels, seed, denoise=1.0):
    return {
        "55": {"inputs": {"unet_name": "krea2_turbo_fp8_scaled.safetensors",
                          "weight_dtype": "default"}, "class_type": "UNETLoader"},
        "56": {"inputs": {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors",
                          "type": "krea2", "device": "default"}, "class_type": "CLIPLoader"},
        "57": {"inputs": {"vae_name": "qwen_image_vae.safetensors"}, "class_type": "VAELoader"},
        "86": {"inputs": {"lora_name": "KNPV4.1_pre.safetensors",
                          "strength_model": 1, "model": ["55", 0]},
               "class_type": "LoraLoaderModelOnly"},
        "71": {"inputs": {"lora_name": "krea2_identity_edit_v1_1.safetensors",
                          "strength_model": 1, "model": ["86", 0]},
               "class_type": "LoraLoaderModelOnly"},
        "72": {"inputs": {"image": image_name}, "class_type": "LoadImage"},
        "83": {"inputs": {"aspect_ratio": aspect, "megapixels": megapixels, "multiple": 8},
               "class_type": "ResolutionSelector"},
        "77": {"inputs": {"resize_type": "scale dimensions",
                          "resize_type.width": ["83", 0], "resize_type.height": ["83", 1],
                          "resize_type.crop": "center", "scale_method": "area",
                          "input": ["72", 0]}, "class_type": "ResizeImageMaskNode"},
        "73": {"inputs": {"pixels": ["77", 0], "vae": ["57", 0]}, "class_type": "VAEEncode"},
        "79": {"inputs": {"model": ["71", 0], "source_latent": ["73", 0]},
               "class_type": "Krea2EditModelPatch"},
        "84": {"inputs": {"prompt": prompt, "grounding_px": 768, "clip": ["56", 0],
                          "image": ["77", 0]}, "class_type": "Krea2EditGroundedEncode"},
        "85": {"inputs": {"prompt": neg, "grounding_px": 768, "clip": ["56", 0],
                          "image": ["77", 0]}, "class_type": "Krea2EditGroundedEncode"},
        "82": {"inputs": {"width": ["83", 0], "height": ["83", 1], "batch_size": 1},
               "class_type": "EmptySD3LatentImage"},
        "53": {"inputs": {"seed": seed, "steps": 8, "cfg": 1, "sampler_name": "euler",
                          "scheduler": "simple", "denoise": denoise, "model": ["79", 0],
                          "positive": ["84", 0], "negative": ["85", 0],
                          "latent_image": ["82", 0]}, "class_type": "KSampler"},
        "54": {"inputs": {"samples": ["53", 0], "vae": ["57", 0]}, "class_type": "VAEDecode"},
        "29": {"inputs": {"filename_prefix": "krea2_edit", "images": ["54", 0]},
               "class_type": "SaveImage"},
    }


def upload_image(s, host, path):
    p = pathlib.Path(path)
    if not p.exists(): sys.exit(f"입력 이미지 없음: {path}")
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    files = {"image": (p.name, p.read_bytes(), mime)}
    r = s.post(f"{host}/upload/image", files=files, data={"overwrite": "true"}, timeout=60)
    if r.status_code != 200: sys.exit(f"업로드 실패 {r.status_code}: {r.text[:200]}")
    j = r.json()
    name = j["name"]
    if j.get("subfolder"): name = f"{j['subfolder']}/{name}"
    print(f"업로드: {name}")
    return name


def run(s, host, workflow, out_path):
    r = s.post(f"{host}/prompt", json={"prompt": workflow}, timeout=30)
    if r.status_code != 200: sys.exit(f"제출 실패 {r.status_code}: {r.text[:400]}")
    pid = r.json().get("prompt_id")
    print(f"제출 prompt_id={pid}")
    t0 = time.time()
    while True:
        hist = s.get(f"{host}/history/{pid}", timeout=20).json()
        if pid in hist:
            for node in hist[pid].get("outputs", {}).values():
                for img in node.get("images", []):
                    q = {"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                         "type": img.get("type", "output")}
                    data = s.get(f"{host}/view", params=q, timeout=120).content
                    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    pathlib.Path(out_path).write_bytes(data)
                    print(f"저장: {out_path} ({len(data)//1024} KB)")
                    return
            sys.exit(f"출력 없음: {hist[pid].get('outputs')}")
        if time.time() - t0 > 600: sys.exit("타임아웃")
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="레퍼런스 이미지 경로")
    ap.add_argument("prompt", help="편집 지시 (명령형)")
    ap.add_argument("--neg", default="")
    ap.add_argument("--aspect", default="3:4 (Portrait Standard)", choices=ASPECTS)
    ap.add_argument("--mp", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--denoise", type=float, default=1.0)
    ap.add_argument("--out", default="krea2_edit_out.png")
    ap.add_argument("--host", default=os.environ.get("COMFY_HOST", DEFAULT_HOST))
    a = ap.parse_args()
    seed = a.seed or (int.from_bytes(os.urandom(6), "big") % (2**53))

    s = requests.Session()
    name = upload_image(s, a.host, a.input)
    wf = build_workflow(name, a.prompt, a.neg, a.aspect, a.mp, seed, a.denoise)
    print(f'편집: "{a.prompt[:60]}" | {a.aspect} {a.mp}MP denoise={a.denoise} seed={seed}')
    run(s, a.host, wf, a.out)


if __name__ == "__main__":
    main()
