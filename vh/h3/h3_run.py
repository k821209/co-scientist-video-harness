#!/usr/bin/env python3
"""
MiniMax-H3 실행 스크립트 (ComfyUI, VH_H3_COMFY_HOST — default :8189)

사용법:
  python3 h3_run.py --image i2v_first.png --prompt "..." --out mytest
  python3 h3_run.py --mode t2v --prompt "..." --w 768 --h 1344
  python3 h3_run.py --image ref.png --prompt "..." --steps 20 --no-lora   # 고품질(느림)

전제:
  - comfyui-h3 컨테이너가 떠 있고 comfyui(8188)는 꺼져 있어야 함
  - 입력 이미지는 $VH_H3_HOME/input/ 에 두고 파일명만 넘긴다
  - 출력은 $VH_H3_HOME/output/
"""
import argparse, json, sys, time, urllib.request


def _detect_env():
    """(python_bin, docs_dir): VH_H3_PYTHON (default: this interpreter) and the
    directory this file lives in — krea2_edit.py / cq_enhance.py sit next to it."""
    import os as _os, sys as _sys
    return (_os.environ.get("VH_H3_PYTHON") or _sys.executable,
            _os.path.dirname(_os.path.abspath(__file__)))


def _io_dir(kind):
    """ComfyUI's input/output dir on THIS machine: $VH_H3_HOME/<kind>, else the
    container's own path when running inside it."""
    import os as _os
    home = _os.environ.get("VH_H3_HOME", "").rstrip("/")
    if home:
        return f"{home}/{kind}"
    ctn = f"/opt/ComfyUI/{kind}"
    if _os.path.isdir(ctn):
        return ctn
    raise SystemExit(f"VH_H3_HOME is not set and {ctn} does not exist")


def _ensure_ext_scripts(docs):
    """컨테이너 안에서 실행 시 cq_enhance.py / krea2_edit.py 사본을 output/ 에 배치.
    호스트에서 실행하면 원본 그대로 사용."""
    import os as _os, shutil as _shutil
    if docs == "/opt/ComfyUI/output":
        for name in ("cq_enhance.py", "krea2_edit.py", "cq_h3_api_template.json"):
            src = f"/opt/ComfyUI/output/{name}"  # 호스트에서 미리 넣어두거나
            if not _os.path.exists(src):
                # 없으면 skip — 호스트에서 파일 sync 필요
                print(f"[env] warn: {src} 없음 — 호스트에서 파일 복사 필요")

import os
HOST = os.environ.get("VH_H3_COMFY_HOST", "http://localhost:8189")

DIT_I2V = "minimax_h3_fl2va_pruned_bf16.safetensors"     # 첫/마지막 프레임 기반
DIT_REF = "minimax_h3_ref2va_pruned_bf16.safetensors"    # 참조 이미지/영상 기반
ENCODER = "qwen3vl_32b_minimax_h3_ultra_uncensored_heretic_int8_convrot.safetensors"
# 4스텝 터보 LoRA. fl2v 와 ref2v 는 서로 다른 가중치를 쓴다 (텐서 624 vs 416개).
LORA_FL2V  = "minimax_h3/fl2v_turbo_4step_v1.0_768p_bf16.safetensors"       # i2v(fl2va)용 v1.0
LORA_REF2V = "minimax_h3/ref2v_lightx2v_turbo_4step_v0.1.safetensors"       # ref2v/t2v(ref2va)용
LORA_FL2V_OLD = "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors"  # 구버전(플랫 폴더)
LORA_TAOMATE_I2V = "minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors"  # TaoMate 3-step fl2va default
# 애니 LoRA. 실사에서도 과한 번들거림을 잡아줘서 기본으로 켜둔다 (2026-08-24 확인).
# ref2v/t2v 기본으로 체인되는 스타일 LoRA 이름들 (아래 STYLE_PRESETS 의 키)
# 단독 hm 이 기본. 아나운서 실사에는 효과가 눈에 안 띄지만(차이 14~20) 오디오는
# 무해했고, 성인 프롬프트에서는 21~26 로 유의미했다.
#
# 상황별 프리셋 선택 (실측 근거, 2026-09-07):
#   일반 실사·아나운서    → 기본값 그대로 (hm 자동)
#   성인 콘텐츠 + 신음    → --style moan  (hm 은 지배당해 무의미, 단독 권장)
#   자위씬 (신음 있음)    → --style moan  (moan 이 자위 자세/뒤틀림도 함께 조절)
#   자위 텍스처만(조용)   → --style hmm   (질척 사운드, 신음 없음, 단독)
#     주의: moan+hmm 체인 = 신음 나오나 질척 사라지고 아나토미 부담. 값어치 없음.
#   성인 + 손가락 삽입    → --style moan --style beanflk:0.4  (beanflk 는 저자 고정)
#   특정 스타일·동작       → --style anime|motion|bmotion 등
#   완전히 비활성          → --no-style
#
# 체이닝은 원칙적으로 위험 (얼굴/오디오 망침). 저자·학습축이 겹치는 조합만 고려.
# 콜론으로 개별 강도: --style moan:0.7 --style hm:0.4 (전역 --style-strength 보다 우선).
STYLE_DEFAULT_CHAIN = ["hm"]

# 이름으로 고르는 스타일/동작 LoRA. --style <이름> 으로 쓴다.
# 강도는 각 저자 권장값. 측정치는 아나운서 프롬프트 기준 평균절대차(0~255).
# 프리셋 등록: (파일, 강도, 목적, 언제 쓰나, 주의사항)
# 강도는 저자 권장 또는 실측 후 최적치. --style-strength 로 덮어쓸 수 있다.
STYLE_PRESETS = {
    # 이름   파일 · 강도 · 목적 · 사용 시점 · 주의
    "hm":     ("minimax_h3/HMNSFW-AIO-V2.5.safetensors", 0.7,
               "실사 리얼리즘 강화. **ref2v/t2v 기본값** (일반 콘텐츠). "
               "성인 프롬프트에는 눈에 띄게(차이 21~26), 아나운서 같은 정적 실사에는 안 보이지만(14~20) "
               "오디오 무해라 상시 붙여둔다. 학습 분포는 성인 동작. "
               "**성인 콘텐츠 + moan 조합에선 무의미** (moan 이 지배해 hm 강도 조정이 안 보임). "
               "예시: (트리거 없음, 아무 실사 프롬프트에 자동으로 얹힘)"),

    "beanflk":("minimax_h3/BEANFLK_H3_V1.safetensors", 0.4,
               "손가락 삽입 동작 특화. 다른 LoRA가 표면만 맴돌 때 실제 삽입을 강제한다. "
               "**프롬프트 첫 단어에 BEANFLK 대문자 필수.** 강도 0.4 고정(overdrive 금지). "
               "성인 콘텐츠 전용. "
               "예시: 'BEANFLK. Close-up on her pelvis, her fingers push in fully, then withdraw halfway, then push in again, wet sliding motion visible.'"),

    "anime":  ("minimax_h3/FlatAnime_MiniMax_H3.safetensors", 1.0,
               "번들거림 억제(스페큘러) + 화면 전체 밝기 +44. "
               "실사 톤을 부드럽게 만들지만 **얼굴이 애니체로 흔들리는 위험**이 있다. "
               "프롬프트에 애니 언급 금지. hm 과 체이닝하면 얼굴/오디오 망침. "
               "예시: 평범한 실사 프롬프트 그대로 ('A Korean anchor in a studio, medium shot...') — 'anime', 'cel shading' 같은 단어 절대 금지"),

    "motion": ("minimax_h3/Motion_Repair.safetensors", 0.7,
               "범용 동작 연속성 보정. 뛰기·춤·격투 같은 동적 장면용. "
               "nfe4 기본에선 차이 미미(19~22)라 눈에 잘 안 띔. hm 대체용으로 쓸 것. "
               "예시: 'A woman running through a park, arms swinging, camera tracks alongside.'"),

    "bmotion":("minimax_h3/BetterMotion_h3_v1.safetensors", 0.6,
               "부드러운 인체 동작. **저자가 15~30 스텝을 요구**해서 우리 nfe4 와 안 맞음. "
               "쓰려면 `--no-pdd --steps 20` 조합. 지금 프리셋 기본에선 차이 15~19. "
               "예시: '--no-pdd --steps 20' + 'A dancer performing a slow pirouette, graceful arm movements, full-body shot.'"),

    "boob":   ("minimax_h3/MiniMax3_BoobPhysics_v1.safetensors", 1.0,
               "가슴 물리. **미검증** — 실제로 얹어본 적 없음. 텐서 416(ref2v 계열). "
               "예시: 미검증이라 확실치 않음. 시험 후 여기 갱신할 것."),

    "cameltoe": ("Cameltoe_v2.safetensors", 1.0,
               "카멜토(보지) 모양 특화. **미검증** — H3 호환 미확인. 텐서 155MB(9/14 추가). "
               "예시: 미검증이라 확실치 않음. 시험 후 여기 갱신할 것."),

    "moan":   ("minimax_h3/moawxx_2500.safetensors", 0.7,
               "여성 신음(pleasure moans) + 몸 뒤틀림·아치·떨림. **오디오 특화, 검증됨.** "
               "성인 콘텐츠에서 신음이 필요할 때 이걸 쓴다 — 유일하게 실제 신음을 낸다. "
               "**강도는 소리+자세 개방도를 함께 조절**(0.7=다리 벌림, 0.4=다리 모음). "
               "**단독 사용 권장** — hm 을 체이닝해도 moan 이 지배해서 hm 강도 변화가 안 보인다. "
               "체이닝은 아나토미 붕괴 위험까지 있음(다른 저자). "
               "예시: 'moawxx, soft sensual female moan, body gently writhing and arching. "
               "[본문 상황 서술]. Audio: soft breathy moans, occasional gasps.' "
               "체크포인트 2500 (2000/2500/2750 중 중간, 안전한 기본)."),

    "hmm":    ("minimax_h3/HMMasturbationV1.safetensors", 0.7,
               "**자위씬 전용, 좁은 용도.** 자위 동작 + 질척 텍스처 사운드 (신음 아님). "
               "트리거 hmmasturbation 첫단어. **hm 보다 2.5배 세게** (차이 60, R+62). "
               "**어떤 체이닝도 실패**: hm+hmm=아나토미 무너짐, moan+hmm=신음은 나오나 질척 사라지고 아나토미 부담 (2026-09-07). "
               "moan 이 오디오·자세 축을 지배해서 hmm 이 눌린다. **단독으로만, 조용한 자위 텍스처 필요할 때** 사용. "
               "예시: 'hmmasturbation. She lies on her back, her fingers slide between her thighs, wet sliding motion. Quiet room, only faint wet sounds.'"),
}
# PDD Acc (alibaba-pai). models/pdd_acc/ 에 둔다. 전용 노드로만 로드된다.
PDD_FL2V  = "minimax_h3_fl2va_pdd_acc_8step_comfyui.safetensors"
PDD_REF2V = "minimax_h3_ref2va_pdd_acc_8step_comfyui.safetensors"
VAE_VIDEO = "minimax_h3_video_vae_int8_convrot.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"


def build(a):
    i2v = a.mode == "i2v"
    dit = DIT_I2V if i2v else DIT_REF
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": dit, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": ENCODER, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": a.vae_video or VAE_VIDEO}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_AUDIO}},
    }
    model_src = ["1", 0]

    # PDD Acc (8스텝). 평범한 LoRA 가 아니라 trunk LoRA + 32구간 head bank 라
    # 전용 노드(MiniMaxH3PDDAccApply)로만 적용된다. 일반 로더는 head bank 를 버린다.
    # 시그마도 전용 스케줄러가 내는 것을 써야 한다.
    if a.pdd:
        wf["5"] = {"class_type": "MiniMaxH3PDDAccApply", "inputs": {
            "model": model_src,
            "pdd_file": (PDD_FL2V if i2v else PDD_REF2V),
            "nfe": str(a.pdd_nfe), "lora_strength": a.lora_strength,
            "head_strength": a.pdd_head_strength, "on_off_grid": "error"}}
        model_src = ["5", 0]

    # 4-step turbo LoRA. --no-lora 로 끄면 20스텝 고품질 경로.
    # 모드에 맞는 가중치를 골라야 한다. 예전에는 ref2v 에도 fl2v LoRA 를 걸고 있었다.
    if not a.no_lora and not a.pdd:
        turbo = a.turbo_lora or (LORA_TAOMATE_I2V if i2v else LORA_REF2V)
        wf["5"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": model_src, "lora_name": turbo, "strength_model": a.lora_strength}}
        model_src = ["5", 0]

    # 스타일 LoRA 체인. 터보/PDD 뒤에 순서대로 붙인다.
    for i, (fn, st) in enumerate(getattr(a, "style_chain", [])):
        nid = "5b%d" % i
        wf[nid] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": model_src, "lora_name": fn, "strength_model": st}}
        model_src = [nid, 0]

    # PDD Acc 헤드는 SigmaShift 12.0/3.0 으로 학습됐다.
    # 다른 값을 주면 MiniMaxH3PDDAccApply 가 거부한다.
    # ★shift_video / shift_audio. PDD Acc 헤드는 12/3 으로 학습됐다.
    #   비-PDD(터보 LoRA) 경로는 예전엔 둘 다 --shift(기본 5) 였는데, 커뮤니티 노드
    #   (Smite79/MiniMax-H3-LongVideos, RealRebelAI 포크)와 TaoMate 모델카드는 **12/3**,
    #   "video:audio 를 4:1 근처로 유지하지 않으면 오디오가 깨진다"고 적고 있다.
    #   기존 설정을 조용히 바꾸지 않으려고 기본값은 그대로 두고, --shift-audio 로 따로 준다.
    sv, sa = (12.0, 3.0) if a.pdd else (a.shift, a.shift_audio if a.shift_audio else a.shift)
    wf["6"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {
        "model": model_src, "shift_video": sv, "shift_audio": sa}}
    model_src = ["6", 0]

    # Spectrum(스텝 건너뛰기 가속) — 스텝이 warmup보다 커야 의미 있음
    if a.spectrum:
        if a.steps <= 5:
            print("주의: Spectrum warmup=5. steps<=5 면 효과 없음", file=sys.stderr)
        wf["7"] = {"class_type": "SpectrumApplyMiniMaxH3", "inputs": {
            "model": model_src, "enabled": True, "blend_weight": 0.5, "degree": 4,
            "ridge_lambda": 0.1, "window_size": 2.0, "flex_window": 0.75,
            "warmup_steps": 5, "tail_actual_steps": 1, "max_history": 8,
            "debug": False, "history_storage": "system_ram"}}
        model_src = ["7", 0]

    # 조건 생성
    if i2v:
        wf["8"] = {"class_type": "LoadImage", "inputs": {"image": a.image}}
        cond = {"clip": ["2", 0], "vae": ["3", 0], "prompt": a.prompt,
                "width": a.w, "height": a.h, "length": a.length, "first_frame": ["8", 0]}
        if a.last_image:
            wf["9"] = {"class_type": "LoadImage", "inputs": {"image": a.last_image}}
            cond["last_frame"] = ["9", 0]
        wf["10"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": cond}
    else:
        cond = {"clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0], "prompt": a.prompt,
                "width": a.w, "height": a.h, "length": a.length,
                "ref_image_size": a.ref_size}
        # ref_images 는 Autogrow(prefix="ref_image_"). API 키는 부모 id 와 점으로 이어
        # "ref_images.ref_image_0" 형태여야 한다 (_io.py finalize_prefix).
        # 이걸 빼먹으면 참조 이미지가 통째로 무시되고 사실상 t2v 가 된다.
        refs = []
        if a.image:
            refs.append(a.image)
        refs.extend(a.ref or [])
        for i, fn in enumerate(refs[:9]):
            nid = str(30 + i)
            wf[nid] = {"class_type": "LoadImage", "inputs": {"image": fn}}
            cond[f"ref_images.ref_image_{i}"] = [nid, 0]
        if not refs:
            print("참조 이미지 없음 → 사실상 t2v", file=sys.stderr)
        wf["10"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": cond}

    wf.update({
        "11": {"class_type": "BasicGuider", "inputs": {"model": model_src, "conditioning": ["10", 0]}},
"12": ({"class_type": "MiniMaxH3PDDAccScheduler", "inputs": {
            "nfe": str(a.pdd_nfe), "denoise": 1.0}}
            if a.pdd else
            {"class_type": "BasicScheduler", "inputs": {
            "model": model_src, "scheduler": a.scheduler, "steps": a.steps, "denoise": 1.0}}),
        "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": a.sampler}},
        "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": a.seed}},
        "15": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["14", 0], "guider": ["11", 0], "sampler": ["13", 0],
            "sigmas": ["12", 0], "latent_image": ["10", 1]}},
    })

    # 업스케일 (선택). 이미 만들어진 latent 를 재샘플링해 해상도/디테일을 올린다.
    # 노드는 AV latent 를 그대로 다루므로 오디오는 재생성 없이 통과한다.
    sample_out = ["15", 0]
    if a.upscale:
        # 기본 1.5배. 우리 검증치가 768x1152 → 1152x1728 이었다.
        def snap32(n): return max(32, round(n / 32) * 32)
        uw = a.upscale_w or snap32(a.w * 1.5)
        uh = a.upscale_h or snap32(a.h * 1.5)
        # 3D 모델 기반이 기본. bicubic 보간은 저주파 정보가 비어 재샘플러가 배경/인물을
        # 새로 채우게 되어 겹치고 배경이 바뀐다 (2026-08-24 실측).
        if a.upscale_model:
            wf["40"] = {"class_type": "MMH3LatentUpscaleWithModelParams", "inputs": {
                "model_name": a.upscale_model, "width": uw, "height": uh,
                "device": "cuda", "precision": "fp16"}}
        else:
            wf["40"] = {"class_type": "MMH3LatentUpscaleParams", "inputs": {
                "method": a.upscale_method, "width": uw, "height": uh}}
        wf["41"] = {"class_type": "BasicScheduler", "inputs": {
            "model": model_src, "scheduler": a.scheduler,
            "steps": a.upscale_steps, "denoise": a.upscale_denoise}}
        wf["42"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": a.upscale_sampler}}
        wf["43"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": a.seed + 1}}
        # 업스케일용 conditioning 은 "업스케일 후 해상도" 로 다시 만들어야 한다.
        # 원본 해상도 conditioning 을 그대로 주면 shape mismatch 로 죽는다
        # (864 vs 2808 등). 예제 워크플로도 ReferenceToVideo 를 두 개 둔다.
        cond_up = dict(cond)
        cond_up["width"], cond_up["height"] = uw, uh
        # 업스케일용 conditioning 은 대사가 든 원본 프롬프트가 아니라
        # "디테일 강화" 지시만 준다. 저자 예제도 그렇게 한다.
        # 원본 프롬프트를 그대로 넘기면 토큰 구성이 달라져 shape mismatch 로 죽는다.
        cond_up["prompt"] = a.upscale_prompt
        cond_up["ref_image_size"] = a.upscale_ref_size
        # 참조 이미지를 업스케일 단계에도 주면 참조의 배경까지 끌어온다.
        # --upscale-no-ref 로 빼면 원본 latent 의 배경이 유지되는지 확인할 것.
        if a.upscale_no_ref:
            for k in [k for k in cond_up if k.startswith("ref_images.")]:
                cond_up.pop(k)
        wf["47"] = {"class_type": ("MiniMaxH3ReferenceToVideo" if not i2v
                                   else "MiniMaxH3ImageToVideo"), "inputs": cond_up}
        # 업스케일 단계는 PDD 를 빼야 한다. PDD 는 정해진 학습 시그마 경계에서만 돌아서
        # denoise 0.2 같은 임의 시그마를 만나면 "not a trained PDD block boundary" 로 죽는다.
        # 업스케일용 모델은 원본 UNET(1) 뒤에 SigmaShift 만 새로 걸어서 만든다.
        wf["48"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {
            "model": ["1", 0], "shift_video": a.shift, "shift_audio": a.shift}}
        us = {"model": ["48", 0], "conditioning": ["47", 0], "latent": sample_out,
              "noise": ["43", 0], "sampler": ["42", 0], "sigmas": ["41", 0],
              "cfg": 1.0, "latent_upscale_param": ["40", 0]}
        if a.upscale_chunk or a.upscale_overlap:
            wf["44"] = {"class_type": "MMH3TemporalSplitParams", "inputs": {
                "chunk_length": a.upscale_chunk or 136,
                "temporal_overlap": a.upscale_overlap or 17,
                "anchor_strength": 0.999}}
            us["temporal_split_param"] = ["44", 0]
        if a.upscale_tile:
            wf["45"] = {"class_type": "MMH3SpatialSplitParams", "inputs": {
                "tile_width": a.upscale_tile, "tile_height": a.upscale_tile,
                "spatial_w_overlap": 128, "spatial_h_overlap": 128,
                "fade_width": 32, "fade_height": 32, "min_tile_size": 256,
                "overlap_mode": "earlier", "overlap_blend": "linear"}}
            us["spatial_split_param"] = ["45", 0]
        wf["46"] = {"class_type": "MMH3UltimateUpscale", "inputs": us}
        sample_out = ["46", 0]

    wf.update({
        "16": {"class_type": "VAEDecode", "inputs": {"samples": sample_out, "vae": ["3", 0]}},
        "17": {"class_type": "VAEDecodeAudio", "inputs": {"samples": sample_out, "vae": ["4", 0]}},
        "18": {"class_type": "CreateVideo", "inputs": {"images": ["16", 0], "fps": 24.0, "audio": ["17", 0]}},
        "19": {"class_type": "SaveVideo", "inputs": {
            "video": ["18", 0], "filename_prefix": a.out, "format": "auto", "codec": "auto"}},
    })
    return wf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["i2v", "ref2v", "t2v"], default="i2v",
                   help="i2v=첫프레임(fl2va) / ref2v,t2v=참조(ref2va)")
    p.add_argument("--image", help="첫 프레임 파일명 (input/ 기준). i2v 필수")
    p.add_argument("--last-image", help="마지막 프레임 (선택)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", default="h3_out")
    p.add_argument("--w", type=int, default=768)
    p.add_argument("--h", type=int, default=1024, help="세로 3:4=1024, 9:16=1344, 1:1=768")
    p.add_argument("--length", type=int, default=124, help="24fps 프레임 수. 학습범위 124~362, 17단위")
    p.add_argument("--steps", type=int, default=0,
                   help="0=모드별 기본(i2v/fl2v 4, ref2v 6). --no-lora 면 20")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--vae-video", default=None,
                   help="video VAE 파일명 override. 기본 minimax_h3_video_vae_fp16. "
                        "int8: minimax_h3_video_vae_int8_convrot.safetensors")
    p.add_argument("--krea-ref",
                   help="Krea2 Edit 로 시작 이미지 생성용 레퍼런스 실사진 (input/ 기준). "
                        "지정하면 --image 는 자동 생성됨")
    p.add_argument("--krea-prompt", default=None,
                   help="Krea2 Edit 편집 지시 (명령형). 미지정 시 --prompt 사용")
    p.add_argument("--krea-blur", type=float, default=4.0,
                   help="Krea 결과에 걸 저역통과 반경 (기본 4.0)")
    p.add_argument("--krea-aspect", default="3:4 (Portrait Standard)")
    p.add_argument("--krea-mp", type=float, default=1.5)
    p.add_argument("--cq-enhance", action="store_true",
                   help="생성 완료 후 LTX-2.5 CQ V2 로 화질 개선 (cq_enhance.py 호출)")
    p.add_argument("--cq-strength", type=float, default=0.7,
                   help="CQ enhancer LoRA 강도 (기본 0.7)")
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--shift-audio", type=float, default=None,
                   help="오디오 SigmaShift. 안 주면 --shift 와 같은 값(기존 동작). "
                        "커뮤니티 권장은 video 12 / audio 3 (4:1)")
    p.add_argument("--scheduler", default="simple")
    p.add_argument("--sampler", default="euler")
    p.add_argument("--no-lora", action="store_true", help="4스텝 LoRA 끄기 (20스텝 권장)")
    p.add_argument("--lora-strength", type=float, default=1.0)
    p.add_argument("--turbo-lora", default=None,
                   help="turbo LoRA override (예: minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors)")
    # PDD Acc 는 ref2v/t2v 기본 경로다. nfe4 가 nfe8 과 사실상 동일한 결과를
    # 280초(vs 440초)에 낸다 — 평균 절대차 3~4/255. 기존 터보 6스텝(320초)보다도 빠르다.
    p.add_argument("--pdd", action="store_true", default=None,
                   help="PDD Acc 경로 강제. ref2v/t2v 는 기본 켜짐")
    p.add_argument("--no-pdd", dest="pdd", action="store_false",
                   help="PDD 끄고 기존 터보 LoRA 경로로")
    p.add_argument("--pdd-nfe", type=int, default=4, choices=[4, 6, 8],
                   help="모델 평가 횟수. 4=기본(280초), 8=학습된 블록 크기(440초, 결과는 거의 동일)")
    p.add_argument("--pdd-head-strength", type=float, default=1.0)
    # FlatAnime 은 애니체 전환이 아니라 "번들거림(과한 스페큘러) 억제" 로 쓴다.
    # ref2v 에서만 효과가 있고 i2v 에서는 유무 차이가 없었다(실측). 그래서 ref2v 기본만 켠다.
    # i2v 에서 굳이 쓰려면 --style-lora 로 명시할 것.
    p.add_argument("--style-lora", default=None,
                   help="스타일 LoRA 파일 직접 지정. --style 프리셋이 우선한다")
    p.add_argument("--no-style", action="store_true", help="스타일 LoRA 끄기")
    # 프리셋 요약을 --help 에 노출 (긴 설명은 STYLE_PRESETS 참조)
    _preset_help = " | ".join("%s(%s)" % (k, v[1]) for k, v in sorted(STYLE_PRESETS.items()))
    def _style_arg(v):
        """--style NAME 또는 NAME:STRENGTH 를 (name, strength_or_None) 로."""
        if ":" in v:
            name, st = v.split(":", 1)
            try: st = float(st)
            except ValueError:
                raise argparse.ArgumentTypeError("강도는 숫자: %r" % v)
        else:
            name, st = v, None
        if name not in STYLE_PRESETS:
            raise argparse.ArgumentTypeError(
                "알 수 없는 프리셋 %r. 목록: %s" % (name, sorted(STYLE_PRESETS)))
        return (name, st)
    p.add_argument("--style", action="append", type=_style_arg, metavar="NAME[:STRENGTH]",
                   help=("스타일/동작 LoRA 프리셋. 여러 번 = 체인(얼굴/오디오 위험). "
                         "미지정시 hm 자동, --no-style 로 끔. "
                         "**콜론으로 개별 강도 지정**: --style moan:0.7 --style hm:0.4. "
                         "콜론 없으면 프리셋 기본. --style-strength 는 전역 덮어쓰기(모든 항목 같은 값). "
                         "옵션(기본강도): " + _preset_help))
    p.add_argument("--style-strength", type=float, default=1.0)
    p.add_argument("--ref", action="append",
                   help="ref2v 추가 참조 이미지. 여러 번 지정 가능 (최대 9장)")
    # 업스케일 (재샘플링). w/h 미지정시 각각 2배.
    # 업스케일은 선택. 지글거림 잡을 때 --upscale 만 붙이면 되고 크기는 1.5x 로 자동.
    # 배경이 다시 그려지므로 배경이 중요한 컷엔 쓰지 말 것.
    p.add_argument("--upscale", action="store_true",
                   help="latent 재샘플링으로 해상도/디테일 개선 (+40초). 배경이 재생성될 수 있음")
    p.add_argument("--upscale-w", type=int, help="기본: 원본 w x 1.5, 32 배수")
    p.add_argument("--upscale-h", type=int, help="기본: 원본 h x 1.5, 32 배수")
    p.add_argument("--upscale-prompt",
                   default="segment video which takes <Picture 1> as reference, clear and sharp details",
                   help="업스케일 단계 전용 프롬프트. 대사를 넣지 말 것")
    p.add_argument("--upscale-ref-size", choices=["match","max"], default="max",
                   help="업스케일 단계 ref 크기. 예제는 max")
    p.add_argument("--upscale-no-ref", action="store_true",
                   help="업스케일 단계에서 참조 이미지를 빼기 (참조 배경 유입 방지)")
    p.add_argument("--upscale-model", default="minimax_h3_latent_upscaler_3d_fp16.safetensors",
                   help="3D latent upscaler 파일명. 빈 문자열이면 아래 --upscale-method 로 보간")
    p.add_argument("--upscale-method", default="bicubic",
                   choices=["nearest-exact","bilinear","area","bicubic"])
    # 저자 예제: 1스텝, denoise 0.2, sa_solver. 이보다 세게 주면 배경/인물이 재생성된다.
    # 실측: 6/0.35 로 돌렸다가 배경·인물 크기가 완전히 바뀜 (2026-08-24).
    p.add_argument("--upscale-steps", type=int, default=1)
    p.add_argument("--upscale-denoise", type=float, default=0.2,
                   help="낮게: 디테일만. 0.3 넘으면 재생성 시작 (기본 0.2)")
    p.add_argument("--upscale-sampler", default="sa_solver",
                   help="저자 권장 sa_solver. euler 도 됨")
    # 시간 청킹은 VRAM 부족한 GPU 용이다. GB10(128GB 통합)에서는 필요 없고,
    # 켜면 conditioning 재앵커링에서 shape mismatch 로 죽는다 (2026-08-24 실측).
    p.add_argument("--upscale-chunk", type=int, default=0, help="시간 청킹 (17 배수). 0=끔(GB10 권장)")
    p.add_argument("--upscale-overlap", type=int, default=0, help="청킹 오버랩 (17 배수). chunk 켤 때만")
    p.add_argument("--upscale-tile", type=int, default=0, help="공간 타일 크기 (32 배수). 0=끔")
    p.add_argument("--ref-size", choices=["match", "max"], default="match",
                   help="max 는 짧은 변 2048 로 정체성 보존이 좋지만 몇 배 느리다")
    p.add_argument("--spectrum", action="store_true", help="스텝 건너뛰기 가속 (steps>5일 때만)")
    a = p.parse_args()

    if a.mode == "t2v":
        a.mode = "ref2v"
    if a.krea_ref:
        import subprocess, os
        IN = _io_dir("input")
        krea_out_base = f"krea_{a.out}"
        krea_out_png = f"{IN}/{krea_out_base}.png"
        krea_prompt = a.krea_prompt or a.prompt
        py, docs = _detect_env()
        cli = f"{docs}/krea2_edit.py"
        print(f"\n[KREA] {a.krea_ref} + '{krea_prompt[:60]}...' -> {krea_out_base}.png", flush=True)
        r = subprocess.run([py, cli, f"{IN}/{a.krea_ref}", krea_prompt,
                            "--aspect", a.krea_aspect, "--mp", str(a.krea_mp),
                            "--out", krea_out_png])
        if r.returncode != 0 or not os.path.exists(krea_out_png):
            sys.exit(f"[KREA] 실패 rc={r.returncode}")
        # lp blur
        from PIL import Image, ImageFilter
        blur_name = f"{krea_out_base}_lp{int(a.krea_blur)}.png"
        blur_path = f"{IN}/{blur_name}"
        Image.open(krea_out_png).convert("RGB").filter(
            ImageFilter.GaussianBlur(radius=a.krea_blur)).save(blur_path)
        print(f"[KREA] blur {a.krea_blur}px -> {blur_name}", flush=True)
        a.image = blur_name

    if a.mode == "i2v" and not a.image:
        p.error("--mode i2v 에는 --image 가 필요합니다")
    # ref2v 는 4스텝에서 몸이 겹쳐 보이는 아티팩트가 난다(실측).
    # 6스텝이면 잡힌다. 4→6 이 임계. ref2v 터보 LoRA 가 v0.1 초기물이라 증류가 덜 됐다.
    # fl2v 터보는 v1.0 이라 4스텝으로 충분하다.
    if a.steps == 0:
        a.steps = 20 if a.no_lora else (3 if a.mode == "i2v" else 6)  # TaoMate 3-step
    # PDD 는 ref2v/t2v 기본. i2v(fl2v)는 기존 v1.0 터보가 4스텝으로 충분해 그대로 둔다.
    if a.pdd is None:
        a.pdd = (a.mode != "i2v") and not a.no_lora
    # 스타일 LoRA 체인을 [(파일, 강도), ...] 로 확정한다.
    # 우선순위: --no-style > --style 프리셋 > --style-lora 직접지정 > 모드별 기본
    if a.no_style:
        a.style_chain = []
    elif a.style:
        # 강도 우선순위: 콜론(:X) > --style-strength(1.0 아닐 때) > 프리셋 기본
        def _pick(default, colon):
            if colon is not None: return colon
            if a.style_strength != 1.0: return a.style_strength
            return default
        a.style_chain = [(STYLE_PRESETS[n][0], _pick(STYLE_PRESETS[n][1], st))
                         for n, st in a.style]
    elif a.style_lora:
        a.style_chain = [(a.style_lora, a.style_strength)]
    elif a.mode != "i2v":
        a.style_chain = [(STYLE_PRESETS[n][0], STYLE_PRESETS[n][1]) for n in STYLE_DEFAULT_CHAIN]
    else:
        a.style_chain = []
    if a.no_lora and a.steps == 4:
        a.steps = 20
        print("--no-lora 이므로 steps를 20으로 조정", file=sys.stderr)

    wf = build(a)
    print(f"[{a.mode}] {a.w}x{a.h} {a.length}f {a.steps}steps "
          f"lora={'off' if a.no_lora else 'on'} spectrum={'on' if a.spectrum else 'off'}", flush=True)

    req = urllib.request.Request(HOST + "/prompt", data=json.dumps({"prompt": wf}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        pid = json.loads(urllib.request.urlopen(req, timeout=60).read())["prompt_id"]
    except urllib.error.HTTPError as e:
        print("거부됨:", e.read().decode()[:2000]); sys.exit(1)
    except urllib.error.URLError as e:
        print(f"8189 접속 실패: {e}\n → comfyui-h3 컨테이너가 떠 있는지 확인"); sys.exit(1)
    print("prompt_id:", pid, flush=True)

    t0 = time.time()
    while True:
        time.sleep(20)
        h = json.loads(urllib.request.urlopen(HOST + f"/history/{pid}", timeout=60).read())
        if pid in h:
            st = h[pid].get("status", {})
            print(f"[{time.time()-t0:.0f}s] {st.get('status_str')}", flush=True)
            for m in st.get("messages", [])[-4:]:
                s = str(m)
                if "error" in s.lower() or "Traceback" in s:
                    print("  !", s[:600], flush=True)
            fn = None
            for k, v in h[pid].get("outputs", {}).items():
                print(f"  {json.dumps(v, ensure_ascii=False)[:250]}", flush=True)
                for arr in (v.get("gifs") or v.get("images") or v.get("videos") or []):
                    if isinstance(arr, dict) and arr.get("filename", "").endswith(".mp4"):
                        fn = arr["filename"]
            if a.cq_enhance and fn:
                import subprocess, os
                OUT = _io_dir("output")
                src = f"{OUT}/{fn}"
                out = src.replace(".mp4", "_cq.mp4")
                py, docs = _detect_env()
                cli = f"{docs}/cq_enhance.py"
                print(f"\n[CQ] {src} -> {out} (strength={a.cq_strength})", flush=True)
                r = subprocess.run([py, cli, "--src", src, "--out", out,
                                    "--strength", str(a.cq_strength), "--keep-audio"])
                if r.returncode == 0:
                    print(f"[CQ] 완료 {out}", flush=True)
                else:
                    print(f"[CQ] 실패 rc={r.returncode}", flush=True)
            break
        print(f"[{time.time()-t0:.0f}s] 진행중", flush=True)


if __name__ == "__main__":
    main()
