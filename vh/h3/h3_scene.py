"""h3_scene.py — 경계 키프레임 체인으로 H3 장면 하나를 만든다.

디스커션 그래프 그대로:

    샷 프롬프트 i ─┬─→ (Krea) 경계 이미지 B_i+1 ──┐
                   └─────────────────────────────┼─→ H3 조각 i  (B_i → B_i+1)
    경계 이미지 B_i ─────────────────────────────┘

경계 한 장을 앞 조각의 last 이자 다음 조각의 first 로 **공유**한다. 생성된 프레임을 다음
조각에 넘기지 않으므로 톤·대비 누적 경로가 없고, 이음매 양쪽이 같은 그림이라 어긋날 수 없다.

────────────────────── ★이 스크립트는 경로를 조립하지 않는다 (2026-09-27) ──────────────────────

**경계 이미지의 기준은 로컬 파일명이 아니라 Video 탭의 행이다.** 사용자가 GO 를 켠 시점에
행에 들어 있는 파일이 유일한 진실이고, 여기 있는 png 는 사본일 뿐이다.

그래서 이 스크립트는 **조각의 입력 경로를 스스로 정하지 않는다.** `--first` / `--last` 로
받은 것만 쓴다. `f"{out}_b{n}.png"` 같은 이름 조립이 어긋남의 근원이었다 — 같은 이름의
_v2 가 생기거나, 앞 조각을 다시 뽑아 실제 마지막 프레임이 키프레임과 달라진 순간
이름만 보고 집으면 틀린 그림으로 5분을 태운다.

이 스크립트는 MCP 에 접근할 수 없다(GPU 머신에서 돈다). 그래서 역할이 이렇게 나뉜다:

    에이전트                              이 스크립트
    ──────────────────────────────────    ────────────────────────────────
    boundaries 실행                  →    B_i 를 만들고 **매니페스트**를 돌려준다
    update_video_chunk(last_image=)  ←    (행에 붙이는 건 에이전트)
    사용자가 탭에서 GO
    get_video_chunk_image(n,"start") →
    chunk --n --first --last         →    받은 경로 그대로 생성, 결과 경로를 돌려준다
    add_video_chunk(local_path=)     ←
    join --parts                     →    이어붙인 파일 경로를 돌려준다
    join_video_chunks(local_path=)   ←

매니페스트와 결과 경로는 `###JSON###` 줄 뒤에 한 줄 JSON 으로 나온다. 행 번호(n)는
**탭과 같은 1-based**, 내부 조각 번호(i)는 0-based다: n = i + 1.

────────────────────────── 씬을 쓸 때 지킬 것 (전부 실측) ──────────────────────────
① **한 조각에 큰 동작 하나.** "마시고 내려놓기"처럼 두 단계를 넣으면 끝에서 디졸브가 잦다.
   통과율 실측: 3동작 6회 중 2회 · 2동작 6회 중 3회 · **1동작 7회 중 5회**.
② **경계는 정지 자세가 아니라 동작 중간**으로 잡는다(컵이 입가에 닿은 순간 등).
③ **입을 쓰는 동작이 있는 조각엔 대사를 넣지 않는다.** H3 는 조각 전체에 오디오를 만든다.
④ 키프레임 지시는 **5초 안에 닿을 자세**로.
⑤ Krea 는 **소품 상태 변경이 약하다**("책을 덮는다"가 반영 안 됨).
⑥ **이동하는 씬은 경계마다 배경을 길의 다른 구간으로.** 네 장이 같은 지점이면 모델이
   전진을 포기하고 제자리 뛰기가 된다(실측: 달리기 씬). 배경 요소는 전 경계에 넣거나 전부 뺀다.

────────────────────────── 자동 검사 (사용자가 원본 영상으로 정한 값) ──────────────────────────
· `sparkle` 밝은 점 비율 % — 기준 0.8. 0.16~0.27 깨끗 / 0.61 "안 보임" / 1.13 "심하다"
· `tailjump` 뒤 40프레임 최대 프레임차 — 기준 14. 1~7 정상 / 12.5 "괜찮음" / 16.1 "거슬린다"
· 둘 중 하나라도 넘으면 **시드를 밀어 재시도**(기본 2회). 전부 미달이면 **제일 나은 것**을 쓴다.
· `motion` 프레임간 중앙값 — 1.0 미만이면 **경고만**. 정상 2.5~3.0.
· ★검사 셋 다 '되돌아오는 움직임'과 '중간에 생겼다 사라지는 배경 요소'는 못 잡는다.
  이동하는 씬은 생성 후 필름스트립(f0/f24/f48/f72/f96/f123)을 눈으로 볼 것.

설정 (JSON): krea_ref/krea_prompt 또는 first, w, h, length, seed, args, start_pose,
shots[{prompt, last_pose | last_prompt | last_image, style, lora, args}],
검사 완화는 sparkle_max / tailjump_max / motion_min.

사용:
    h3_scene.py boundaries --shots run.json --out run
    h3_scene.py chunk --shots run.json --out run --n 1 --first <png> --last <png> [--blur-first 8]
    h3_scene.py join   --shots run.json --out run --parts a_GL.mp4,b_GL.mp4,c_GL.mp4

결과: `<out>_c<i>_00001_.mp4`(원본) · `<out>_c<i>_GL.mp4`(톤 고정) · `<out>_all.mp4`(이어붙임)
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time

# ── machine config: environment only, never hardcoded (vh convention, VH_*) ──
#   VH_H3_HOME        host dir ComfyUI reads/writes: <HOME>/input, <HOME>/output   (required)
#   VH_H3_CONTAINER   docker container running ComfyUI; empty/unset = bare metal
#   VH_H3_COMFY_IN    the input dir AS THE CONTAINER SEES IT   (default /opt/ComfyUI/input)
#   VH_H3_COMFY_OUT   the output dir as the container sees it  (default /opt/ComfyUI/output)
#                     bare metal: both equal the host dirs, set automatically
#   VH_H3_PYTHON      interpreter for krea2_edit.py / h3_run.py (default: this one)
#   VH_H3_COMFY_HOST  ComfyUI API base (default http://localhost:8189)
# The helper scripts live next to this file; DOCS is that directory.
HOME = os.environ.get("VH_H3_HOME", "").rstrip("/")
IN, OUT = f"{HOME}/input", f"{HOME}/output"
DOCS = os.path.dirname(os.path.abspath(__file__))
CONTAINER = os.environ.get("VH_H3_CONTAINER", "")
CIN = os.environ.get("VH_H3_COMFY_IN", "/opt/ComfyUI/input") if CONTAINER else IN
COUT = os.environ.get("VH_H3_COMFY_OUT", "/opt/ComfyUI/output") if CONTAINER else OUT
PY = os.environ.get("VH_H3_PYTHON") or sys.executable


def need_home():
    if not HOME or not os.path.isdir(IN) or not os.path.isdir(OUT):
        sys.exit("★VH_H3_HOME 이 없거나 <HOME>/input, <HOME>/output 이 없다: "
                 f"VH_H3_HOME={HOME!r}. ComfyUI 가 읽고 쓰는 호스트 디렉터리를 가리켜야 한다.")

DEFAULT_LEN = 124     # 5.2초(모델 최소). ★길수록 재생성이 잦다 — 두 검사 통과율
                      #   실측 2026-09-24: 124f 4/8 · 158f 1/4 · 175f 0/2 · 209f 1/3.
START_BLUR = 8.0      # 씬의 **첫 조각 시작 이미지만**. 정지 사진은 입을 안 움직이게 한다.
                      #   경계 키프레임에는 걸지 않는다 — 이음매에 '흐렸다 선명해지는 맥박'이 생긴다.
GRADE_LOCK = 0.8      # 출력 톤 고정 강도
SPARKLE_MAX = 0.8
MOTION_MIN = 1.0      # 미만이면 경고만(자동 폐기 안 함)
TAILJUMP_MAX = 14.0
CROSSFADE = 0         # ★기본 0. 경계 양쪽이 같은 그림이라 섞을 게 없다.
                      #   실측(158f): 이음매 차 0 → 2.9 / 4 → 6.7 / 8 → 31.2.
KEEP = ("Keep her face, hairstyle and outfit exactly the same. Keep the background, the lighting, "
        "the colors, the camera distance and the framing exactly the same — her head stays the same "
        "size in the frame. Only change her pose and expression: ")


def sh(cmd):
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, text=True,
                          capture_output=True).stdout


def util(*args):
    """PyAV 헬퍼 (_h3util.py). 컨테이너면 그 안에서(호스트엔 ffmpeg 도 PyAV 도 없다는
    전제), 베어메탈이면 VH_H3_PYTHON 으로 직접 — 그 인터프리터에 PyAV 가 있어야 한다."""
    if CONTAINER:
        return sh(["docker", "exec", CONTAINER, "python", f"{COUT}/_h3util.py", *map(str, args)])
    return sh([PY, f"{DOCS}/_h3util.py", *map(str, args)])


def ensure_util():
    """컨테이너가 보는 output 에 helper 사본을 둔다. 베어메탈은 DOCS 에서 바로 돈다."""
    if not CONTAINER:
        return
    src, dst = f"{DOCS}/_h3util.py", f"{OUT}/_h3util.py"
    h = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
    if not os.path.exists(dst) or h(src) != h(dst):
        shutil.copy(src, dst)


def emit(obj):
    """기계가 읽을 결과. 에이전트가 이 줄만 파싱한다."""
    print("###JSON###")
    print(json.dumps(obj, ensure_ascii=False))


def stage(path):
    """바깥에서 받은 이미지를 ComfyUI 입력 폴더에 **내용 주소**로 들여놓고 basename 을 돌려준다.

    이름을 내용 해시로 짓는 게 핵심이다. 조각 번호나 씬 이름으로 지으면 서로 다른 그림이
    같은 이름을 갖는 순간(재생성·_v2)부터 어느 쪽이 들어갔는지 알 수 없게 된다. 해시면
    그 사고가 성립하지 않는다 — 같은 이름은 항상 같은 그림이다."""
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        sys.exit(f"★입력 이미지가 없다: {path}")
    if os.path.dirname(path) == IN:
        return os.path.basename(path)
    digest = hashlib.md5(open(path, "rb").read()).hexdigest()[:10]
    dst = f"staged_{digest}.png"
    if not os.path.exists(f"{IN}/{dst}"):
        shutil.copy(path, f"{IN}/{dst}")
        print(f"   들여옴 {os.path.basename(path)} → {dst}")
    return dst


def fit(name, w, h):
    """영상 비율로 가운데 자르기. H3 노드가 first 는 늘리고 last 는 자르므로 비율이 다르면 기하가 어긋난다."""
    from PIL import Image, ImageOps
    im = Image.open(f"{IN}/{name}")
    if abs(im.height / im.width - h / w) < 0.01:
        return name
    dst = f"{os.path.splitext(name)[0]}_fit{w}x{h}.png"
    if not os.path.exists(f"{IN}/{dst}"):
        tw = im.width if im.height / im.width >= h / w else round(im.height * w / h)
        ImageOps.fit(im.convert("RGB"), (tw, round(tw * h / w)), Image.LANCZOS).save(f"{IN}/{dst}")
        print(f"   비율 맞춤 → {dst}")
    return dst


def blur(name, radius):
    from PIL import Image, ImageFilter
    dst = f"{os.path.splitext(name)[0]}_lp{int(radius)}.png"
    if not os.path.exists(f"{IN}/{dst}"):
        Image.open(f"{IN}/{name}").convert("RGB").filter(
            ImageFilter.GaussianBlur(radius)).save(f"{IN}/{dst}")
    return dst


def krea(src, prompt, dst_name, seed, aspect):
    """Krea2 Edit. 프롬프트는 **명령형 편집 지시**여야 한다(서술형이면 인물이 복제된다).

    ★앵커에서 '새로 그리는' 단계는 제외 지시를 무시하고, 기존 이미지를 '편집하는' 단계는
    따른다(실측: 셀카 팔 제거를 시작 이미지에서 3회 연속 실패, 편집본에서는 3회 다 성공).
    빼야 할 것이 있으면 시작 이미지를 깨끗한 경계 한 장에서 편집해 만들 것."""
    if os.path.exists(f"{IN}/{dst_name}"):
        print(f"   재사용 {dst_name}")
        return dst_name
    print(f"   [KREA] {dst_name} <- {src}: {prompt[:70]!r}…")
    r = subprocess.run([PY, f"{DOCS}/krea2_edit.py", f"{IN}/{src}", prompt,
                        "--aspect", aspect, "--mp", "1.5", "--seed", str(seed),
                        "--out", f"{IN}/{dst_name}"])
    if r.returncode or not os.path.exists(f"{IN}/{dst_name}"):
        sys.exit(f"[KREA] 실패 {dst_name}")
    return dst_name


def aspect_of(cfg):
    r = cfg.get("h", 1152) / cfg.get("w", 768)
    table = {"1:1 (Square)": 1.0, "2:3 (Portrait Photo)": 1.5,
             "3:4 (Portrait Standard)": 4/3, "9:16 (Portrait Widescreen)": 16/9}
    return min(table, key=lambda k: abs(table[k] - r))


def boundaries(cfg, out):
    """B_0 … B_n 을 만들고 **절대 경로**로 돌려준다. B_0 = 시작 이미지,
    B_i+1 = 샷 i 의 끝 자세를 Krea 로 그린 것. 전부 시작 이미지에서 편집한다 —
    앞 결과를 물리면 그게 곧 누적 경로가 된다.

    shot["last_image"] 가 있으면 그 이미지를 그대로 경계로 쓴다."""
    w, h, asp = cfg.get("w", 768), cfg.get("h", 1152), aspect_of(cfg)
    if cfg.get("krea_ref"):
        start = krea(cfg["krea_ref"], cfg["krea_prompt"], f"{out}_start.png",
                     int(cfg.get("seed", 777)), asp)
    else:
        start = stage(cfg["first"]) if os.path.sep in cfg["first"] else cfg["first"]
    start = fit(start, w, h)
    bs = [start]
    for i, shot in enumerate(cfg["shots"]):
        if shot.get("last_image"):
            bs.append(fit(stage(shot["last_image"]), w, h))
            continue
        pose = shot.get("last_pose")
        if not pose:
            bs.append(None)                      # 마지막 조각은 끝 목표 없이 둘 수 있다
            continue
        bs.append(fit(krea(start, shot.get("last_prompt") or (KEEP + pose),
                           f"{out}_b{i+1}.png", int(cfg.get("seed", 777)) + 100 + i, asp), w, h))
    return [f"{IN}/{b}" if b else None for b in bs]


def endpoints(cfg, i):
    """프롬프트 꼬리 — 양 끝 상태. 없으면 모델이 끝 상태를 모른 채 중간을 만든다."""
    a = cfg.get("start_pose") if i == 0 else cfg["shots"][i-1].get("last_pose")
    b = cfg["shots"][i].get("last_pose")
    s = ""
    if a: s += f" The clip starts with this exact state: {a}"
    if b: s += f" By the last frame she must be in this exact state: {b}"
    return s


def shot_args(cfg, i):
    """조각 i 에 넘길 h3_run 인자. 공통 `args` + 샷별 조정(style / lora / args).

    ★주의: 조각마다 스타일 LoRA 를 바꾸면 인물 인상과 질감이 조각 경계에서 바뀐다.
      톤은 grade_lock 이 잡지만 화풍은 못 잡는다."""
    shot = cfg["shots"][i]
    common = [str(x) for x in cfg.get("args", [])]
    styles = shot.get("style")
    if styles is not None:
        styles = [styles] if isinstance(styles, str) else list(styles)
        kept, skip = [], False
        for x in common:                      # 공통 --style NAME 쌍 제거
            if skip:
                skip = False; continue
            if x == "--style":
                skip = True; continue
            kept.append(x)
        common = kept + sum([["--style", s] for s in styles], [])
    if shot.get("lora"):
        kept, skip = [], False
        for x in common:
            if skip:
                skip = False; continue
            if x == "--turbo-lora":
                skip = True; continue
            kept.append(x)
        common = kept + ["--turbo-lora", shot["lora"]]
    return common + [str(x) for x in shot.get("args", [])]


def chunk_path(out, i):
    p = f"{OUT}/{out}_c{i}_00001_.mp4"
    return p if os.path.exists(p) else None


def gate(cfg, key, default):
    """검사 기준. 설정에서 덮어쓸 수 있다 — 기본값은 **고정 카메라·앉은 장면**에서 정한 것이라,
    달리기·핸드헬드처럼 화면 전체가 움직이는 씬에서는 정상 동작을 오탐한다."""
    return float(cfg.get(key, default))


def generate(cfg, out, i, first, last, retries=2, bump=0):
    """조각 하나. `first`/`last` 는 **호출자가 준 IN/ 안의 basename** 이다 — 이 함수는
    경로를 만들지 않는다. 스파클이면 시드를 밀어 다시 뽑는다(실측: 0.50~1.6% → 0.2%)."""
    best = None                      # 전부 기준을 못 넘으면 **가장 좋았던 것**을 쓴다
    sp_max, tj_max = gate(cfg, "sparkle_max", SPARKLE_MAX), gate(cfg, "tailjump_max", TAILJUMP_MAX)
    for k in range(retries + 1):
        cmd = [PY, f"{DOCS}/h3_run.py", "--mode", "i2v", "--image", first,
               "--prompt", cfg["shots"][i]["prompt"] + endpoints(cfg, i),
               "--out", f"{out}_c{i}", "--w", str(cfg.get("w", 768)), "--h", str(cfg.get("h", 1152)),
               "--length", str(cfg.get("length", DEFAULT_LEN)),
               "--seed", str(int(cfg.get("seed", 777)) + i + bump + k * 1000)]
        if last:
            cmd += ["--last-image", last]
        cmd += shot_args(cfg, i)
        print(f"\n── 조각 {i}  {first} → {last or '(끝 목표 없음)'}"
              + (f"  [재시도 {k}]" if k else ""))
        t0 = time.time()
        subprocess.run(cmd, check=True, cwd=DOCS)
        cur = chunk_path(out, i)
        if not cur:
            # ComfyUI 는 입력·시드가 같으면 캐시를 돌려주고 파일을 안 쓴다 → 시드를 밀어야 비켜간다
            print("   ★결과 파일 없음(ComfyUI 캐시). 시드를 밀어 다시")
            continue
        base = os.path.basename(cur)
        sp = float(util("sparkle", "--src", f"{COUT}/{base}").strip())
        tj = float(util("tailjump", "--src", f"{COUT}/{base}").strip())
        print(f"   {time.time()-t0:.0f}초 · 밝은 점 {sp:.2f}% (기준 {sp_max}%) · "
              f"뒤쪽 최대 변화 {tj:.1f} (기준 {tj_max})")
        if sp <= sp_max and tj <= tj_max:
            mo = float(util("motion", "--src", f"{COUT}/{base}").strip())
            mo_min = gate(cfg, "motion_min", MOTION_MIN)
            if mo < mo_min:
                print(f"   ⚠ 거의 움직이지 않는다 (프레임간 중앙값 {mo:.2f} < {mo_min}). "
                      f"키프레임이 시작 이미지와 너무 비슷한지 확인할 것 — 자동으로 버리지는 않는다")
            return cur, {"시도": k + 1, "판정": "통과", "밝은점_%": round(sp, 2),
                         "끝부분_변화": round(tj, 1), "프레임간_중앙값": round(mo, 2)}
        keep = f"{OUT}/{out}_c{i}_rej{k}_00001_.mp4"
        os.replace(cur, keep)
        score = max(sp / sp_max, tj / tj_max)      # 둘 중 더 나쁜 쪽으로 순위
        if best is None or score < best[0]:
            best = (score, keep, sp, tj)
        if k < retries:
            print("   ★" + ("스파클" if sp > sp_max else "끝부분 디졸브") + " — 시드를 밀어 다시")
    print(f"   ★{retries + 1}번 모두 기준 미달 — 제일 나은 것(밝은점 {best[2]:.2f}% / 뒤쪽 {best[3]:.1f})을 쓴다. "
          f"다른 시드로 다시 뽑으려면 --seed-bump 3000")
    os.replace(best[1], f"{OUT}/{out}_c{i}_00001_.mp4")
    cur = chunk_path(out, i)
    mo = float(util("motion", "--src", f"{COUT}/{os.path.basename(cur)}").strip())
    return cur, {"시도": retries + 1, "판정": "기준 미달 — 제일 나은 것",
                 "밝은점_%": round(best[2], 2), "끝부분_변화": round(best[3], 1),
                 "프레임간_중앙값": round(mo, 2)}


def graded(out, i, anchor):
    """출력 톤을 시작 이미지에 고정. 원본 조각은 그대로 남는다."""
    raw, gl = chunk_path(out, i), f"{OUT}/{out}_c{i}_GL.mp4"
    if not os.path.exists(gl) or os.path.getmtime(gl) < os.path.getmtime(raw):
        print(util("gradelock", "--src", f"{COUT}/{os.path.basename(raw)}",
                   "--dst", f"{COUT}/{os.path.basename(gl)}", "--anchor", anchor,
                   "--strength", GRADE_LOCK, "--keep-audio").strip())
    return gl


def anchor_for(cfg, out, start_png):
    """톤 기준. 씬의 시작 이미지에서 한 번만 뽑는다."""
    a = f"{IN}/{out}_grade.json"
    if not os.path.exists(a):
        util("anchor", "--src", f"{CIN}/{os.path.basename(start_png)}", "--dst", f"{CIN}/{out}_grade.json")
    return f"{CIN}/{out}_grade.json"


def need_container():
    need_home()
    if not CONTAINER:
        return
    if sh(["docker", "ps", "--filter", f"name={CONTAINER}", "--format", "{{.Status}}"]).strip()[:2] != "Up":
        sys.exit(f"★{CONTAINER} 가 안 떠 있다: docker start {CONTAINER}")


def load(path):
    cfg = json.load(open(path, encoding="utf-8"))
    L = cfg.get("length", DEFAULT_LEN)
    if L < 124 or (L - 124) % 17:
        print(f"★length={L} 는 학습 범위 밖 (124~362, 17 단위)")
    return cfg


def cmd_boundaries(a):
    cfg = load(a.shots); need_container(); ensure_util()
    n = len(cfg["shots"])
    print(f"   조각 {n} × {cfg.get('length', DEFAULT_LEN)}프레임 · "
          f"{cfg.get('w',768)}x{cfg.get('h',1152)} · seed {cfg.get('seed',777)}")
    bs = boundaries(cfg, a.out)
    rows = []
    for i in range(n):
        row = {"n": i + 1, "last": bs[i + 1]}
        if i == 0:
            row["first"] = bs[0]              # 씬의 시작. 컷이 더 있으면 에이전트가 따로 붙인다
        rows.append(row)
    print("\n경계 이미지:")
    for k, b in enumerate(bs):
        if not b:
            continue
        if k == 0:
            where = "   ← 행 1 의 start (씬의 시작)"
        elif k == n:
            where = f"   ← 행 {k} 의 last (씬의 끝)"
        else:
            where = f"   ← 행 {k} 의 last = 행 {k+1} 의 start"
        print(f"  B{k}  {b}{where}")
    print("\n이제 에이전트가 행에 붙인다: update_video_chunk(video_id, n, last_image=<last>).\n"
          "사용자가 탭에서 보고 GO 를 켜면, GO 켜진 행만 chunk 로 생성한다.\n"
          "키프레임을 다시 뽑으려면 그 png 를 지우고 이 명령을 다시.")
    emit({"cmd": "boundaries", "out": a.out, "start": bs[0], "rows": rows})


def cmd_chunk(a):
    cfg = load(a.shots); need_container(); ensure_util()
    i = a.n - 1                                   # 탭의 행 n ↔ 내부 조각 i
    if not (0 <= i < len(cfg["shots"])):
        sys.exit(f"★행 {a.n} 은 이 설정의 범위 밖이다 (샷 {len(cfg['shots'])}개)")
    w, h = cfg.get("w", 768), cfg.get("h", 1152)

    first = fit(stage(a.first), w, h)
    if a.blur_first:
        first = blur(first, a.blur_first)         # 씬의 첫 조각만. 경계에는 걸지 않는다
    last = fit(stage(a.last), w, h) if a.last else None

    anchor = anchor_for(cfg, a.out, a.anchor or a.first)
    if a.force and chunk_path(a.out, i):
        os.replace(chunk_path(a.out, i), f"{OUT}/{a.out}_c{i}_prev_{int(time.time())}.mp4")
    if chunk_path(a.out, i) and not a.force:
        print(f"── 조각 {i} 는 이미 있다 (다시 뽑으려면 --force)")
        raw, metrics = chunk_path(a.out, i), None
    else:
        raw, metrics = generate(cfg, a.out, i, first, last, a.retries, a.seed_bump)
    gl = graded(a.out, i, anchor)
    print(f"\n완료  원본 {raw}\n      톤고정 {gl}")
    emit({"cmd": "chunk", "n": a.n, "i": i, "raw": raw, "graded": gl,
          "metrics": metrics, "first": f"{IN}/{first}", "last": f"{IN}/{last}" if last else None,
          "seed": int(cfg.get("seed", 777)) + i + a.seed_bump,
          "prompt": cfg["shots"][i]["prompt"] + endpoints(cfg, i)})


def cmd_join(a):
    cfg = load(a.shots); need_container(); ensure_util()
    parts = [p.strip() for p in a.parts.split(",") if p.strip()]
    for p in parts:
        if not os.path.exists(p):
            sys.exit(f"★없는 조각: {p}")
    dst = f"{OUT}/{a.out}_all.mp4"
    args = ["concat", "--dst", f"{COUT}/{os.path.basename(dst)}", "--fps", 24, "--drop-first", 1,
            "--trim-settle", "auto", "--trim-head", "auto",
            "--crossfade", a.crossfade, "--keep-audio"]
    for q in parts:
        args += ["--src", f"{COUT}/{os.path.basename(q)}"]
    print(util(*args))
    if a.anchor:
        print(util("grade", "--anchor", a.anchor,
                   *sum([["--src", f"{COUT}/{os.path.basename(q)}"] for q in parts], [])))
    print(f"\n완성  {dst}")
    emit({"cmd": "join", "out": a.out, "path": dst, "parts": parts})


def main():
    p = argparse.ArgumentParser(description="경계 키프레임 체인 H3 씬 생성 (행이 기준)")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("boundaries", help="경계 이미지만 만들고 매니페스트를 돌려준다")
    b.add_argument("--shots", required=True)
    b.add_argument("--out", required=True)
    b.set_defaults(fn=cmd_boundaries)

    c = sub.add_parser("chunk", help="조각 하나를 **받은 경로 그대로** 생성한다")
    c.add_argument("--shots", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--n", type=int, required=True, help="탭의 행 번호 (1-based)")
    c.add_argument("--first", required=True,
                   help="시작 프레임 경로. get_video_chunk_image(n,'start') 가 준 것을 그대로 넘긴다")
    c.add_argument("--last", default=None, help="끝 프레임 경로. 없으면 끝 목표 없이 생성")
    c.add_argument("--blur-first", type=float, default=0.0,
                   help=f"시작 프레임 가우시안 블러 반경. **씬의 첫 조각에만** {START_BLUR} 를 준다 — "
                        f"정지 사진이 입을 안 움직이게 한다. 경계 키프레임에는 걸지 않는다"
                        f"(이음매에 '흐렸다 선명해지는 맥박'이 생긴다)")
    c.add_argument("--anchor", default=None, help="톤 기준 이미지. 기본은 --first")
    c.add_argument("--seed-bump", type=int, default=0)
    c.add_argument("--retries", type=int, default=2)
    c.add_argument("--force", action="store_true", help="이미 있어도 다시 뽑는다(기존 것은 보관)")
    c.set_defaults(fn=cmd_chunk)

    j = sub.add_parser("join", help="조각들을 이어붙인다 (등록은 에이전트가 join_video_chunks 로)")
    j.add_argument("--shots", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--parts", required=True, help="이어붙일 mp4 경로들, 쉼표로")
    j.add_argument("--crossfade", type=int, default=CROSSFADE)
    j.add_argument("--anchor", default=None, help="주면 이어붙인 뒤 톤 보고서를 낸다")
    j.set_defaults(fn=cmd_join)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
