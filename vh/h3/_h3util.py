"""PyAV 헬퍼. 컨테이너 안에서(호스트에 ffmpeg/PyAV 가 없을 때) 또는 VH_H3_PYTHON 으로 돈다.

출력 디렉터리가 호스트와 공유되므로 h3_long.py 가 이 파일을 거기 써 두고
`docker exec ... python /opt/ComfyUI/output/_h3util.py` 로 부른다.
"""
import argparse
import json
import os
from fractions import Fraction

import av
import numpy as np

OUT = "/opt/ComfyUI/output"
IN = "/opt/ComfyUI/input"


def _laplacian_var(rgb):
    """3x3 라플라시안 커널의 분산. 노이즈·고주파 지표. 문서 실측(같은 클립 6조각): 288→796."""
    g = rgb.astype(np.float32).mean(axis=2)
    h, w = g.shape
    pad = np.pad(g, 1, mode="edge")
    # (0,-1,0; -1,4,-1; 0,-1,0)
    out = (4 * pad[1:1+h, 1:1+w]
           - pad[0:h, 1:1+w] - pad[2:2+h, 1:1+w]
           - pad[1:1+h, 0:w] - pad[1:1+h, 2:2+w])
    return float(out.var())


def _auto_lowpass(cur_lap, base_lap):
    """라플라시안 비율 → blur radius. 문서 근거(1024폭): radius 8 이 최적.
    ratio ≤ 1.2 → 0px (필요 없음), 1.5 → 2px, 2.0 → 4px, 2.5 → 6px, 그 이상 → 8px.
    """
    if base_lap <= 0:
        return 0.0
    r = cur_lap / base_lap
    if r <= 1.2: return 0.0
    if r <= 1.5: return 2.0
    if r <= 2.0: return 4.0
    if r <= 2.5: return 6.0
    return 8.0


def _match_hist(img, anchor_hist):
    """채널별 히스토그램 매칭. mean/std 매칭보다 강함.
    커뮤니티(KJNodes ColorMatch)에서 'hm' 모드로 검증된 표준 방식.
    'mvgd' 는 더 강하지만 어두워지는 부작용이 있어 'hm' 이 권장.
    anchor_hist: {"r":[256],"g":[256],"b":[256]} 정규화 히스토그램.
    """
    out = np.empty_like(img)
    for c, key in enumerate(("r", "g", "b")):
        src_hist, _ = np.histogram(img[..., c].ravel(), bins=256, range=(0, 256))
        src_cdf = src_hist.cumsum().astype(np.float64)
        src_cdf /= max(src_cdf[-1], 1)
        tgt_cdf = np.cumsum(anchor_hist[key])
        tgt_cdf /= max(tgt_cdf[-1], 1e-12)
        # src bin i 의 CDF 값이 tgt CDF 에서 어느 bin j 에 매핑되는지 (역 CDF)
        lut = np.interp(src_cdf, tgt_cdf, np.arange(256)).clip(0, 255).astype(np.uint8)
        out[..., c] = lut[img[..., c]]
    return out


def _compute_hist(img):
    """채널별 정규화 히스토그램 (256 bin)."""
    return {
        key: (np.histogram(img[..., c].ravel(), bins=256, range=(0, 256))[0]
              / max(img.shape[0] * img.shape[1], 1)).round(6).tolist()
        for c, key in enumerate(("r", "g", "b"))
    }


def _match_tone(img, anchor):
    """채널별 (mean, std) 매칭. 구 앵커 파일 호환용 폴백.
    실측(kink30c back c2): R-14 G-18 B-20 편이 발생 → 이 함수로 원복.
    hist 매칭보다 약하다. 새 앵커에는 hist 도 저장되고 그쪽이 우선.
    """
    out = img.astype(np.float32)
    for c in range(3):
        m_now = out[..., c].mean()
        s_now = out[..., c].std() + 1e-6
        m_a = anchor["mean"][c]
        s_a = max(anchor["std"][c], 1.0)
        out[..., c] = (out[..., c] - m_now) / s_now * s_a + m_a
    return np.clip(out, 0, 255).astype(np.uint8)


def last_frame(src, dst, lowpass=0.0, anchor=None, blend_ref=None, blend_weight=0.0, no_tone=False,
               detail_cap=0.0, lc_ref=0.0):
    """마지막 프레임을 PNG 로 저장. ★재압축 손실을 안 쌓으려고 PNG 다.

    ★★lowpass — 조건 이미지에 저역통과(가우시안 블러)를 건다. 두 문제를 한 번에 잡는다:
      ① **정지·선명한 조건 이미지는 동작을 억제한다.** i2v 가 t2v 보다 정적인 게 알려진
         현상이고(arXiv 2506.08456 ALG), 원인은 「입력 이미지의 고주파에 일찍 노출돼
         정적 외형에 과적합하는 지름길로 샘플링이 치우치는 것」이다.
         우리 실측(같은 시드): 블러 0→0.0215, 2px→0.0250, 4px→0.0337, **8px→0.0947**.
      ② **고주파가 되먹임되며 커지는 것**도 막는다. 블러 없이 6조각을 이으면
         라플라시안 분산이 288→796 으로 2.8배가 된다.

    ★★lowpass="auto" — anchor 의 baseline 라플라시안과 비교해 자동 결정한다.
      anchor 미주어짐이나 baseline 없으면 0 으로 폴백.

    ★★anchor — JSON 경로. 톤 앵커링:
      · 파일 없음: 이 이미지의 (mean/std/lap) 을 저장 (첫 청크 = 앵커 확정).
      · 파일 있음: 이 이미지를 앵커 톤에 매칭한 뒤 저장. lowpass=auto 도 여기 값 사용.
    """
    c = av.open(src)
    img = None
    for f in c.decode(video=0):
        img = f.to_ndarray(format="rgb24")
    c.close()
    if img is None:
        raise SystemExit(f"프레임 없음: {src}")

    cur_lap = _laplacian_var(img)
    anchor_data = None
    if anchor:
        if os.path.exists(anchor):
            anchor_data = json.load(open(anchor))
        else:
            # 첫 청크 — 앵커 확정
            stats = _anchor_stats(img, os.path.basename(src))
            os.makedirs(os.path.dirname(anchor) or ".", exist_ok=True)
            json.dump(stats, open(anchor, "w"), indent=2)
            m = stats["mean"]; l = stats["lap"]
            print(f"tone_anchor saved -> {os.path.basename(anchor)} lap={l} mean={m}")
            anchor_data = stats  # 첫 청크: 자기 자신에 매칭·auto = no-op

    # 톤 매칭 (앵커가 이미 있을 때만). hist 우선, 없으면 mean/std (구 앵커 호환).
    # ★no_tone — gradelock 된 청크에서 뽑을 때. 이미 앵커 분위수에 맞춰졌으므로
    #   여기서 전체 hist 매칭을 또 걸면 오히려 과보정된다. 앵커는 lowpass=auto 용으로만 쓴다.
    tone_note = ""
    if anchor_data is not None and not no_tone:
        before_mean = img.reshape(-1, 3).mean(axis=0)
        if "hist" in anchor_data:
            img = _match_hist(img, anchor_data["hist"])
            method = "hm"
        else:
            img = _match_tone(img, anchor_data)
            method = "mean/std"
        d_before = (before_mean - np.array(anchor_data["mean"])).round(1).tolist()
        tone_note = f"  톤편차{d_before}→{method}매칭"

    # C-lite: 원본 참조와 알파 블렌드. 자세는 마지막 프레임(img), 톤은 원본 쪽으로 pull.
    #   pushback 억제 목적. blend_weight=0.25 정도가 실용 기본값.
    if blend_ref and blend_weight > 0 and os.path.exists(blend_ref):
        from PIL import Image as _Im
        ref = np.array(_Im.open(blend_ref).convert("RGB"))
        if ref.shape != img.shape:
            ref = np.array(_Im.fromarray(ref).resize((img.shape[1], img.shape[0]), _Im.LANCZOS))
        w = float(blend_weight)
        blended = ((1 - w) * img.astype(np.float32) + w * ref.astype(np.float32))
        img = blended.clip(0, 255).astype(np.uint8)
        tone_note += f"  +blend원본{w:.2f}"

    # lowpass 자동 결정
    if isinstance(lowpass, str) and lowpass.strip().lower() == "auto":
        base_lap = (anchor_data or {}).get("lap", 0)
        lp_val = _auto_lowpass(cur_lap, base_lap)
        ratio = (cur_lap / base_lap) if base_lap > 0 else 0.0
        print(f"  auto_lowpass lap={cur_lap:.0f} base={base_lap} ratio={ratio:.2f} -> radius={lp_val}")
    else:
        try: lp_val = float(lowpass)
        except (TypeError, ValueError): lp_val = 0.0

    from PIL import Image, ImageFilter
    out = Image.fromarray(img)
    if lp_val > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=lp_val))

    # ★detail_cap — 인계 프레임 국소대비(σ≈8px) 상한 = 기준 lc × detail_cap. 줄이기만 한다.
    #   조각마다 쌓이는 건 이 대역이고(인계 lc 14.3→19.4, 5조각), 인계 프레임이 그 통로다.
    #   lowpass_chain(2px) 은 이 대역을 거의 못 누른다.
    #   ★lowpass **뒤**에 잰다 — 모델이 보는 건 최종 조건 이미지다. 앞에서 재면 정상 청크
    #     (lowpass 후 14.3)도 17.6 으로 잡혀 매번 걸린다.
    #   ★출력 프레임에 걸면 헤일로(뿌연 번짐)가 보이지만, 인계 프레임은 모델이 디테일을
    #     다시 그리므로 결과물에 남지 않는다.
    #   lc_ref — 기준 lc 를 직접 준다(시작 이미지 값). 없으면 anchor 의 lc.
    base_lc = lc_ref or (anchor_data or {}).get("lc") or 0.0
    if detail_cap > 0 and base_lc > 0:
        arr = np.asarray(out)
        cur_lc = _lc(arr)
        g = min(1.0, base_lc * detail_cap / max(cur_lc, 1e-3))
        if g < 0.999:
            L = _luma(arr)
            d = L - _gblur(L, _LC_SIGMA * arr.shape[1] / _LC_W)
            out = Image.fromarray((arr.astype(np.float32) + ((g - 1.0) * d)[..., None])
                                  .clip(0, 255).astype(np.uint8))
        tone_note += f"  lc {cur_lc:.1f}→상한{base_lc * detail_cap:.1f}(gain {g:.2f})"
    out.save(dst)
    print(f"last_frame {os.path.basename(src)} -> {os.path.basename(dst)} "
          f"{img.shape[1]}x{img.shape[0]}  lap={cur_lap:.0f} lp={lp_val}{tone_note}")


def probe(src):
    c = av.open(src)
    vs = c.streams.video[0]
    n = sum(1 for _ in c.decode(video=0))
    fps = float(vs.average_rate or 24)
    has_a = len(c.streams.audio) > 0
    c.close()
    print(f"{os.path.basename(src)}\t{n}\t{fps:g}\t{'audio' if has_a else 'silent'}")


def _hist_from_source(path):
    """이미지 파일 → hist dict, 또는 앵커 JSON → hist dict."""
    if path.lower().endswith(".json"):
        data = json.load(open(path))
        if "hist" not in data:
            raise SystemExit(f"앵커 JSON 에 hist 없음 (구버전 mean/std 만): {path}")
        return data["hist"]
    from PIL import Image
    img = np.array(Image.open(path).convert("RGB"))
    return _compute_hist(img)


def hmatch(src, dst, anchor_hist, strength=1.0, keep_audio=True):
    """★사후 히스토그램 매칭. 완성된 mp4 의 매 프레임을 앵커 hist 에 매칭한 새 mp4 로 저장.
       청크 내부 pushback + chain 경계 튐 둘 다 잡는 유일한 근본 처방.
       - strength=1.0: 완전 매칭 (권장)
       - strength<1: (1-s)*원본 + s*매칭 alpha blend (flicker 우려 시 완화)
       오디오는 그대로 복사한다.
    """
    ic = av.open(src)
    ivs = ic.streams.video[0]
    W, Hh = ivs.width, ivs.height
    fps = ivs.average_rate or Fraction(24, 1)
    has_audio = keep_audio and bool(ic.streams.audio)
    arate = ic.streams.audio[0].rate if has_audio else None

    oc = av.open(dst, "w")
    ov = oc.add_stream("libx264", rate=Fraction(fps))
    ov.width, ov.height, ov.pix_fmt = W, Hh, "yuv420p"
    ov.options = {"crf": "17", "preset": "medium"}
    oa = oc.add_stream("aac", rate=arate) if has_audio else None

    n = 0
    for f in ic.decode(video=0):
        img = f.to_ndarray(format="rgb24")
        matched = _match_hist(img, anchor_hist)
        if strength < 1.0:
            out = ((1.0 - strength) * img.astype(np.float32)
                   + strength * matched.astype(np.float32)).clip(0, 255).astype(np.uint8)
        else:
            out = matched
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(out), format="rgb24")
        for pkt in ov.encode(vf):
            oc.mux(pkt)
        n += 1
    for pkt in ov.encode(None):
        oc.mux(pkt)
    ic.close()

    # 오디오는 별도 pass 로 원본에서 그대로 복사 (프레임 재인코딩만 필요)
    if has_audio:
        c2 = av.open(src)
        rs = av.AudioResampler(format="fltp", layout="stereo", rate=arate)
        for fr in c2.decode(audio=0):
            for rf in rs.resample(fr):
                for pkt in oa.encode(rf):
                    oc.mux(pkt)
        for pkt in oa.encode(None):
            oc.mux(pkt)
        c2.close()

    oc.close()
    print(f"hmatch {os.path.basename(src)} -> {os.path.basename(dst)}  "
          f"{n}frames  strength={strength}{' +audio' if has_audio else ''}")


# ════════════════════════════════════════════════════════════════════
# ★grade lock — 청크 **안** 톤 드리프트까지 잡는 사후 보정
# ════════════════════════════════════════════════════════════════════
# 왜 필요한가 (2026-09-22 실측):
#   H3 i2v 는 청크 앞 1/4 구간에서 luma std 를 +6~17 올린다 (office_pov c0 58.7→76.6).
#   tone_anchor 는 **인계 프레임 한 장**만 앵커에 맞추고, 그 앵커도 c0 의 **마지막**
#   프레임(이미 굳어진 대비)에서 잡혔다. 그래서 청크마다 "굳은 톤에서 출발 → 더 굳음"이
#   되풀이되고, 청크 안 변화는 전혀 안 건드린다.
# 처방:
#   ① 앵커를 **블러 전 시작 이미지**에서 잡는다 (`anchor` 명령).
#   ② 완성된 청크의 **매 프레임**을 앵커 분위수에 맞춘다. 전체 히스토그램 매칭(hmatch)은
#      구도가 바뀌면(일어서기 등) 톤을 억지로 비틀고 프레임마다 깜빡이므로, 채널별 분위수
#      13개만 맞추고 그 궤적을 시간축으로 평활한다 → 부드러운 단조 곡선 하나.
#   ③ 다음 청크 인계 프레임도 이 보정본에서 뽑는다 → 굳은 톤이 되먹임되지 않는다.
_GL_Q = np.array([0.5, 2, 5, 10, 20, 35, 50, 65, 80, 90, 95, 98, 99.5])


def _pct(img):
    """채널별 분위수 (3, len(_GL_Q))."""
    return np.stack([np.percentile(img[..., c], _GL_Q) for c in range(3)])


_LC_W = 768      # 국소대비는 이 폭으로 맞춰 잰다 (시작 이미지와 영상 해상도가 다르다)
_LC_SIGMA = 8.0  # σ≈8px 대역 — 2026-09-22 실측에서 청크마다 쌓인 게 이 대역(lc8 18→26)


def _luma(rgb):
    return rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def _gblur(x, sigma):
    """float32 2D 가우시안 블러 (PIL 은 8bit 라 detail 이 뭉개진다 → 분리형 컨볼루션)."""
    r = int(3 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)
    k /= k.sum()
    p = np.pad(x, r, mode="reflect")
    h = np.zeros((p.shape[0], x.shape[1]), np.float32)
    for i in range(2 * r + 1):
        h += k[i] * p[:, i:i + x.shape[1]]
    v = np.zeros_like(x)
    for i in range(2 * r + 1):
        v += k[i] * h[i:i + x.shape[0], :]
    return v


def _lc(rgb, sigma=_LC_SIGMA):
    """국소대비 = std(L - blur(L)). 폭 _LC_W 기준으로 잰다."""
    L = _luma(rgb)
    if L.shape[1] != _LC_W:
        from PIL import Image
        hh = round(L.shape[0] * _LC_W / L.shape[1])
        L = np.asarray(Image.fromarray(L).resize((_LC_W, hh), Image.BILINEAR), np.float32)
    return float((L - _gblur(L, sigma)).std())


def _anchor_stats(img, src_name):
    return {
        "lc": round(_lc(img), 3),
        "mean": img.reshape(-1, 3).mean(axis=0).round(2).tolist(),
        "std":  img.reshape(-1, 3).std(axis=0).round(2).tolist(),
        "lap":  round(_laplacian_var(img), 2),
        "hist": _compute_hist(img),
        "pct":  _pct(img).round(2).tolist(),
        "src":  src_name,
    }


def make_anchor(src, dst):
    """이미지 한 장 → 앵커 JSON. ★블러 **전** 시작 이미지를 줄 것.
    lap 은 lowpass=auto 의 baseline 이 된다."""
    from PIL import Image
    img = np.asarray(Image.open(src).convert("RGB"))
    st = _anchor_stats(img, os.path.basename(src))
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    json.dump(st, open(dst, "w"), indent=2)
    print(f"anchor {os.path.basename(src)} -> {os.path.basename(dst)}  "
          f"mean={st['mean']} std={st['std']} lap={st['lap']}")


def _smooth(x, win):
    """(T, ...) 를 시간축 중앙 이동평균. ★가장자리에선 창을 **대칭으로** 줄여 t=0·끝은 자기 값.

    ★한쪽만 자르면(창 [0, t+h]) 첫 프레임 LUT 가 뒤쪽 프레임에 끌려간다. H3 는 청크 앞
      20프레임 안에 밝기가 확 오르는 일이 흔해서(drift_C c2: 96→135) 첫 프레임만 어둡게
      눌려 **이음매에 어두운 깜빡임**이 났다(c1 끝 99 → c2 f0 81 → c2 중간 105)."""
    if win <= 1 or len(x) < 2:
        return x
    h = win // 2
    T = len(x)
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    out = np.empty_like(x)
    for t in range(T):
        r = min(h, t, T - 1 - t)
        out[t] = (c[t + r + 1] - c[t - r]) / (2 * r + 1)
    return out


def _lut_from_pct(src_p, tgt_p, strength):
    """분위수 쌍 → 채널별 256 LUT (단조). strength 로 항등과 섞는다."""
    ident = np.arange(256, dtype=np.float64)
    luts = []
    for c in range(3):
        xs = np.concatenate([[0.0], src_p[c], [255.0]])
        ys = np.concatenate([[0.0], tgt_p[c], [255.0]])
        xs = np.maximum.accumulate(xs + np.arange(len(xs)) * 1e-3)  # 엄격 증가
        ys = np.maximum.accumulate(ys)
        lut = np.interp(ident, xs, ys)
        luts.append(((1 - strength) * ident + strength * lut).clip(0, 255))
    return np.stack(luts).round().astype(np.uint8)


def gradelock(src, dst, anchor, strength=0.8, smooth=25, keep_audio=True, detail=0.0):
    """완성된 청크 mp4 의 매 프레임을 앵커 분위수에 맞춘다 (오디오는 원본 복사).
    strength 0.8 기본 — 1.0 은 구도가 크게 바뀌는 청크에서 과보정한다.

    ★detail — 국소대비 상한. 전역 곡선(분위수)은 **국소대비를 못 건드린다.** 그런데
      청크마다 실제로 쌓이는 건 σ≈8px 대역의 국소대비다 (2026-09-22 실측, bus_girl 과 같은
      설정: 인계 프레임 lc 14.6→15.3→19.3→20.0, 조각 끝 18→26). 그래서 매 프레임
      lc 를 재고, 앵커 lc × detail 을 **넘는 만큼만** (L - blur(L)) 성분을 줄인다.
      키우지는 않는다(gain ≤ 1). 이득은 시간축으로 평활. 0 이면 끈다.
      ★기본 0(끔) — 출력 프레임에 gain 0.6 대로 걸면 헤일로(뿌연 번짐)가 보인다(A c3 실측).
        누적 차단은 인계 프레임 쪽(last --detail-cap)이 맡는다."""
    if anchor.endswith(".json"):
        d = json.load(open(anchor))
        if "pct" not in d:
            raise SystemExit(f"앵커 JSON 에 pct 없음 (구버전) — `anchor` 명령으로 다시 만들 것: {anchor}")
        tgt = np.array(d["pct"])
    else:
        from PIL import Image
        tgt = _pct(np.asarray(Image.open(anchor).convert("RGB")))
    ic = av.open(src)
    ivs = ic.streams.video[0]
    W, Hh = ivs.width, ivs.height
    fps = ivs.average_rate or Fraction(24, 1)
    frames = [f.to_ndarray(format="rgb24") for f in ic.decode(video=0)]
    ic.close()
    P = _smooth(np.stack([_pct(f) for f in frames]), smooth)
    G = None
    if detail > 0:
        a_lc = json.load(open(anchor)).get("lc") if anchor.endswith(".json") else None
        if not a_lc:
            print("  ★앵커에 lc 없음 — detail lock 생략 (`anchor` 로 다시 만들 것)")
        else:
            cap = a_lc * detail
            lcs = np.array([_lc(f) for f in frames])
            G = _smooth(np.minimum(1.0, cap / np.maximum(lcs, 1e-3)), smooth)

    has_audio = False
    if keep_audio:
        c0 = av.open(src); has_audio = bool(c0.streams.audio)
        arate = c0.streams.audio[0].rate if has_audio else None; c0.close()
    oc = av.open(dst, "w")
    ov = oc.add_stream("libx264", rate=Fraction(fps))
    ov.width, ov.height, ov.pix_fmt = W, Hh, "yuv420p"
    ov.options = {"crf": "17", "preset": "medium"}
    oa = oc.add_stream("aac", rate=arate) if has_audio else None
    for t, (f, p) in enumerate(zip(frames, P)):
        if G is not None and G[t] < 0.999:
            # 폭이 _LC_W 와 다르면 σ 도 같은 비율로 — 잰 대역과 줄이는 대역을 맞춘다
            L = _luma(f)
            d = L - _gblur(L, _LC_SIGMA * f.shape[1] / _LC_W)
            f = (f.astype(np.float32) + ((G[t] - 1.0) * d)[..., None]).clip(0, 255).astype(np.uint8)
        lut = _lut_from_pct(p, tgt, strength)
        out = np.stack([lut[c][f[..., c]] for c in range(3)], axis=-1)
        for pkt in ov.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(out), format="rgb24")):
            oc.mux(pkt)
    for pkt in ov.encode(None):
        oc.mux(pkt)
    if has_audio:
        c2 = av.open(src)
        rs = av.AudioResampler(format="fltp", layout="stereo", rate=arate)
        for fr in c2.decode(audio=0):
            for rf in rs.resample(fr):
                for pkt in oa.encode(rf):
                    oc.mux(pkt)
        for pkt in oa.encode(None):
            oc.mux(pkt)
        c2.close()
    oc.close()
    d0 = np.abs(P[0] - tgt).mean(); d1 = np.abs(P[-1] - tgt).mean()
    print(f"gradelock {os.path.basename(src)} -> {os.path.basename(dst)}  {len(frames)}frames "
          f"s={strength} win={smooth}  분위수편차 f0={d0:.1f} last={d1:.1f}"
          f"{f'  detail gain {G.min():.2f}~{G.max():.2f}' if G is not None else ''}"
          f"{' +audio' if has_audio else ''}")


def lcmax(src, step=6):
    """영상에서 가장 큰 국소대비 값 하나. 스파클(화면 전체 반짝이는 노이즈) 검출용 —
    실측: 정상 조각 최대 16~21, 스파클 조각 30~34 (같은 씬·같은 앵커 기준)."""
    c = av.open(src)
    v = max(_lc(f.to_ndarray(format="rgb24"))
            for i, f in enumerate(c.decode(video=0)) if i % step == 0)
    c.close()
    print(f"{v:.2f}")
    return v


def sparkle(src, step=4):
    """스파클 지표 — **밝은 점 비율**(%) 의 최댓값. det = L - blur(L, σ4), det>60 인 픽셀 비율.

    ★lc(σ8 표준편차)는 약한 스파클을 못 잡는다. 실측(2026-09-24):
      깨끗 0.17~0.23% / 눈에 보이는 약한 반짝임 0.50% / 심한 스파클 1.19~1.63%
      (같은 사례의 lc 는 16~17 / 22.6 / 31~33 으로 경계가 모호했다.)
    """
    c = av.open(src)
    best = 0.0
    for i, f in enumerate(c.decode(video=0)):
        if i % step: continue
        a = f.to_ndarray(format="rgb24"); L = _luma(a)
        best = max(best, float(((L - _gblur(L, 4.0)) > 60).mean() * 100))
    c.close()
    print(f"{best:.3f}")
    return best


def tailjump(src, tail=40):
    """조각 **뒤쪽**에서 가장 큰 프레임 간 변화. 모델이 목표 키프레임에 닿으려고 화면 안에서
    **디졸브(교차 용해)** 를 해 버리는 것을 잡는다 — 겹쳐 보이는 유령 프레임이 생기고
    재생하면 툭 끊긴 것처럼 보인다. 이어붙이기로는 못 고치고(잘라내면 구도가 다른 지점과
    붙어 더 큰 구멍), 시드를 바꿔 다시 뽑아야 한다.
    실측(2026-09-24, 158f): 정상 조각 뒤쪽 최대 2~4 / 디졸브 낀 조각 16.1.
    """
    c = av.open(src)
    fs = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    c.close()
    seg = fs[-tail:] if len(fs) > tail else fs
    d = [float(np.abs(seg[i+1].astype(np.float32) - seg[i].astype(np.float32)).mean())
         for i in range(len(seg) - 1)]
    print(f"{max(d):.2f}")
    return max(d)


def motion(src):
    """프레임 간 변화의 **중앙값** — 조각이 실제로 움직이는지. 실측(2026-09-25):
    카페 씬 정상 2.46~3.03 / park 씬(키프레임 3장이 서로 비슷해 거의 정지) 0.31~0.88."""
    c = av.open(src)
    fs = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    c.close()
    d = [float(np.abs(fs[i+1].astype(np.float32) - fs[i].astype(np.float32)).mean())
         for i in range(len(fs) - 1)]
    print(f"{float(np.median(d)):.3f}")
    return float(np.median(d))


def grade_report(paths, anchor=None):
    """★톤 지표 — 자세·구도 변화에 둔감한 통계만. `drift`(레퍼런스 픽셀차)는 내용 변화가
    섞여 톤 판단에 못 쓴다. 청크마다 f0·중간·끝 프레임의 luma std, p2/p98, 국소대비(lc8)."""
    def st(rgb):
        L = _luma(rgb)
        return L.mean(), L.std(), np.percentile(L, 2), np.percentile(L, 98), _lc(rgb)
    if anchor:
        a = json.load(open(anchor))
        print(f"앵커({a.get('src')}) mean={a['mean']} std={a['std']} lc={a.get('lc')}")
    print("조각\t프레임\tmean\tstd\tp2\tp98\tlc")
    for i, p in enumerate(paths):
        c = av.open(p); fs = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]; c.close()
        for k in (0, len(fs) // 2, len(fs) - 1):
            print(f"c{i}\t{k}\t" + "\t".join(f"{v:.1f}" for v in st(fs[k])))


def _tail_cut(frames, nxt_first=None, still_thr=0.6, still_cap=24, jump_win=15, jump_min=25.0):
    """조각 **끝**에서 버릴 프레임 수. 앞쪽은 절대 자르지 않는다(다음 조각 앞엔 멈춤이 없다).

    ① **멈춤** — 목표 키프레임에 도달한 뒤 굳는 구간(4~12프레임 실측). 잘라도 안전하다.
    ② **막판 되돌림** — 끝에서 구도가 한 프레임 만에 키프레임으로 복귀하는 큰 튐.
       209프레임 조각에서 41.5 로 나타났고, 그 앞에서 자르면 이음매로 옮겨져 블렌드로 지워졌다.
       ★단 **충분히 클 때만**(jump_min) 자른다. 작은 되돌림까지 자르면 자른 자리가
       키프레임과 달라져 **더 큰 구멍**이 생긴다 — 158프레임 조각 실측: 원래 이음매 2.9 였는데
       16.1 짜리 되돌림 앞에서 11프레임 자르니 35.6 이 됐다.
    ③ 그래서 `nxt_first`(다음 조각의 첫 프레임)를 주면 **자른 뒤가 더 나쁘면 되돌린다**.
    """
    if len(frames) < 40:
        return 0
    d = np.array([np.abs(frames[i+1].astype(np.float32) - frames[i].astype(np.float32)).mean()
                  for i in range(len(frames) - 1)])
    cut = 0
    for i in range(len(d) - 1, 0, -1):            # ① 끝의 멈춤
        if d[i] < still_thr and cut < still_cap:
            cut += 1
        else:
            break
    end, lo = len(d) - cut, max(0, len(d) - cut - jump_win)
    if end > lo:                                   # ② 큰 되돌림만
        seg = d[lo:end]; j = int(seg.argmax())
        if seg[j] >= jump_min:
            cut = len(d) - (lo + j)
    if cut and nxt_first is not None:              # ③ 손해면 취소
        before = np.abs(frames[-1].astype(np.float32) - nxt_first.astype(np.float32)).mean()
        after = np.abs(frames[-1 - cut].astype(np.float32) - nxt_first.astype(np.float32)).mean()
        if after > before + 1.0:
            print(f"  trim_settle 취소 — 자르면 이음매가 나빠진다 ({before:.1f} → {after:.1f})")
            return 0
    return cut


def _head_still(frames, thr=0.6, cap=8):
    """조각 **앞**의 '아직 안 움직이는' 프레임 수. 경계 키프레임 체인에서 다음 조각은 정지
    이미지에서 출발하므로 움직임이 붙는 데 0.3~0.5초 걸린다 — 이음매 프레임차는 작아도
    **동작이 멈췄다 다시 시작해** 연결이 어색하다(실측 scene3: 이음매 뒤 10프레임 0.1~0.5).
    ★8프레임까지만 자른다 — 더 자르면 실제 동작이 잘려 이음매가 커진다(12프레임: 3.0→4.5)."""
    n = 0
    for i in range(min(cap, len(frames) - 1)):
        if np.abs(frames[i+1].astype(np.float32) - frames[i].astype(np.float32)).mean() < thr:
            n += 1
        else:
            break
    return n


def concat(srcs, dst, fps, drop_first, keep_audio, xfade=0, audio_xfade=0, fade_frames=0,
           trim_settle=0, trim_head=0):
    """조각을 잇는다.

    ★조각 1부터는 첫 프레임이 앞 조각의 마지막 프레임과 같은 그림이라 버린다
      (안 버리면 매 이음매에서 한 프레임 멈춘다).
    ★★**오디오를 조각의 영상 길이에 정확히 맞춘다.** 셋 다 필요하다 —
      ① 앞: 버린 프레임만큼 자른다. 안 자르면 이음매마다 drop_first/fps 초
         (24fps 에서 42ms)씩 밀리고 조각이 쌓이면 누적된다.
      ② 뒤가 남으면 자른다. 안 자르면 **다음 조각 대사를 덮는다.**
      ③ 뒤가 모자라면 무음으로 채운다. 안 채우면 다음 조각 소리가 먼저 시작한다.
      조각마다 대사가 따로 있는 게 이 파이프의 정상 사용법이라 셋 다 실제 문제다.
    """
    first = av.open(srcs[0])
    vs = first.streams.video[0]
    W, Hh = vs.width, vs.height
    arate = first.streams.audio[0].rate if (keep_audio and first.streams.audio) else None
    first.close()

    oc = av.open(dst, "w")
    ov = oc.add_stream("libx264", rate=Fraction(fps))
    ov.width, ov.height, ov.pix_fmt = W, Hh, "yuv420p"
    ov.options = {"crf": "17", "preset": "medium"}
    oa = oc.add_stream("aac", rate=arate) if arate else None

    total = 0
    tail_buffer = []  # 이전 조각 마지막 xfade 프레임 (crossfade 용)
    audio_tail = None  # 이전 조각 오디오 tail (audio crossfade 용)
    for i, p in enumerate(srcs):
        skip = 0 if i == 0 else drop_first
        # ── 영상 ────────────────────────────────────────────────
        c = av.open(p)
        frames = []
        for j, f in enumerate(c.decode(video=0)):
            if j < skip:
                continue
            frames.append(f.to_ndarray(format="rgb24"))
        c.close()
        # ★trim_head: 조각 **앞**의 아직 안 움직이는 프레임을 버린다 (조각 0 제외).
        #   이음매 프레임차는 그대로인데(3.0) 멈춤이 사라져 연결이 자연스러워진다.
        if trim_head and i > 0 and len(frames) > 30:
            h = _head_still(frames) if str(trim_head).lower() == "auto" else int(trim_head)
            if h:
                print(f"  trim_head {os.path.basename(p)} 앞 {h}프레임 버림 (아직 안 움직임)")
                frames = frames[h:]
        # ★trim_settle: 조각 **끝**의 멈춘 프레임을 버린다 (마지막 조각은 제외).
        #   키프레임 체인에서 모델이 목표 키프레임에 도달한 뒤 굳어 있어 이음매마다
        #   0.3~0.5초 멈칫한다(실측 drift_J: 조각별 4/8/12 프레임). 그 구간은 사실상 같은
        #   그림이라 잘라도 연속성은 유지된다. "auto"=자동 검출, 숫자=고정, 0=끔.
        #   ★앞쪽은 자르지 말 것 — 다음 조각 앞부분엔 멈춤이 없어 실제 동작이 잘린다
        #     (양쪽 6프레임씩 잘랐더니 이음매 차 4.3→12.3).
        if trim_settle and i < len(srcs) - 1 and len(frames) > 30:
            # 다음 조각의 첫 프레임(= 실제로 이어질 그림)을 미리 읽어 손익을 비교한다
            nf = None
            c2 = av.open(srcs[i+1])
            for j2, f2 in enumerate(c2.decode(video=0)):
                if j2 >= drop_first:
                    nf = f2.to_ndarray(format="rgb24"); break
            c2.close()
            t = _tail_cut(frames, nf) if str(trim_settle).lower() == "auto" else int(trim_settle)
            if t:
                print(f"  trim_settle {os.path.basename(p)} 끝 {t}프레임 버림 (멈춤/막판 되돌림)")
                frames = frames[:-t]
        # ★crossfade: 이 조각 첫 N 프레임을 이전 조각 tail 과 alpha blend.
        #   시간축은 그대로 유지 (오버랩 없이). 이음매 그림자·톤튐 완화용.
        if i > 0 and xfade > 0 and tail_buffer:
            N = min(xfade, len(frames), len(tail_buffer))
            for k in range(N):
                w = (k + 1) / (N + 1)  # 양 끝값 피해서 0.2~0.8 부드럽게
                a = tail_buffer[-N + k].astype(np.float32)
                b = frames[k].astype(np.float32)
                frames[k] = ((1 - w) * a + w * b).clip(0, 255).astype(np.uint8)
        # ★fade-to-black: 각 청크 boundary 에 검은 fade in/out (움찔 없이 확실히 이음매 가림).
        #   crossfade(alpha) 는 자세 다른 프레임에서 이중 노출. fade-to-black 은 검은 순간 통과.
        if fade_frames > 0:
            fn = min(fade_frames, len(frames) // 2)
            # ★공식: 양 끝 프레임이 정확히 0 (완전 검은) 이어야 이음매에 컷 튐 없다.
            #   기존 (k+1)/(fn+1) 은 끝값 fn/(fn+1) ≈ 0.14 라 여전히 픽셀 남아 자세 차이가 튄다.
            if i > 0 and fn > 0:  # 앞 fade in: k=0 (첫 프레임) → w=0 완전 검은
                for k in range(fn):
                    w = k / (fn - 1) if fn > 1 else 0.0
                    frames[k] = (frames[k].astype(np.float32) * w).clip(0, 255).astype(np.uint8)
            if i < len(srcs) - 1 and fn > 0:  # 뒤 fade out: 마지막 프레임 → w=0 완전 검은
                for k in range(fn):
                    w = k / (fn - 1) if fn > 1 else 0.0
                    idx = len(frames) - 1 - k
                    frames[idx] = (frames[idx].astype(np.float32) * w).clip(0, 255).astype(np.uint8)
        # 인코딩
        nvid = 0
        for arr in frames:
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")
            for pkt in ov.encode(vf):
                oc.mux(pkt)
            nvid += 1
        # 다음 조각 blend 용 tail 저장
        tail_buffer = [arr.copy() for arr in frames[-xfade:]] if xfade > 0 else []
        total += nvid
        if not oa:
            continue
        # ── 오디오: 이 조각의 영상 길이에 **정확히** 맞춘다 ──────
        #   ★앞은 버린 프레임만큼 자르고, 뒤는 남으면 자르고 모자라면 채운다.
        #   길면 다음 조각 대사를 덮고, 짧으면 다음 조각 소리가 먼저 시작한다.
        want = int(round(nvid / float(fps) * arate))
        skip_s = skip / float(fps)
        got = 0
        audio_buf = np.zeros((2, want), dtype=np.float32)
        c2 = av.open(p)
        if c2.streams.audio:
            atb = c2.streams.audio[0].time_base
            rs = av.AudioResampler(format="fltp", layout="stereo", rate=arate)
            for fr in c2.decode(audio=0):
                if got >= want:
                    break
                if skip_s > 0 and fr.pts is not None and float(fr.pts * atb) < skip_s:
                    continue
                for rf in rs.resample(fr):
                    if got >= want:
                        break
                    arr = rf.to_ndarray()                 # (ch, n) fltp
                    n = arr.shape[1]
                    if got + n > want:                    # ★넘치면 잘라 낸다
                        arr = arr[:, : want - got]
                        n = arr.shape[1]
                    if n == 0:
                        break
                    audio_buf[:, got:got+n] = arr
                    got += n
        c2.close()
        # ★audio crossfade: 이음매에서 fade-out/fade-in
        if i > 0 and audio_xfade > 0 and audio_tail is not None:
            N = min(audio_xfade, want, audio_tail.shape[1])
            fade = np.linspace(0.0, 1.0, N, dtype=np.float32)[:, None]
            # current head fades in, previous tail fades out
            audio_buf[:, :N] = (1 - fade.T) * audio_tail[:, -N:] + fade.T * audio_buf[:, :N]
        # mux the (possibly crossfaded) audio buffer
        for chunk_start in range(0, want, 1024):
            chunk_end = min(chunk_start + 1024, want)
            chunk = audio_buf[:, chunk_start:chunk_end]
            of = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(chunk), format="fltp", layout="stereo")
            of.sample_rate = arate
            of.pts = None
            for pkt in oa.encode(of):
                oc.mux(pkt)
        # save tail for next crossfade
        if audio_xfade > 0:
            audio_tail = audio_buf[:, -audio_xfade:].copy()

    for pkt in ov.encode():
        oc.mux(pkt)
    if oa:
        for pkt in oa.encode():
            oc.mux(pkt)
    oc.close()
    print(f"concat {len(srcs)}조각 -> {os.path.basename(dst)}  {total}프레임 {total/fps:.2f}초")


def drift(paths, ref):
    """★조각이 쌓이며 그림이 얼마나 밀려나는지 잰다.
    앞 조각의 마지막 프레임을 그대로 물리므로 이음매 자체는 0이어야 하고,
    원본 레퍼런스 대비 차이는 조각마다 커진다 — 그 값을 보고 몇 조각까지 쓸지 정한다."""
    from PIL import Image
    base = np.asarray(Image.open(ref).convert("RGB"), dtype=np.float32)
    print("조각\t레퍼런스 대비 평균절대차(0~255)")
    for i, p in enumerate(paths):
        c = av.open(p)
        img = None
        for f in c.decode(video=0):
            img = f.to_ndarray(format="rgb24")
        c.close()
        a = np.asarray(Image.fromarray(img).resize(
            (base.shape[1], base.shape[0])), dtype=np.float32)
        print(f"c{i}\t{np.abs(a - base).mean():.1f}")



def lipsync(paths, verbose=True):
    """★★★검증에 실패한 지표다. 판정에 쓰지 말 것 (2026-09-06).

    눈으로 「입이 거의 안 움직인다」고 확인한 클립이 0.487,
    입이 잘 벌어지던 클립이 0.453 으로 **순서가 뒤집혔다.**
    참고용으로만 남긴다. 판정은 `mouthstrip` 으로 눈으로 한다.

    (아래는 원래 의도) 입이 **소리와 함께** 움직이는지 잰다 = 립싱크의 정의.

    얼굴 검출기가 컨테이너에 없다(cv2·mediapipe 없음). 그래서 입을 찾지 않고
    **소리와 상관이 높은 영역을 찾는다.**

      · 프레임을 흑백 48x64 로 줄이고 블록별 |차분| 시계열을 만든다
      · 오디오를 프레임 단위 RMS 포락선으로 만든다
      · 블록마다 둘의 상관계수를 구한다
      · 점수 = 얼굴이 있을 만한 중앙 영역에서 상위 블록 상관의 평균

    ★상관계수라 **화각에 안 흔들린다.** 고정 크롭으로 차분만 재면 얼굴이 클수록
      값이 커져 시드끼리 비교가 안 된다 (2026-09-06 에 실제로 그렇게 틀렸다).
    """
    out = []
    for p in paths:
        c = av.open(p)
        gs = []
        for f in c.decode(video=0):
            a = f.to_ndarray(format="rgb24").astype(np.float32).mean(axis=2)
            h, w = a.shape
            bh, bw = h // 48, w // 32
            a = a[:48 * bh, :32 * bw].reshape(48, bh, 32, bw).mean(axis=(1, 3))
            gs.append(a)
        c.close()
        n = len(gs)
        aud = None
        c2 = av.open(p)
        if c2.streams.audio:
            xs = []
            for fr in c2.decode(audio=0):
                xs.append(fr.to_ndarray().astype(np.float32).mean(axis=0))
            if xs:
                sig = np.concatenate(xs)
                step = max(1, len(sig) // n)
                env = np.array([np.sqrt((sig[i * step:(i + 1) * step] ** 2).mean() + 1e-12)
                                for i in range(n)])
                aud = env
        c2.close()
        if aud is None or n < 8:
            out.append((p, float("nan"), 0.0)); continue

        g = np.stack(gs)                       # (n,48,32)
        d = np.abs(np.diff(g, axis=0))         # (n-1,48,32)
        e = aud[1:]                            # 같은 길이로
        dm = d - d.mean(axis=0, keepdims=True)
        em = e - e.mean()
        denom = (np.sqrt((dm ** 2).sum(axis=0)) * np.sqrt((em ** 2).sum()) + 1e-9)
        corr = (dm * em[:, None, None]).sum(axis=0) / denom      # (48,32)
        # 인물이 있을 만한 중앙 영역만 (가로 25~75%, 세로 8~72%)
        sub = corr[int(48 * 0.08):int(48 * 0.72), int(32 * 0.25):int(32 * 0.75)]
        flat = np.sort(sub.ravel())[::-1]
        score = float(flat[:12].mean())        # 상위 12블록 평균
        # 말이 실제로 있었는지 (무음이면 판정 불가)
        speech = float((aud > aud.max() * 0.25).mean())
        out.append((p, score, speech))
        if verbose:
            print(f"{os.path.basename(p):26s} 립싱크 {score:5.3f}   유성구간 {speech*100:4.0f}%")
    return out



def mouthstrip(src, dst, n=10, y0=0.26, y1=0.40, x0=0.33, x1=0.67):
    """★입 부분만 가로로 이어 붙인 검수용 띠를 만든다.

    자동 판정이 안 되므로(위 lipsync 참조) **눈으로 보는 걸 빠르게** 한다.
    한 장만 열면 말하는 구간에서 입이 벌어지는지 바로 보인다.
    화각이 조각마다 다르므로 기본 상자가 안 맞으면 y0/y1 을 조절할 것.
    """
    from PIL import Image, ImageDraw
    c = av.open(src)
    fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    c.close()
    if not fr:
        raise SystemExit("프레임 없음")
    H, W, _ = fr[0].shape
    idx = [round(i * (len(fr) - 1) / (n - 1)) for i in range(n)]
    crops = []
    for i in idx:
        im = Image.fromarray(fr[i]).crop((int(W * x0), int(H * y0), int(W * x1), int(H * y1)))
        crops.append(im.resize((240, int(240 * im.height / im.width))))
    cw, ch = crops[0].size
    out = Image.new("RGB", (cw * n, ch + 22), (16, 16, 22))
    d = ImageDraw.Draw(out)
    for i, (im, j) in enumerate(zip(crops, idx)):
        out.paste(im, (i * cw, 22))
        d.text((i * cw + 5, 5), f"{j / 24:.2f}s", fill=(230, 230, 240))
    out.save(dst)
    print(f"mouthstrip {os.path.basename(src)} -> {os.path.basename(dst)}  {n}장 {idx}")



def split(src, dstdir, prefix, size, fps):
    """★영상을 size 프레임씩 잘라 둔다. 오디오도 같이 나눈다.

    업스케일을 배치로 돌릴 때 배치마다 원본 전체를 다시 디코드하면 낭비가 크다
    (493프레임 영상에서 64프레임 배치 하나에 350초 걸렸다).
    미리 잘라 두면 배치마다 자기 조각만 읽는다.
    """
    from fractions import Fraction
    c = av.open(src)
    vs = c.streams.video[0]
    W, H = vs.width, vs.height
    frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    c.close()
    arate = None
    aud = None
    c2 = av.open(src)
    if c2.streams.audio:
        arate = c2.streams.audio[0].rate
        xs = [fr.to_ndarray() for fr in c2.decode(audio=0)]
        if xs:
            aud = np.concatenate([x if x.ndim == 2 else x[None, :] for x in xs], axis=1)
    c2.close()
    n = len(frames)
    out = []
    for i, s0 in enumerate(range(0, n, size)):
        seg = frames[s0:s0 + size]
        path = f"{dstdir}/{prefix}_seg{i:02d}.mp4"
        oc = av.open(path, "w")
        ov = oc.add_stream("libx264", rate=Fraction(int(fps)))
        ov.width, ov.height, ov.pix_fmt = W, H, "yuv420p"
        ov.options = {"crf": "14", "preset": "veryfast"}
        oa = oc.add_stream("aac", rate=arate) if aud is not None else None
        for f in seg:
            for pkt in ov.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                oc.mux(pkt)
        if oa is not None:
            a0 = int(s0 / fps * arate)
            a1 = int((s0 + len(seg)) / fps * arate)
            chunk = aud[:, a0:a1]
            step = 1024
            for k in range(0, chunk.shape[1], step):
                blk = np.ascontiguousarray(chunk[:, k:k + step])
                if blk.shape[1] == 0:
                    break
                if blk.shape[0] == 1:
                    blk = np.repeat(blk, 2, axis=0)
                af = av.AudioFrame.from_ndarray(blk.astype(np.float32),
                                                format="fltp", layout="stereo")
                af.sample_rate = arate
                af.pts = None
                for pkt in oa.encode(af):
                    oc.mux(pkt)
        for pkt in ov.encode():
            oc.mux(pkt)
        if oa is not None:
            for pkt in oa.encode():
                oc.mux(pkt)
        oc.close()
        out.append(os.path.basename(path))
        print(f"split {os.path.basename(path)}  {len(seg)}프레임")
    print("TOTAL", n)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["last", "concat", "probe", "drift", "lipsync", "mouthstrip", "split", "hmatch",
                                   "anchor", "gradelock", "grade", "lcmax", "sparkle", "tailjump",
                                   "motion"])
    p.add_argument("--src", action="append", default=[])
    p.add_argument("--dst")
    p.add_argument("--ref")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--drop-first", type=int, default=1)
    p.add_argument("--keep-audio", action="store_true")
    p.add_argument("-n", type=int, default=10)
    p.add_argument("--prefix", default="seg")
    p.add_argument("--y0", type=float, default=0.26)
    p.add_argument("--y1", type=float, default=0.40)
    p.add_argument("--lowpass", default="0.0",
                   help="조건 이미지 저역통과 반지름(px, 1024폭 기준 8 최적). auto 도 가능")
    p.add_argument("--anchor", default=None,
                   help="톤 앵커 JSON 경로. 없으면 만들고, 있으면 매칭에 사용")
    p.add_argument("--blend-ref", default=None,
                   help="원본 참조 이미지 경로. 매칭된 인계 프레임과 알파 블렌드하여 pushback 억제")
    p.add_argument("--blend-weight", type=float, default=0.0,
                   help="원본 블렌드 가중치(0~1). 0.25 실용 기본. 자세는 마지막 프레임 우세")
    p.add_argument("--strength", type=float, default=1.0,
                   help="hmatch: 매칭 강도 0~1. 1.0=완전 매칭(권장), <1=원본과 alpha blend")
    p.add_argument("--smooth", type=int, default=25,
                   help="gradelock: 분위수 궤적 시간 평활 창(프레임). 25≈1초")
    p.add_argument("--detail", type=float, default=0.0,
                   help="gradelock: 국소대비 상한 = 앵커 lc × 이 값. 0=끔. 줄이기만 한다")
    p.add_argument("--detail-cap", type=float, default=0.0,
                   help="last: 인계 프레임 국소대비 상한 = 앵커 lc × 이 값. 0=끔")
    p.add_argument("--lc-ref", type=float, default=0.0,
                   help="last: detail-cap 기준 lc 를 직접 지정 (기본: --anchor 의 lc)")
    p.add_argument("--no-tone", action="store_true",
                   help="last: 톤 매칭 생략 (gradelock 된 청크에서 뽑을 때)")
    p.add_argument("--trim-head", default="0",
                   help="concat: 조각 앞의 아직 안 움직이는 프레임 버리기. auto|숫자|0(끔)")
    p.add_argument("--trim-settle", default="0",
                   help="concat: 조각 끝의 멈춘 프레임 버리기. auto|숫자|0(끔)")
    p.add_argument("--fade-frames", type=int, default=0,
                   help="concat: 각 청크 boundary 에 검은 fade in/out (움찔 방지). 6 = 0.25초")
    p.add_argument("--crossfade", type=int, default=0,
                   help="concat: 조각 이음매에서 N 프레임 crossfade blend (기본 0=off)")
    p.add_argument("--audio-xfade", type=int, default=0,
                   help="concat: 오디오 이음매에서 N 샘플 crossfade fade (기본 0=off)")
    a = p.parse_args()
    if a.cmd == "last":
        last_frame(a.src[0], a.dst, a.lowpass, a.anchor, a.blend_ref, a.blend_weight, a.no_tone,
                   a.detail_cap, a.lc_ref)
    elif a.cmd == "anchor":
        make_anchor(a.src[0], a.dst)
    elif a.cmd == "gradelock":
        gradelock(a.src[0], a.dst, a.anchor, a.strength, a.smooth, a.keep_audio, a.detail)
    elif a.cmd == "motion":
        motion(a.src[0])
    elif a.cmd == "tailjump":
        tailjump(a.src[0])
    elif a.cmd == "sparkle":
        sparkle(a.src[0])
    elif a.cmd == "lcmax":
        lcmax(a.src[0])
    elif a.cmd == "grade":
        grade_report(a.src, a.anchor)
    elif a.cmd == "probe":
        for s in a.src:
            probe(s)
    elif a.cmd == "drift":
        drift(a.src, a.ref)
    elif a.cmd == "lipsync":
        lipsync(a.src)
    elif a.cmd == "split":
        split(a.src[0], a.dst, a.prefix, a.n, a.fps)
    elif a.cmd == "mouthstrip":
        mouthstrip(a.src[0], a.dst, a.n, a.y0, a.y1)
    elif a.cmd == "hmatch":
        if not a.anchor:
            raise SystemExit("hmatch 에는 --anchor <이미지 or JSON> 필요")
        h = _hist_from_source(a.anchor)
        hmatch(a.src[0], a.dst, h, a.strength, a.keep_audio or True)
    else:
        concat(a.src, a.dst, a.fps, a.drop_first, a.keep_audio, a.crossfade, a.audio_xfade,
               a.fade_frames, a.trim_settle, a.trim_head)
