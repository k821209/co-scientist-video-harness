"""lint_copy — the check that asks whether the script MAKES SENSE.

The scripts here are the ones that actually shipped broken. That is the point:
a lint built only against synthetic cases tells you it is working right up until
it isn't. Both of the traps noted below were fallen into during development and
were caught by these scripts, not by reasoning.
"""
from vh import qc


# 2026-08-22, Odyssey episode. Passed lint_vo, passed narration_match at 0.96,
# and a listener asked "proposed to WHOM?" — Penelope had been compressed out of
# the script entirely.
BROKEN = ("그리고 결말입니다. 오디세우스는 돌아와 문을 잠그고 청혼자 백여덟 명을 "
          "전부 죽입니다. 학살이 아니라 화해까지가 이야기입니다.")
FIXED = ("그리고 결말입니다. 오디세우스에게는 페넬로페라는 아내가 있었어요. "
         "남편이 돌아오지 않는 동안, 페넬로페와 결혼하겠다고 몰려온 남자가 "
         "백여덟 명이었습니다. 오디세우스는 돌아와 문을 잠그고 그들을 전부 죽입니다. "
         "그런데 거기서 끝이 아니에요. 죽은 남자들의 아버지와 친척들이 무장하고 "
         "복수하러 몰려옵니다. 그때 제우스가 벼락을 던져 싸움을 멈추고, "
         "아테나가 양쪽을 붙잡아 맹약을 맺게 해요. "
         "학살이 아니라 화해까지가 이야기입니다.")
# 2026-08-23, Pearl Abyss episode. "삼만 육천구백 원" came back as `36,900억 원`
# because every earlier figure in the script was in 억.
UNITS = ("이 분기 매출이 천구백이십육억 원입니다. 영업이익은 육백칠십육억 원이고요. "
         "실적을 발표한 팔월 십일일 종가는 삼만 육천구백 원.")
UNITS_OK = "종가는 삼만 육천구백 원. 다음 날 삼만 천오백오십 원이 됐습니다."
SUBSTR = "롯데의 실점이 늘었습니다. 그런데 롯데의 감소폭이 리그의 두 배가 넘습니다."


def test_the_shipped_broken_script_is_flagged():
    kinds = [h["kind"] for h in qc.lint_copy(BROKEN)]
    assert kinds == ["no_counterpart", "no_counterpart"]


def test_the_rewrite_is_silent():
    # The fix names Penelope, the fathers and kinsmen, and both sides of the
    # oath. A check that flagged this too would be noise.
    assert qc.lint_copy(FIXED) == []


def test_a_verb_ending_is_not_a_counterpart():
    # ★ 와/과/랑/하고 are case particles AND verb endings. Accepting them as
    # counterpart markers let BROKEN pass — "돌아와 문을 잠그고" satisfied the
    # rule. This is that trap, pinned.
    assert any(h["kind"] == "no_counterpart"
               for h in qc.lint_copy("오디세우스는 돌아와 청혼을 거절했습니다."))
    # ...while a real counterpart marker still clears it.
    assert qc.lint_copy("오디세우스는 페넬로페에게 청혼했습니다.") == []


def test_a_lone_small_figure_among_big_ones_is_flagged():
    hits = [h for h in qc.lint_copy(UNITS) if h["kind"] == "money_scale_mix"]
    assert len(hits) == 1
    assert "삼만 육천구백 원" in hits[0]["match"]


def test_a_consistent_script_is_not_flagged():
    # Nothing to be dragged towards, so nothing to warn about.
    assert qc.lint_copy(UNITS_OK) == []


def test_the_numeral_keeps_its_largest_unit():
    # ★ Stripping the unit off the end reads "백십조 원" as a 원-scale figure and
    # then flags the biggest number in the script as the small one.
    assert qc._money_scale("백십조") == 4
    assert qc._money_scale("천구백이십육억") == 3
    assert qc._money_scale("삼만 육천구백") == 2
    hits = qc.lint_copy("예산이 백십조 원입니다. 주가는 삼만 원입니다.")
    assert [h["match"] for h in hits if h["kind"] == "money_scale_mix"] == ["삼만 원"]


def test_a_substring_is_not_a_pronoun():
    # ★ '리그의' matched '그의'.
    assert qc.lint_copy(SUBSTR) == []


def test_names_suppress_counterpart_noise():
    script = "아킬레우스는 화해를 받아들였습니다."
    assert qc.lint_copy(script)
    assert qc.lint_copy(script, names=("아킬레우스",)) == []


def test_it_reads_beats_as_well_as_a_string():
    class Beat:
        def __init__(self, text): self.text = text
    beats = [Beat("이 분기 매출이 천구백이십육억 원입니다."),
             Beat("종가는 삼만 육천구백 원.")]
    # The two figures are in DIFFERENT beats — the whole reason this is not part
    # of lint_vo, which only ever sees one beat at a time.
    assert any(h["kind"] == "money_scale_mix" for h in qc.lint_copy(beats))


def test_it_never_raises_on_junk():
    for junk in ("", "   ", "...", "!!!", "12345"):
        assert qc.lint_copy(junk) == []


def test_a_kinship_word_hiding_inside_a_verb_does_not_silence_the_check():
    # 받아**들**였습니다 contains 아들. With a plain `in` test that counted as a
    # named counterpart and the warning disappeared — a false positive would
    # have been noise, this was silence, which is the failure that matters.
    # 정신/신라 do the same with 신, and 형태 with 형.
    for script in ("아킬레우스는 화해를 받아들였습니다.",
                   "정신을 차리고 청혼을 준비했습니다.",
                   "형태가 달라진 뒤에 화해가 이루어졌습니다."):
        assert [h["kind"] for h in qc.lint_copy(script)] == ["no_counterpart"], script


def test_a_real_kinship_word_still_clears_it():
    for script in ("아들이 아버지에 대한 복수를 마쳤습니다.",
                   "두 사람은 화해했습니다.",
                   "양쪽이 맹약을 맺었습니다."):
        assert qc.lint_copy(script) == [], script


def test_a_relation_word_buried_in_a_compound_does_not_fire():
    # The mirror image of the kinship bug, and the one this fix left behind:
    # PERSON was boundary-guarded and RELATION was not. 자유계약선수 is not a
    # contract with anybody and 손해배상보험 is not a settlement with anybody.
    for script in ("구단은 자유계약선수 세 명을 데려왔습니다.",
                   "손해배상보험에 가입했습니다.",
                   "결혼정보회사를 통해 만났습니다."):
        assert [h for h in qc.lint_copy(script)
                if h["kind"] == "no_counterpart"] == [], script


def test_a_prefixed_relation_word_still_fires():
    # ★ Why the guard is on the TAIL and not the head: a leading (?<![가-힣])
    # would kill 재계약, which is a real relation word with a real prefix — and
    # killing a relation word SILENCES the check, the direction that matters.
    assert [h["match"] for h in qc.lint_copy("재계약을 맺었습니다.")] == ["계약"]
    # A derived noun that still carries the relation counts too: 청혼자 is one
    # of the two real defects in the shipped script.
    assert [h["match"] for h in qc.lint_copy("청혼자 백여덟 명이 몰려왔습니다.")] \
        == ["청혼"]


def test_a_one_syllable_counterpart_is_a_counterpart():
    # "세 명에게" names who it was, and was being missed because 명 is a single
    # syllable and the marker required two. 에게/한테 are datives — whatever
    # precedes them IS the second party, however short.
    assert qc.lint_copy("구단은 세 명에게 백억 원을 썼습니다. 계약을 맺었습니다.") == []
    assert qc.lint_copy("그에게 사과했습니다.") == []


def test_the_lotte_false_positive_is_gone():
    # From a whole-catalogue run over 14 episodes: the only counterpart hit, and
    # it was wrong. The sentence names its counterpart twice over.
    script = ("감독은 그사이 네 번 바뀌었고, 이천이십삼년 초에는 외부 자유계약선수 "
              "세 명에게 백칠십억 원을 썼습니다.")
    assert [h for h in qc.lint_copy(script) if h["kind"] == "no_counterpart"] == []
