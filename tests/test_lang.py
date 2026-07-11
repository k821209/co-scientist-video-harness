"""Both news builders must expose `lang` (default ko) and feed it to transcribe,
so non-Korean shorts align correctly instead of force-decoding as Korean."""
import inspect
from vh.steps import news


def test_builders_expose_lang_default_ko():
    for fn in (news.build_short, news.build_clip_short):
        assert inspect.signature(fn).parameters["lang"].default == "ko"


def test_lang_is_passed_to_transcribe_not_hardcoded():
    for fn in (news.build_short, news.build_clip_short):
        src = inspect.getsource(fn)
        assert "language=lang" in src
        assert 'transcribe(vo, language="ko")' not in src
