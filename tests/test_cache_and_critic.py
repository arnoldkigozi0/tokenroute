import json

from tokenroute.backends import StaticBackend
from tokenroute.cache import AnswerCache, normalize
from tokenroute.router import Router


def test_normalize_variants_collide():
    assert normalize("What is 2+2?") == normalize("  what is 2+2  ")
    assert normalize("Hello   World.") == normalize("hello world")


def test_cache_roundtrip_and_persistence(tmp_path):
    path = tmp_path / "cache.json"
    cache = AnswerCache(str(path))
    assert cache.get("What is 2+2?") is None
    cache.put("What is 2+2?", "4", "local")
    assert cache.get("what is 2+2") == "4"

    reloaded = AnswerCache(str(path))
    assert reloaded.get("WHAT IS 2+2 ?".replace(" ?", "?")) == "4"
    assert len(reloaded) == 1


def test_cache_survives_corrupt_file(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{not json", encoding="utf-8")
    cache = AnswerCache(str(path))
    assert len(cache) == 0
    cache.put("q", "a")
    assert json.loads(path.read_text())


def test_router_cache_hit_costs_zero_tokens():
    local = StaticBackend(reply="a fine local answer", is_local=True, name="local")
    remote = StaticBackend(reply="a remote answer", is_local=False, name="remote")
    router = Router(local, remote, cache=AnswerCache())

    first = router.answer("What is the capital of Uganda?")
    assert first.route == "local"
    assert local.calls == 1

    second = router.answer("  what is the capital of uganda ")
    assert second.route == "cache"
    assert second.answer == first.answer
    assert second.local_tokens == 0 and second.remote_tokens == 0
    assert local.calls == 1


class CriticBackend(StaticBackend):
    """Answers normally, but says NO when asked to critique."""

    def complete(self, prompt, system=None, max_tokens=512):
        if "strict critic" in prompt:
            reply = self.reply_to_critic
            saved, self.reply = self.reply, reply
            try:
                return super().complete(prompt, system, max_tokens)
            finally:
                self.reply = saved
        return super().complete(prompt, system, max_tokens)


def test_self_check_rejection_escalates():
    local = CriticBackend(reply="a plausible but wrong draft", is_local=True, name="local")
    local.reply_to_critic = "NO"
    remote = StaticBackend(reply="a remote answer", is_local=False, name="remote")
    router = Router(local, remote, self_check=True)

    result = router.answer("Tricky question?")
    assert result.answer == "a remote answer"
    assert result.route == "local->remote"
    assert local.calls == 2
    assert result.billable_tokens > 0


def test_self_check_approval_stays_local():
    local = CriticBackend(reply="a correct draft", is_local=True, name="local")
    local.reply_to_critic = "YES"
    remote = StaticBackend(reply="a remote answer", is_local=False, name="remote")
    router = Router(local, remote, self_check=True)

    result = router.answer("Easy question?")
    assert result.route == "local"
    assert result.billable_tokens == 0
    assert remote.calls == 0
