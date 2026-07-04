import pytest

from tokenroute.backends import StaticBackend
from tokenroute.router import Router, complexity_score, looks_sane


def make_router(local_reply="a fine local answer", local_fail=False, threshold=0.75):
    local = StaticBackend(reply=local_reply, is_local=True, name="local", fail=local_fail)
    remote = StaticBackend(reply="a remote answer", is_local=False, name="remote")
    return Router(local, remote, escalate_threshold=threshold), local, remote


def test_easy_task_stays_local():
    router, local, remote = make_router()
    result = router.answer("What is 2+2?")
    assert result.route == "local"
    assert result.answer == "a fine local answer"
    assert local.calls == 1
    assert remote.calls == 0
    assert result.billable_tokens == 0
    assert result.local_tokens > 0


def test_bad_local_draft_escalates():
    router, local, remote = make_router(local_reply="")
    result = router.answer("What is 2+2?")
    assert result.route == "local->remote"
    assert result.answer == "a remote answer"
    assert local.calls == 1
    assert remote.calls == 1
    assert result.billable_tokens == result.remote_tokens > 0


def test_local_backend_error_escalates():
    router, local, remote = make_router(local_fail=True)
    result = router.answer("What is 2+2?")
    assert result.answer == "a remote answer"
    assert remote.calls == 1


def test_hard_task_skips_local():
    router, local, remote = make_router()
    hard = (
        "Prove step-by-step that the sum of the first n odd numbers is n^2, "
        "then derive a theorem generalizing it. " + "Also explain each step. " * 30
    )
    result = router.answer(hard)
    assert result.route == "remote"
    assert local.calls == 0
    assert remote.calls == 1


def test_refusal_draft_escalates():
    router, _, remote = make_router(local_reply="I cannot help with that.")
    result = router.answer("Summarize this sentence.")
    assert result.answer == "a remote answer"
    assert remote.calls == 1


@pytest.mark.parametrize(
    "prompt,expect_high",
    [
        ("What is the capital of France?", False),
        ("Prove the theorem step-by-step and derive ```code``` " + "with many steps " * 40, True),
    ],
)
def test_complexity_score_ordering(prompt, expect_high):
    score = complexity_score(prompt)
    assert (score >= 0.75) == expect_high
    assert 0.0 <= score <= 1.0


def test_looks_sane_rejects_degenerate_output():
    assert looks_sane("Kampala", "capital?")
    assert not looks_sane("", "anything")
    assert not looks_sane("the the the the the the the the the the the the", "x")
    assert not looks_sane("I'm sorry, I can't do that.", "x")
