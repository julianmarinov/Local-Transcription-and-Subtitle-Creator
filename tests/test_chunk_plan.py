from chunking import MIN_TAIL_SECONDS, plan_chunks


def test_plan_covers_whole_duration_without_gaps():
    plan = plan_chunks(50 * 60, chunk_minutes=15)
    assert plan[0][0] == 0.0
    for (s1, l1), (s2, _) in zip(plan, plan[1:]):
        assert s1 + l1 == s2
    last_start, last_len = plan[-1]
    assert last_start + last_len == 50 * 60


def test_short_tail_is_folded_into_previous_chunk():
    plan = plan_chunks(30 * 60 + 20, chunk_minutes=15)  # 20 s tail
    assert len(plan) == 2
    assert plan[-1] == (15 * 60, 15 * 60 + 20)


def test_long_enough_tail_stays_separate():
    plan = plan_chunks(30 * 60 + MIN_TAIL_SECONDS + 1, chunk_minutes=15)
    assert len(plan) == 3


def test_single_chunk_for_short_file():
    assert plan_chunks(10.0, chunk_minutes=15) == [(0.0, 10.0)]
