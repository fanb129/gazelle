from AAAICOTB.annotations import split_sequence_indices


def test_sequence_split_is_disjoint_deterministic_and_complete():
    sequences = [{"path": f"sequence_{index}"} for index in range(20)]
    train_first, validation_first = split_sequence_indices(sequences, 0.10, 9102)
    train_second, validation_second = split_sequence_indices(sequences, 0.10, 9102)
    assert (train_first, validation_first) == (train_second, validation_second)
    assert not set(train_first) & set(validation_first)
    assert sorted(train_first + validation_first) == list(range(20))
    assert len(validation_first) == 2
