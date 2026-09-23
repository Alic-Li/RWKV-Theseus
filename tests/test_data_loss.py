import json
import numpy as np
import torch
from theseus.data import TokenStream, digest_file
from theseus.losses import statistics, metrics


def write_dataset(path, vocab=64):
    path.mkdir(parents=True, exist_ok=True)
    values = np.arange(37, dtype=np.uint32) % vocab
    values.tofile(path / "tokens.bin")
    np.save(path / "offsets.npy", np.array([0, 11, 28, 37], dtype=np.int64))
    manifest = {"format": 1, "vocab_size": vocab, "sources": [{"tokens": "tokens.bin", "offsets": "offsets.npy", "weight": 1,
                "tokens_sha256": digest_file(path / "tokens.bin"), "offsets_sha256": digest_file(path / "offsets.npy")}]}
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path / "manifest.json"


def test_reader_resume_and_stage_replay(tmp_path):
    manifest = write_dataset(tmp_path)
    a = TokenStream(manifest, 0, 2, 3, 9)
    first = a.next()
    saved = a.state_dict()
    expected = [a.next() for _ in range(10)]
    b = TokenStream(manifest, 0, 2, 3, 9)
    b.load_state_dict(saved)
    for batch in expected:
        actual = b.next()
        torch.testing.assert_close(actual.pop("ids"), batch.pop("ids"))
        assert actual == batch
    restart = TokenStream(manifest, 0, 2, 3, 9)
    torch.testing.assert_close(first["ids"], restart.next()["ids"])


def test_sampler_avoids_short_tails_for_documents_at_least_one_chunk(tmp_path):
    manifest = write_dataset(tmp_path)
    stream = TokenStream(manifest, 0, 1, chunk_tokens=3, context_tokens=9, seed=9)
    for _ in range(40):
        assert stream.next()["ids"].shape[1] == 3


def test_token_weighted_loss_and_mask():
    target = torch.tensor([[[1., 1.], [10., 10.], [0., 0.]]])
    pred = torch.tensor([[[2., 2.], [0., 0.], [100., 100.]]])
    stats = statistics(pred, target, mask=torch.tensor([[1., 1., 0.]]))
    result = metrics(stats)
    assert result["nmse"] == 1 and result["rrms"] == 1 and result["tokens"] == 2


def test_singleton_stage_schedule_is_broadcast_to_all_stages():
    from theseus.config import steps_for
    assert steps_for({"stage_steps": [5]}, 15) == 5
    assert steps_for({"stage_steps": [3, 7]}, 1) == 7


def test_epoch_all_tokens_once_with_tails_and_resume(tmp_path):
    from theseus.data import EpochTokenStream
    manifest = write_dataset(tmp_path)
    seen = []
    for rank in range(7):
        reader = EpochTokenStream(manifest, rank, 7, 4, 7)
        first = reader.next()
        if first:
            seen.extend(first['ids'].flatten().tolist())
        state = reader.state_dict()
        resumed = EpochTokenStream(manifest, rank, 7, 4, 7)
        resumed.load_state_dict(state)
        while True:
            batch, other = reader.next(), resumed.next()
            if batch is None:
                assert other is None
                break
            torch.testing.assert_close(batch['ids'], other['ids'])
            assert batch['reset'] == (batch['position'] == 0)
            assert batch['position'] + batch['ids'].numel() <= 7
            seen.extend(batch['ids'].flatten().tolist())
        assert reader.state_dict() == resumed.state_dict()
    assert sorted(seen) == list(range(37))
