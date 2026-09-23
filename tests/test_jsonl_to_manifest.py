import json
import time

import numpy as np
import pytest

from jsonl_to_manifest import convert, validate_messages
from theseus.data import TokenStream


class FakeTokenizer:
    def __len__(self):
        return 256

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": False, "return_dict": False}
        return [10 + len(messages), 20 + len(messages), 2]


class OutOfOrderTokenizer:
    def __len__(self):
        return 256

    def apply_chat_template(self, messages, **kwargs):
        value = int(messages[-1]["content"])
        time.sleep((15 - value) % 5 * 0.001)
        return [value, 255 - value]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_convert_writes_reader_compatible_manifest(tmp_path):
    tokenizer_path = tmp_path / "model"
    tokenizer_path.mkdir()
    (tokenizer_path / "tokenizer.json").write_text("{}")
    input_path = tmp_path / "train.jsonl"
    write_jsonl(input_path, [
        {"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]},
        {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                      {"role": "assistant", "reasoning_content": "r", "content": "a"}]},
    ])
    manifest_path = convert(input_path, tokenizer_path, tmp_path / "out", tokenizer=FakeTokenizer(),
                            progress_interval=0)
    manifest = json.loads(manifest_path.read_text())
    source = manifest["sources"][0]
    assert manifest["format"] == 1 and manifest["vocab_size"] == 256
    assert source["documents"] == 2 and source["token_count"] == 6
    assert len(manifest["tokenizer_files"]["tokenizer.json"]) == 64
    np.testing.assert_array_equal(np.load(tmp_path / "out" / "corpus.offsets.npy"), [0, 3, 6])
    stream = TokenStream(manifest_path, pair_id=0, pairs=1, chunk_tokens=8, context_tokens=8)
    assert stream.next()["ids"].shape == (1, 3)


@pytest.mark.parametrize("messages, text", [
    ([{"role": "assistant", "content": "a"}], "start with a user"),
    ([{"role": "user", "content": "u"}], "contain user and assistant"),
    ([{"role": "user", "content": "u"}, {"role": "assistant", "content": "a", "extra": 1}],
     "unsupported fields"),
    ([{"role": "user", "content": "u"}, {"role": "assistant", "content": " "}],
     "non-empty string"),
])
def test_schema_errors_are_explicit(messages, text):
    with pytest.raises(ValueError, match=text):
        validate_messages({"messages": messages}, 7)


def test_failed_conversion_removes_partial_output(tmp_path):
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        convert(input_path, tmp_path, tmp_path / "out", tokenizer=FakeTokenizer(), progress_interval=0)
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "out.building").exists()


def test_threaded_conversion_preserves_input_order_and_bytes(tmp_path):
    tokenizer_path = tmp_path / "model"
    tokenizer_path.mkdir()
    input_path = tmp_path / "train.jsonl"
    write_jsonl(input_path, [
        {"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": str(i)}]}
        for i in range(16)
    ])
    one = convert(input_path, tokenizer_path, tmp_path / "one", tokenizer=OutOfOrderTokenizer(),
                  workers=1, progress_interval=0)
    many = convert(input_path, tokenizer_path, tmp_path / "many", tokenizer=OutOfOrderTokenizer(),
                   workers=8, progress_interval=0)
    assert (one.parent / "corpus.bin").read_bytes() == (many.parent / "corpus.bin").read_bytes()
    np.testing.assert_array_equal(np.load(one.parent / "corpus.offsets.npy"),
                                  np.load(many.parent / "corpus.offsets.npy"))
    tokens = np.fromfile(many.parent / "corpus.bin", dtype="<u4").reshape(-1, 2)
    np.testing.assert_array_equal(tokens[:, 0], np.arange(16))
