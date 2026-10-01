# SPDX-License-Identifier: Apache-2.0
"""Unit tests for streaming per-rank safetensors weights onto Neuron."""

import struct
import threading
import warnings

import pytest
import torch
from safetensors.torch import save

from vllm_neuron.worker.nemotron_weight_stream import (
    Chunk,
    S3RangeReader,
    TensorSlice,
    header_length,
    parse_header,
    plan_chunks,
    read_index,
    split_s3_uri,
    stream_rank_weights,
    tensors_of,
)


def rank_shard(rank: int) -> dict[str, torch.Tensor]:
    """A small shard with mixed dtypes, distinct per rank."""
    generator = torch.Generator().manual_seed(rank)
    return {
        "embed.weight": torch.randn(16, 8, generator=generator).to(torch.bfloat16),
        "layers.0.mlp.weight": torch.randn(8, 8, generator=generator),
        "layers.0.norm.weight": torch.randn(8, generator=generator).to(torch.float16),
        "rope_cos": torch.randn(4, 2, generator=generator),
        "positions": torch.arange(5, dtype=torch.int64) + rank,
    }


class BytesReader:
    """Serves objects held in memory, recording every range and peak concurrency."""

    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.ranges: list[tuple[str, int, int]] = []
        self._lock = threading.Lock()
        self._in_flight = 0
        self.peak_in_flight = 0

    def read(self, uri: str, start: int, end: int) -> bytes:
        with self._lock:
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
            self.ranges.append((uri, start, end))
        try:
            return self.objects[uri][start:end]
        finally:
            with self._lock:
                self._in_flight -= 1


class FakeNxDModel:
    """Records what the streamer does to an NxDModel, as device buffers would hold it."""

    def __init__(self):
        self.placeholders: list[dict[str, torch.Tensor]] | None = None
        self.device: list[dict[str, torch.Tensor]] = []
        self.loaded = False
        self.writes: list[tuple[int, str]] = []

    def set_weights(self, sharded_checkpoint):
        assert not self.loaded
        self.placeholders = sharded_checkpoint
        self.device = [
            {name: torch.zeros_like(tensor) for name, tensor in shard.items()}
            for shard in sharded_checkpoint
        ]

    def to_neuron(self):
        self.loaded = True

    def write_to_neuron_buffer(self, tensor, buffer_key, rank):
        assert self.loaded, "write_to_neuron_buffer before to_neuron"
        target = self.device[rank][buffer_key]
        assert tensor.shape == target.shape and tensor.dtype == target.dtype
        target.copy_(tensor)
        self.writes.append((rank, buffer_key))


def objects_for(
    shards: list[dict[str, torch.Tensor]],
) -> tuple[dict[str, bytes], list[str]]:
    uris = [
        f"s3://models/artifact/weights_rank{rank}.safetensors"
        for rank in range(len(shards))
    ]
    return {uri: save(shard) for uri, shard in zip(uris, shards)}, uris


@pytest.mark.parametrize("chunk_bytes", [1, 200, 1 << 30])
def test_every_tensor_of_every_rank_lands_in_its_device_buffer(chunk_bytes):
    shards = [rank_shard(0), rank_shard(1)]
    objects, uris = objects_for(shards)
    model = FakeNxDModel()

    stream_rank_weights(
        model, BytesReader(objects), uris, chunk_bytes=chunk_bytes, prefetch=2
    )

    assert model.loaded
    for rank, shard in enumerate(shards):
        assert model.device[rank].keys() == shard.keys()
        for name, expected in shard.items():
            assert torch.equal(model.device[rank][name], expected), (rank, name)
    assert sorted(model.writes) == sorted(
        (rank, name) for rank, shard in enumerate(shards) for name in shard
    )


def test_placeholders_carry_shape_and_dtype_only():
    shards = [rank_shard(0)]
    objects, uris = objects_for(shards)
    model = FakeNxDModel()

    stream_rank_weights(model, BytesReader(objects), uris)

    for name, expected in shards[0].items():
        placeholder = model.placeholders[0][name]
        assert placeholder.shape == expected.shape
        assert placeholder.dtype == expected.dtype


def test_chunks_fetched_ahead_never_exceed_prefetch():
    shards = [rank_shard(0)]
    objects, uris = objects_for(shards)
    reader = BytesReader(objects)

    stream_rank_weights(FakeNxDModel(), reader, uris, chunk_bytes=1, prefetch=2)

    assert reader.peak_in_flight <= 2
    # Two header reads, then one GET per tensor at a 1-byte budget.
    assert len(reader.ranges) == 2 + len(shards[0])


def test_the_header_is_read_with_two_ranged_gets():
    objects, uris = objects_for([rank_shard(0)])
    reader = BytesReader(objects)

    index = read_index(reader, uris[0])

    (length,) = struct.unpack("<Q", objects[uris[0]][:8])
    assert reader.ranges == [(uris[0], 0, 8), (uris[0], 8, 8 + length)]
    assert {tensor.name for tensor in index} == set(rank_shard(0))
    assert index == sorted(index, key=lambda tensor: tensor.start)
    assert index[0].start == 8 + length


def slices(*spans: tuple[int, int]) -> list[TensorSlice]:
    return [
        TensorSlice(f"t{i}", torch.uint8, (end - start,), start, end)
        for i, (start, end) in enumerate(spans)
    ]


def test_adjacent_tensors_share_a_get_within_the_budget():
    chunks = plan_chunks(slices((10, 20), (20, 30), (30, 70), (70, 75)), chunk_bytes=25)
    assert [(chunk.start, chunk.end) for chunk in chunks] == [
        (10, 30),
        (30, 70),
        (70, 75),
    ]
    assert [len(chunk.tensors) for chunk in chunks] == [2, 1, 1]


def test_a_gap_between_tensors_starts_a_new_get():
    chunks = plan_chunks(slices((10, 20), (24, 30)), chunk_bytes=1 << 20)
    assert [(chunk.start, chunk.end) for chunk in chunks] == [(10, 20), (24, 30)]


def test_a_chunk_of_the_wrong_length_is_refused():
    chunk = Chunk(0, 4, tuple(slices((0, 4))))
    with pytest.raises(ValueError, match="returned 3 bytes"):
        list(tensors_of(chunk, b"abc"))


def header(entries: dict) -> bytes:
    import json

    return json.dumps(entries).encode()


def test_a_tensor_whose_bytes_disagree_with_its_shape_is_refused():
    with pytest.raises(ValueError, match="needs 8"):
        parse_header(
            header({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}})
        )


def test_unknown_dtypes_and_overlapping_tensors_are_refused():
    with pytest.raises(ValueError, match="unsupported safetensors dtype"):
        parse_header(
            header({"w": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [0, 1]}})
        )
    with pytest.raises(ValueError, match="overlap"):
        parse_header(
            header(
                {
                    "a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
                    "b": {"dtype": "U8", "shape": [4], "data_offsets": [2, 6]},
                }
            )
        )


def test_metadata_is_skipped_and_offsets_are_absolute():
    raw = header(
        {
            "__metadata__": {"format": "pt"},
            "w": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]},
        }
    )
    (tensor,) = parse_header(raw)
    assert (tensor.name, tensor.start, tensor.end) == (
        "w",
        8 + len(raw),
        8 + len(raw) + 3,
    )


def test_a_header_length_safetensors_would_not_write_is_refused():
    with pytest.raises(ValueError):
        header_length(struct.pack("<Q", 0))
    with pytest.raises(ValueError):
        header_length(struct.pack("<Q", 1 << 40))
    with pytest.raises(ValueError):
        header_length(b"short")


def test_s3_uris_split_into_bucket_and_key():
    assert split_s3_uri("s3://models/neuron/a/weights_rank0.safetensors") == (
        "models",
        "neuron/a/weights_rank0.safetensors",
    )
    for invalid in ["models/a", "s3://models", "s3:///key"]:
        with pytest.raises(ValueError):
            split_s3_uri(invalid)


def test_empty_tensors_get_a_buffer_but_no_get():
    shard = {"empty": torch.zeros(0, 4), **rank_shard(0)}
    objects, uris = objects_for([shard])
    reader = BytesReader(objects)
    model = FakeNxDModel()

    stream_rank_weights(model, reader, uris, chunk_bytes=1)

    assert model.placeholders[0]["empty"].shape == (0, 4)
    assert "empty" not in {name for _, name in model.writes}
    assert len(reader.ranges) == 2 + len(rank_shard(0))
    for name, expected in rank_shard(0).items():
        assert torch.equal(model.device[0][name], expected), name


def test_plan_chunks_skips_empty_tensors():
    chunks = plan_chunks(slices((0, 4), (4, 4), (4, 8)), chunk_bytes=1 << 20)
    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 8)]
    assert [tensor.name for tensor in chunks[0].tensors] == ["t0", "t2"]


def test_tensors_of_views_the_bytes_without_copying_or_warning():
    chunk = Chunk(0, 8, (TensorSlice("w", torch.float32, (2,), 0, 8),))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ((tensor, view),) = tensors_of(chunk, b"\x00\x00\x80\x3f\x00\x00\x00\x40")
    assert tensor.name == "w"
    assert torch.equal(view, torch.tensor([1.0, 2.0]))


class _Body:
    def __init__(self, data: bytes, error: Exception | None = None):
        self._data, self._error = data, error

    def read(self) -> bytes:
        if self._error is not None:
            raise self._error
        return self._data


class FlakyS3Client:
    """get_object that fails in scripted ways before serving the range."""

    def __init__(self, data: bytes, failures: list[Exception | bytes]):
        self.data = data
        self.failures = list(failures)
        self.calls: list[str] = []

    def get_object(self, Bucket: str, Key: str, Range: str):
        self.calls.append(Range)
        start, end = (int(x) for x in Range.removeprefix("bytes=").split("-"))
        if self.failures:
            failure = self.failures.pop(0)
            if isinstance(failure, bytes):
                return {"Body": _Body(failure)}
            if isinstance(failure, ConnectionError):
                return {"Body": _Body(b"", failure)}
            raise failure
        return {"Body": _Body(self.data[start : end + 1])}


def client_error(status: int):
    from botocore.exceptions import ClientError

    return ClientError(
        {
            "Error": {"Code": str(status)},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "GetObject",
    )


def test_dropped_bodies_short_reads_and_5xx_are_retried():
    client = FlakyS3Client(
        bytes(range(64)),
        [ConnectionError("reset by peer"), b"\x00\x01", client_error(503)],
    )
    naps: list[float] = []

    data = S3RangeReader(client, attempts=4, sleep=naps.append).read("s3://b/k", 8, 16)

    assert data == bytes(range(8, 16))
    assert client.calls == ["bytes=8-15"] * 4
    assert len(naps) == 3


def test_a_missing_key_is_not_retried():
    client = FlakyS3Client(b"", [client_error(404)])
    with pytest.raises(Exception, match="404"):
        S3RangeReader(client, attempts=4, sleep=lambda _: None).read("s3://b/k", 0, 8)
    assert len(client.calls) == 1


def test_the_last_attempt_raises():
    client = FlakyS3Client(b"", [ConnectionError("a"), ConnectionError("b")])
    with pytest.raises(ConnectionError, match="b"):
        S3RangeReader(client, attempts=2, sleep=lambda _: None).read("s3://b/k", 0, 8)
