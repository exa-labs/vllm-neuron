# SPDX-License-Identifier: Apache-2.0
"""Stream per-rank safetensors weights from S3 onto NeuronCores.

A ``nemotron-nxd-tp2-v1`` artifact keeps its weights out of the torchscript
save, in one safetensors file per tensor-parallel rank, so an inf2.xlarge
(16 GiB of host RAM) never holds the ~17 GiB of weights at once. Loading
them from local files means first copying them onto the node's disk. This
module reads them from S3 instead, through whatever endpoint the standard
``AWS_ENDPOINT_URL_S3`` names (for example an S3-compatible cache on the
node), so no copy of the weights is ever written by the pod:

1. Read each rank file's safetensors header with two ranged GETs: the
   8-byte little-endian header length, then the JSON header naming every
   tensor's dtype, shape and byte range.
2. Hand ``NxDModel.set_weights`` placeholders of the right shapes and
   dtypes, and ``to_neuron()`` allocates and initializes the device buffers
   from them. The placeholders are ``torch.empty`` allocations that are
   never written, so the kernel backs them with nothing until they are
   read, and reads see the shared zero page: they cost address space, not
   host memory.
3. Fetch the tensors in byte-range chunks of at most ``chunk_bytes`` and
   write each into its device buffer in place with
   ``NxDModel.write_to_neuron_buffer``. At most ``prefetch`` chunks are in
   flight, which bounds host memory independently of the model's size.

``write_to_neuron_buffer`` copies a tensor verbatim, so this only applies to
artifacts compiled without a weight layout transformation (plain
``ModelBuilder.compile``), which ``nemotron-nxd-tp2-v1`` is.
"""

import json
import logging
import struct
import time
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Protocol

import torch
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

#: safetensors dtype names (``safetensors/src/tensor.rs`` ``Dtype``) that
#: torch can view a byte buffer as.
SAFETENSORS_DTYPES: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}

#: Default size of one ranged GET: large enough that request overhead is
#: noise, small enough that ``prefetch`` of them sit comfortably in host
#: memory next to the serving process.
DEFAULT_CHUNK_BYTES = 256 * 1024 * 1024

#: Default number of chunks fetched ahead of the device writes.
DEFAULT_PREFETCH = 4

#: Default number of times one ranged GET is attempted before the load fails.
DEFAULT_READ_ATTEMPTS = 4

#: The safetensors format caps its JSON header at 100 MB; a larger declared
#: length means the bytes are not a safetensors file.
MAX_HEADER_BYTES = 100 * 1000 * 1000


@dataclass(frozen=True)
class TensorSlice:
    """One tensor of a safetensors file: where its bytes are, and what they are."""

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    #: Absolute byte offsets in the file, ``end`` exclusive.
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Chunk:
    """A run of byte-adjacent tensors fetched with one ranged GET."""

    start: int
    end: int
    tensors: tuple[TensorSlice, ...]


class ShortRead(Exception):
    """A ranged GET returned fewer bytes than the range it was asked for."""


class RangeReader(Protocol):
    """Reads ``[start, end)`` of an object; what the streamer fetches through."""

    def read(self, uri: str, start: int, end: int) -> bytes: ...


class NeuronWeightTarget(Protocol):
    """The part of ``NxDModel`` the streamer drives."""

    def set_weights(
        self, sharded_checkpoint: list[dict[str, torch.Tensor]]
    ) -> None: ...

    def to_neuron(self) -> None: ...

    def write_to_neuron_buffer(
        self, tensor: torch.Tensor, buffer_key: str, rank: int
    ) -> None: ...


def header_length(prefix: bytes) -> int:
    """The JSON header's length, from a safetensors file's first 8 bytes."""
    if len(prefix) != 8:
        raise ValueError(
            f"expected the 8-byte safetensors prefix, got {len(prefix)} bytes"
        )
    (length,) = struct.unpack("<Q", prefix)
    if length == 0 or length > MAX_HEADER_BYTES:
        raise ValueError(f"implausible safetensors header length {length}")
    return length


def parse_header(header: bytes) -> list[TensorSlice]:
    """Every tensor a safetensors JSON header declares, in file order.

    Offsets in the header are relative to the end of the header; the
    returned slices carry absolute file offsets. A tensor whose byte range
    disagrees with its dtype and shape, an unknown dtype, or ranges that
    overlap are errors: streaming such a file would write garbage into the
    device buffers.
    """
    data_start = 8 + len(header)
    entries: dict[str, Any] = json.loads(header)
    slices = []
    for name, entry in entries.items():
        if name == "__metadata__":
            continue
        dtype_name = entry["dtype"]
        dtype = SAFETENSORS_DTYPES.get(dtype_name)
        if dtype is None:
            raise ValueError(
                f"tensor {name}: unsupported safetensors dtype {dtype_name}"
            )
        shape = tuple(int(dim) for dim in entry["shape"])
        begin, end = (int(offset) for offset in entry["data_offsets"])
        numel = 1
        for dim in shape:
            numel *= dim
        expected = numel * torch.empty((), dtype=dtype).element_size()
        if end - begin != expected:
            raise ValueError(
                f"tensor {name}: byte range [{begin}, {end}) holds {end - begin} bytes, "
                f"but {dtype_name}{list(shape)} needs {expected}"
            )
        slices.append(
            TensorSlice(name, dtype, shape, data_start + begin, data_start + end)
        )
    slices.sort(key=lambda tensor: tensor.start)
    for before, after in pairwise(slices):
        if after.start < before.end:
            raise ValueError(f"tensors {before.name} and {after.name} overlap")
    return slices


def plan_chunks(slices: list[TensorSlice], chunk_bytes: int) -> list[Chunk]:
    """Group file-ordered tensors into ranged GETs of at most ``chunk_bytes``.

    Consecutive tensors share a GET while their bytes are adjacent and the
    GET stays within budget; a tensor larger than the budget is a GET of its
    own. Every tensor with bytes lands in exactly one chunk; an empty tensor
    has nothing to fetch and its device buffer nothing to receive.
    """
    if chunk_bytes <= 0:
        raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}")
    chunks: list[Chunk] = []
    current: list[TensorSlice] = []
    for tensor in slices:
        if tensor.nbytes == 0:
            continue
        if current and (
            tensor.start != current[-1].end
            or tensor.end - current[0].start > chunk_bytes
        ):
            chunks.append(Chunk(current[0].start, current[-1].end, tuple(current)))
            current = []
        current.append(tensor)
    if current:
        chunks.append(Chunk(current[0].start, current[-1].end, tuple(current)))
    return chunks


def placeholders(slices: list[TensorSlice]) -> dict[str, torch.Tensor]:
    """One never-written tensor per slice, for ``set_weights`` to size the device buffers."""
    return {
        tensor.name: torch.empty(tensor.shape, dtype=tensor.dtype) for tensor in slices
    }


def tensors_of(chunk: Chunk, data: bytes) -> Iterator[tuple[TensorSlice, torch.Tensor]]:
    """The chunk's tensors as views over its fetched bytes."""
    if len(data) != chunk.end - chunk.start:
        raise ValueError(
            f"ranged read of [{chunk.start}, {chunk.end}) returned {len(data)} bytes"
        )
    for tensor in chunk.tensors:
        numel = tensor.nbytes // torch.empty((), dtype=tensor.dtype).element_size()
        with warnings.catch_warnings():
            # The views are only ever read from (``copy_`` sources), so the
            # read-only ``bytes`` need not be copied into a writable buffer.
            warnings.simplefilter("ignore", UserWarning)
            view = torch.frombuffer(
                data, dtype=tensor.dtype, count=numel, offset=tensor.start - chunk.start
            )
        yield tensor, view.reshape(tensor.shape)


def read_index(reader: RangeReader, uri: str) -> list[TensorSlice]:
    """A safetensors object's tensor index, from its header alone."""
    length = header_length(reader.read(uri, 0, 8))
    return parse_header(reader.read(uri, 8, 8 + length))


def stream_rank_weights(
    model: NeuronWeightTarget,
    reader: RangeReader,
    rank_uris: list[str],
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    prefetch: int = DEFAULT_PREFETCH,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Load ``model``'s weights from one safetensors object per rank onto Neuron.

    ``rank_uris[r]`` holds rank ``r``'s shard. On return every tensor of
    every shard has been written into its device buffer and the model is
    ready to serve.
    """
    if prefetch <= 0:
        raise ValueError(f"prefetch must be positive, got {prefetch}")
    started = clock()
    indexes = [read_index(reader, uri) for uri in rank_uris]
    model.set_weights([placeholders(index) for index in indexes])
    model.to_neuron()
    allocated = clock()
    total = 0
    with ThreadPoolExecutor(max_workers=prefetch) as pool:
        for rank, (uri, index) in enumerate(zip(rank_uris, indexes)):
            chunks = plan_chunks(index, chunk_bytes)
            pending: list[tuple[Chunk, Future[bytes]]] = []
            for chunk in chunks:
                pending.append(
                    (chunk, pool.submit(reader.read, uri, chunk.start, chunk.end))
                )
                if len(pending) >= prefetch:
                    total += _write_chunk(model, rank, *pending.pop(0))
            while pending:
                total += _write_chunk(model, rank, *pending.pop(0))
    finished = clock()
    logger.info(
        "streamed %d weight bytes for %d ranks: device allocation %.1fs, "
        "streaming %.1fs (%.0f MB/s)",
        total,
        len(rank_uris),
        allocated - started,
        finished - allocated,
        total / max(finished - allocated, 1e-9) / 1e6,
    )


def _write_chunk(
    model: NeuronWeightTarget, rank: int, chunk: Chunk, fetched: Future[bytes]
) -> int:
    """Write one fetched chunk's tensors into rank ``rank``'s device buffers."""
    data = fetched.result()
    for tensor, view in tensors_of(chunk, data):
        model.write_to_neuron_buffer(view, tensor.name, rank)
    return len(data)


def split_s3_uri(uri: str) -> tuple[str, str]:
    """``s3://bucket/key`` as ``(bucket, key)``."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// URI: {uri}")
    bucket, _, key = uri[len("s3://") :].partition("/")
    if not bucket or not key:
        raise ValueError(f"s3 URI needs a bucket and a key: {uri}")
    return bucket, key


class S3RangeReader:
    """Ranged GETs through boto3, which honors ``AWS_ENDPOINT_URL_S3``.

    With the endpoint pointed at an S3-compatible cache on the node, repeat
    loads are served from that cache; otherwise the reads go to S3.

    botocore only retries the request itself; a connection dropped while the
    body is being read, or a body shorter than the range, surfaces as an
    exception here. Transport errors (``OSError``, botocore's own errors,
    5xx and throttling responses) and short bodies are retried by re-issuing
    the ranged GET up to ``attempts`` times in total, with a short backoff.
    Anything else, including other client errors (a missing key, denied
    access), fails at once.
    """

    def __init__(
        self,
        client: Any = None,
        *,
        attempts: int = DEFAULT_READ_ATTEMPTS,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client("s3", config=Config(max_pool_connections=32))
        if attempts <= 0:
            raise ValueError(f"attempts must be positive, got {attempts}")
        self._client = client
        self._attempts = attempts
        self._sleep = sleep

    def read(self, uri: str, start: int, end: int) -> bytes:
        bucket, key = split_s3_uri(uri)
        for attempt in range(1, self._attempts + 1):
            try:
                response = self._client.get_object(
                    Bucket=bucket, Key=key, Range=f"bytes={start}-{end - 1}"
                )
                data = response["Body"].read()
                if len(data) != end - start:
                    raise ShortRead(
                        f"ranged read of {uri} [{start}, {end}) returned {len(data)} bytes"
                    )
                return data
            except (OSError, BotoCoreError, ClientError, ShortRead) as error:
                if attempt == self._attempts or not _is_transient(error):
                    raise
                logger.warning(
                    "ranged read of %s [%d, %d) failed on attempt %d/%d: %s",
                    uri,
                    start,
                    end,
                    attempt,
                    self._attempts,
                    error,
                )
                self._sleep(min(2.0**attempt, 10.0))
        raise AssertionError("unreachable")


def _is_transient(error: Exception) -> bool:
    """Whether re-issuing the GET could succeed: anything but a non-throttling 4xx."""
    if not isinstance(error, ClientError):
        return True
    status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 500)
    return status >= 500 or status == 429
