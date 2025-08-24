#!/usr/bin/env python3
import sys
from collections import Counter
from gguf.gguf_reader import GGUFReader

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <model.gguf>")
    sys.exit(1)

path = sys.argv[1]
reader = GGUFReader(path)

counter = Counter()
for tensor in reader.tensors:
    counter[tensor.tensor_type.name] += 1

total = sum(counter.values())
print(f"=== Quantization Stats for {path} ===")
for ttype, count in counter.items():
    pct = (count / total) * 100
    print(f" - {ttype:7}: {count:6} tensors  ({pct:.2f}%)")

print(f"Total tensors: {total}")
