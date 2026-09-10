"""Fixed-order metrics collectives with a fixed-size schema handshake."""
import hashlib
import torch
import torch.distributed as dist


def reduce_metrics(values, device):
    keys = sorted(key for key in values if key != 'lr')
    if not dist.is_available() or not dist.is_initialized():
        return {key: values[key] for key in keys}
    signature = int.from_bytes(hashlib.sha256('\0'.join(keys).encode()).digest()[:7], 'big')
    schema = torch.tensor([len(keys), signature], device=device, dtype=torch.int64)
    schemas = [torch.empty_like(schema) for _ in range(dist.get_world_size())]
    dist.all_gather(schemas, schema)
    if any(not torch.equal(schema, other) for other in schemas):
        raise RuntimeError('metric schema differs across ranks; local keys={}'.format(keys))
    packed = torch.tensor([values[key] for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(packed)
    packed /= dist.get_world_size()
    return dict(zip(keys, packed.tolist()))
