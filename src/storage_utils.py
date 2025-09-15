import numpy as np

def pack_params_float32(params):
    return b"".join(np.asarray(p, dtype=np.float32).tobytes(order="C") for p in params)

def unpack_params_float32(blob, template):
    arr = np.frombuffer(blob, dtype=np.float32)
    out, offset = [], 0
    for t in template:
        size = int(np.prod(t.shape))
        vals = arr[offset:offset+size].reshape(t.shape)
        out.append(vals.copy())
        offset += size
    return out
