"""KDA のチャンク内計算の Metal カーネル。M2 と M3 の 32KB のスレッドグループメモリに収める。

A_q[i, j] = sum_d q[i, d] exp(g[i, d] - g[j, d]) k[j, d]（i >= j）と、q を k に換えた A_k を返す。
減衰の表をメモリに作らず、スレッドの中で exp を取る。i >= j なので指数は 0 以下で、溢れない。
"""
import mlx.core as mx

THREADS = 256
THREADGROUP_BYTES = 32768

_FORWARD_SOURCE = """
    constexpr uint STRIDE = K + 1;
    threadgroup float tq[C * STRIDE];
    threadgroup float tk[C * STRIDE];
    threadgroup float tg[C * STRIDE];

    uint tid = thread_position_in_threadgroup.x;
    uint chunk = threadgroup_position_in_grid.x;
    uint base = chunk * C * K;

    for (uint p = tid; p < C * K; p += THREADS) {
        uint row = p / K;
        uint col = p % K;
        tq[row * STRIDE + col] = q[base + p];
        tk[row * STRIDE + col] = k[base + p];
        tg[row * STRIDE + col] = g[base + p];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint out_base = chunk * C * C;
    for (uint p = tid; p < C * C; p += THREADS) {
        uint i = p / C;
        uint j = p % C;
        float sum_q = 0.0f;
        float sum_k = 0.0f;
        if (j <= i) {
            for (uint d = 0; d < K; ++d) {
                float right = metal::exp(tg[i * STRIDE + d] - tg[j * STRIDE + d]) * tk[j * STRIDE + d];
                sum_q += tq[i * STRIDE + d] * right;
                sum_k += tk[i * STRIDE + d] * right;
            }
        }
        a_q[out_base + p] = sum_q;
        a_k[out_base + p] = sum_k;
    }
"""

_BACKWARD_SOURCE = """
    threadgroup float tq[C * K];
    threadgroup float tk[C * K];
    threadgroup float tg[C * K];

    uint tid = thread_position_in_threadgroup.x;
    uint chunk = threadgroup_position_in_grid.x;
    uint base = chunk * C * K;
    uint a_base = chunk * C * C;

    for (uint p = tid; p < C * K; p += THREADS) {
        tq[p] = q[base + p];
        tk[p] = k[base + p];
        tg[p] = g[base + p];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // d を連続させると、同じ simdgroup の d_a の読み出しが同じ番地になる
    for (uint p = tid; p < C * K; p += THREADS) {
        uint r = p / K;
        uint d = p % K;
        float q_r = tq[p];
        float k_r = tk[p];
        float g_r = tg[p];
        float grad_q = 0.0f;
        float grad_k = 0.0f;
        float grad_g = 0.0f;

        // r が左（i = r）の項
        for (uint j = 0; j <= r; ++j) {
            float right = metal::exp(g_r - tg[j * K + d]) * tk[j * K + d];
            float dq_ij = d_a_q[a_base + r * C + j];
            float dk_ij = d_a_k[a_base + r * C + j];
            grad_q += dq_ij * right;
            grad_k += dk_ij * right;
            grad_g += right * (dq_ij * q_r + dk_ij * k_r);
        }
        // r が右（j = r）の項
        for (uint i = r; i < C; ++i) {
            float decay = metal::exp(tg[i * K + d] - g_r);
            float mixed = d_a_q[a_base + i * C + r] * tq[i * K + d] + d_a_k[a_base + i * C + r] * tk[i * K + d];
            grad_k += mixed * decay;
            grad_g -= mixed * decay * k_r;
        }
        d_q[base + p] = grad_q;
        d_k[base + p] = grad_k;
        d_g[base + p] = grad_g;
    }
"""

_header = f"constant constexpr uint THREADS = {THREADS};\n"
_forward = mx.fast.metal_kernel(
    name="kda_intra_forward",
    input_names=["q", "k", "g"],
    output_names=["a_q", "a_k"],
    source=_FORWARD_SOURCE,
    header=_header,
)
_backward = mx.fast.metal_kernel(
    name="kda_intra_backward",
    input_names=["q", "k", "g", "d_a_q", "d_a_k"],
    output_names=["d_q", "d_k", "d_g"],
    source=_BACKWARD_SOURCE,
    header=_header,
)


def _launch(q):
    *lead, size, key_dim = q.shape
    chunks = 1
    for n in lead:
        chunks *= n
    return {
        "template": [("C", size), ("K", key_dim)],
        "grid": (chunks * THREADS, 1, 1),
        "threadgroup": (THREADS, 1, 1),
    }


@mx.custom_function
def intra_chunk(q, k, g):
    shape = (*q.shape[:-1], q.shape[-2])
    return tuple(_forward(
        inputs=[q, k, g],
        output_shapes=[shape, shape],
        output_dtypes=[mx.float32, mx.float32],
        **_launch(q),
    ))


@intra_chunk.vjp
def _intra_chunk_vjp(primals, cotangents, _):
    q, k, g = primals
    d_a_q, d_a_k = cotangents
    return tuple(_backward(
        inputs=[q, k, g, d_a_q, d_a_k],
        output_shapes=[q.shape] * 3,
        output_dtypes=[mx.float32] * 3,
        **_launch(q),
    ))


_SCAN_SOURCE = """
    constexpr uint ITEMS = (K * TILE + THREADS - 1) / THREADS;
    threadgroup float tm[K * K];
    threadgroup float ty[K * TILE];

    uint tid = thread_position_in_threadgroup.x;
    uint group = threadgroup_position_in_grid.x;
    uint tiles = V / TILE;
    uint sequence = group / tiles;
    uint column = (group % tiles) * TILE;

    for (uint p = tid; p < K * TILE; p += THREADS) {
        uint row = p / TILE;
        uint index = (sequence * N * K + row) * V + column + p % TILE;
        float value = x[index];
        ty[p] = value;
        out[index] = value;
    }

    for (uint i = 1; i < N; ++i) {
        uint m_base = (sequence * N + i - 1) * K * K;
        for (uint p = tid; p < K * K; p += THREADS) {
            tm[p] = m[m_base + p];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float acc[ITEMS];
        for (uint t = 0; t < ITEMS; ++t) {
            uint p = tid + t * THREADS;
            if (p < K * TILE) {
                uint row = p / TILE;
                uint col = p % TILE;
                float sum = x[((sequence * N + i) * K + row) * V + column + col];
                for (uint j = 0; j < K; ++j) {
                    sum += tm[row * K + j] * ty[j * TILE + col];
                }
                acc[t] = sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = 0; t < ITEMS; ++t) {
            uint p = tid + t * THREADS;
            if (p < K * TILE) {
                ty[p] = acc[t];
                out[((sequence * N + i) * K + p / TILE) * V + column + p % TILE] = acc[t];
            }
        }
    }
"""

SCAN_TILE = 16
_scan = mx.fast.metal_kernel(
    name="kda_linear_scan",
    input_names=["m", "x"],
    output_names=["out"],
    source=_SCAN_SOURCE,
    header=_header,
)


def _run_scan(carry, x):
    *lead, steps, key_dim, value_dim = x.shape
    sequences = 1
    for n in lead:
        sequences *= n
    groups = sequences * (value_dim // SCAN_TILE)
    return _scan(
        inputs=[carry, x],
        template=[("N", steps), ("K", key_dim), ("V", value_dim), ("TILE", SCAN_TILE)],
        grid=(groups * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[mx.float32],
    )[0]


@mx.custom_function
def linear_scan(carry, x):
    """y[0] = x[0]、y[i] = carry[i - 1] @ y[i - 1] + x[i] を、軸 -3 に沿って計算する。"""
    return _run_scan(carry, x)


@linear_scan.vjp
def _linear_scan_vjp(primals, cotangent, output):
    carry, _ = primals
    # 逆伝播も同じ形の漸化式で、向きが逆になり行列が転置になる
    transposed = carry[..., :-1, :, :].swapaxes(-1, -2)[..., ::-1, :, :]
    padded = mx.concatenate([transposed, carry[..., -1:, :, :]], axis=-3)
    total = _run_scan(padded, cotangent[..., ::-1, :, :])[..., ::-1, :, :]
    d_carry = total[..., 1:, :, :] @ output[..., :-1, :, :].swapaxes(-1, -2)
    d_carry = mx.concatenate([d_carry, mx.zeros_like(carry[..., -1:, :, :])], axis=-3)
    return d_carry, total
