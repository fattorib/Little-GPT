"""
Random utiliity functions

"""


def compute_num_steps(
    n_devices: int,
    batch_per_gpu: int,
    accum_steps: int,
    n_ctx: int,
    total_tokens: int,
) -> int:
    total_batch_size = n_devices * batch_per_gpu * accum_steps

    print(f"Effective Batch Size: {total_batch_size}")
    tokens_per_batch = total_batch_size * n_ctx
    print(f"Tokens per batch: {tokens_per_batch/1e6 :.2f}M")

    return int(total_tokens // tokens_per_batch)


def compute_num_tokens(
    n_devices: int,
    batch_per_gpu: int,
    accum_steps: int,
    n_ctx: int,
    num_gradient_updates: int,
) -> int:
    total_batch_size = n_devices * batch_per_gpu * accum_steps
    tokens_per_batch = total_batch_size * n_ctx

    return tokens_per_batch * num_gradient_updates
