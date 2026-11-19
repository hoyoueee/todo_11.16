# vLLM 与 SGLang Hybrid KV Cache：存储逻辑与分层驱逐可能性调研

> 调研日期：2026-11-18  

> 代码块均来自所标注的本地源码路径；`# 【说明】` 是为便于阅读添加的解释，`...` 表示省略了与当前结论无关的参数或分支。

本文只关注两个问题：

1. vLLM 用于分配、匹配和索引的一个**逻辑 block**，是否覆盖多个 layer 的 cache，能否对部分层进行分层cache驱逐
2. SGLang 用于分配、匹配和索引的一个**逻辑 page/slot**，是否覆盖多个 layer 的 cache，能否对部分层进行分层cache驱逐

本文使用以下术语：

- **逻辑 block/page**：cache manager、block table、请求映射或前缀匹配所使用的编号对象。
- **layer payload**：某个逻辑 block/page 在某一层实际保存的 K/V、latent 或 recurrent state。
- **物理 layout**：这些 layer payload 在 backing allocation 中的排列方式。

---



## 2. vLLM：逻辑 block 是 cache group 的 block，不是单层 block

### 2.1 block 只按 cache group 分配一次

vLLM 为每个 `kv_cache_group` 创建一个 manager，而不是为 group 中的每个 layer 创建 manager。


group manager 申请 block 时也没有遍历 layer，只从 `BlockPool` 申请一次 `KVCacheBlock` 对象序列。

源码路径：`sources/vllm/vllm/v1/core/single_type_kv_cache_manager.py:505-515`

```python
req_blocks = self.req_to_blocks[request_id]
num_required_blocks = cdiv(num_tokens, self.block_size)
num_new_blocks = num_required_blocks - len(req_blocks)
if num_new_blocks <= 0:
    return cow_blocks
else:
    # 【说明】这里没有 for layer；得到的是 group-level blocks。
    new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
    req_blocks.extend(new_blocks)
    if self._record_new_block_ids:
        self.new_block_ids.extend(b.block_id for b in new_blocks)
    return cow_blocks + new_blocks
```


`BlockPool` 自身也只有一维 `blocks[block_id]`，没有 layer 维度。

源码路径：`sources/vllm/vllm/v1/core/block_pool.py:139-153`

```python
self.num_gpu_blocks = num_gpu_blocks
self.medium = medium
self.enable_caching = enable_caching
self.hash_block_size = hash_block_size
self.num_hash_lookahead_tokens = num_hash_lookahead_tokens
self._weak_self = weakref.ref(self)

# 【说明】唯一索引是 block_id，不存在 blocks[layer_id][block_id]。
self.blocks: list[KVCacheBlock] = [
    KVCacheBlock(idx, pool=self._weak_self) for idx in range(num_gpu_blocks)
]
self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
```


### 3.2 同一个 group block ID 被送给 group 中所有 layer

worker 端按 group 取得一份 `slot_mapping`，然后将同一个 tensor 对象赋给该 group 的所有 layer。

源码路径：`sources/vllm/vllm/v1/worker/gpu/attn_utils.py:453-461`

```python
def build_slot_mappings_by_layer(
    slot_mappings: torch.Tensor, kv_cache_config: KVCacheConfig
) -> dict[str, torch.Tensor]:
    slot_mappings_by_layer: dict[str, torch.Tensor] = {}
    kv_cache_groups = kv_cache_config.kv_cache_groups

    # 【说明】外层一项对应一个 cache group 的逻辑 block/slot 映射。
    for slot_mapping, kv_cache_group in zip(slot_mappings, kv_cache_groups):
        # 【说明】group 内所有 layer 复用同一份 mapping。
        for layer_name in kv_cache_group.layer_names:
            slot_mappings_by_layer[layer_name] = slot_mapping
    return slot_mappings_by_layer
```

正常 GPU model runner 确实执行这一步。

源码路径：`sources/vllm/vllm/v1/worker/gpu/model_runner.py:2040-2044`

```python
elif not (dummy_run and skip_attn_for_dummy_run):
    assert slot_mappings is not None
    # 【说明】将 group-level mapping 展开给各层 forward。
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, self.kv_cache_config
    )
```

同样，attention metadata 也是每个 group 只选择一份 block table 和 slot mapping。

源码路径：`sources/vllm/vllm/v1/worker/gpu/attn_utils.py:512-518, 540-550`

```python
num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
for i in range(num_kv_cache_groups):
    groups = groups_with_metadata_builders(attn_groups[i])
    if not groups:
        continue

    # 【说明】group i 只有一份逻辑 block table 与 slot mapping。
    block_table = block_tables[i]
    slot_mapping = slot_mappings[i]

    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc_gpu,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max_query_len,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        causal=group_causal,
    )
```

以上代码共同排除了“每层独立逻辑 block、编号碰巧相同”的解释：

```text
实际结构：

group-level KVCacheBlock(id=b)
    ├── layer A 使用 b 定位自己的 payload
    ├── layer B 使用 b 定位自己的 payload
    └── layer C 使用 b 定位自己的 payload
```

### 3.3 一个 vLLM 逻辑 block 的内容

对于一个 cache group，逻辑 block `b` 可以写成：

```text
logical_block(b)
= {
    cache(layer_0, b),
    cache(layer_1, b),
    ...,
    cache(layer_n, b)
  }
```

这不是只根据注释推导出的抽象。下面的实际执行链会把 group 内多个 layer 的 `cache(layer, b)` 都落到同一个逻辑 block 编号 `b` 下。



#### 代码证据 1：同一 backing allocation 被切成每层 view，且每个 view 都保留 block 维

源码路径：`sources/vllm/vllm/v1/kv_cache_interface.py:358-423`

```python
def create_kv_cache_views(
    raw: torch.Tensor,
    spec: KVCacheSpec,
    num_blocks: int,
    layout: KVCacheLayout,
    kv_cache_tensor: KVCacheTensor,
    kernel_block_size: int | None = None,
) -> list[torch.Tensor]:
    num_layers = len(kv_cache_tensor.layers)
    layer_stride = kv_cache_tensor.layer_stride
    block_stride = kv_cache_tensor.block_stride
    shape_bytes = compute_layer_kv_cache_shape_bytes(
        spec, num_blocks, kernel_block_size
    )

    # 【说明】这里的 logical_shape 是 tensor 的规范坐标轴形状，
    # 不是“一个 manager 逻辑 block 的内部 shape”。它同时保留 L、B 两个轴。
    logical_shape = (num_layers, *shape_bytes)
    strides = compute_layout_strides(
        spec,
        num_blocks,
        num_layers,
        layout,
        kernel_block_size,
        fixed_strides=(layer_stride, block_stride, None, None, None),
    )
    dtype = getattr(spec, "dtype", None)

    view_5d = torch.as_strided(
        raw,
        size=logical_shape,
        stride=strides,
        storage_offset=raw.storage_offset() + kv_cache_tensor.offset,
    )

    views = []
    for layer_idx in range(num_layers):
        # 【说明】每个 layer 得到 view_5d[layer_idx]；该 view 的第一维
        # 仍然是 block 维，因此 view_5d[layer_idx][b] 就是 cache(layer, b)。
        cache_logical = view_5d[layer_idx]
        if dtype is not None:
            cache_logical = cache_logical.view(dtype)
        views.append(cache_logical)
    return views
```

这些 view 随后按 `tensor.layers` 的同一顺序绑定到对应层。

源码路径：`sources/vllm/vllm/v1/worker/utils.py:623-639, 952-962`

```python
kv_caches: dict[str, torch.Tensor] = {}
for tensor in kv_cache_config.kv_cache_tensors:
    group_id, spec = layer_spec_for_tensor(kv_cache_config, tensor)
    if not spec.has_layer_views:
        kv_caches.update((name, buf) for name in tensor.layers)
        continue

    views = create_kv_cache_views(
        buf,
        spec,
        kv_cache_config.num_blocks_of(tensor),
        layout,
        tensor,
        kernel_block_size=layer_kernel_block_size(group_id, kernel_block_sizes),
    )
    # 【说明】tensor.layers[i] 对应 views[i]，即每层取得自己的 [B, ...] view。
    kv_caches.update(zip(tensor.layers, views))


def bind_kv_cache_to_layers(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
) -> None:
    for layer_name, kv_cache in kv_caches.items():
        # 【说明】把该层的 [B, ...] view 绑定到该层实例。
        forward_context[layer_name].bind_kv_cache(kv_cache)
```

#### 代码证据 2：group 内所有层取得完全相同的 slot 编号

源码路径：`sources/vllm/vllm/v1/worker/gpu/attn_utils.py:453-461`

```python
def build_slot_mappings_by_layer(
    slot_mappings: torch.Tensor, kv_cache_config: KVCacheConfig
) -> dict[str, torch.Tensor]:
    slot_mappings_by_layer: dict[str, torch.Tensor] = {}
    kv_cache_groups = kv_cache_config.kv_cache_groups

    for slot_mapping, kv_cache_group in zip(slot_mappings, kv_cache_groups):
        for layer_name in kv_cache_group.layer_names:
            # 【说明】这里没有为各 layer 重新计算 block/page ID；
            # group 内每个 layer 都引用同一个 slot_mapping tensor。
            slot_mappings_by_layer[layer_name] = slot_mapping
    return slot_mappings_by_layer
```

其中一个 token 的 slot 由 group 的 block table 计算为：

源码路径：`sources/vllm/vllm/v1/worker/gpu/block_table.py:415-427`

```python
block_indices = tl.where(mapping_enabled, local_positions // kv_block_size, 0)
block_offsets = local_positions % kv_block_size
block_numbers = tl.load(
    block_table_ptr + req_state_idx * block_table_stride + block_indices,
    mask=is_local & is_real_req,
    other=0,
)

# 【说明】slot_id = 逻辑/物理 block 编号 * block_size + block 内偏移。
# 这份 slot_ids 随后被复用给 group 内所有 layer。
slot_ids = block_numbers * kv_block_size + block_offsets
tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)
```

#### 代码证据 3：每层 forward 用相同 slot，写入自己的 cache view

源码路径：`sources/vllm/vllm/model_executor/layers/attention/attention.py:752-793`

```python
# 【说明】这是当前 layer 在代码证据 1 中绑定的 [B, ...] cache view。
kv_cache = attn_layer.kv_cache

slot_mapping = forward_context.slot_mapping
assert isinstance(slot_mapping, dict)

# 【说明】按 layer_name 取 mapping；但代码证据 2 表明，同组各层取到的是
# 同一个 slot_mapping tensor，因此相同 token 会得到相同的 block b 和 offset。
layer_slot_mapping = slot_mapping.get(layer_name)

attn_layer.impl.do_kv_cache_update(
    attn_layer,
    key,
    value,
    kv_cache,
    layer_slot_mapping,
)
```

以 FlashAttention backend 为例，真正的写入调用如下。

源码路径：`sources/vllm/vllm/v1/attention/backends/flash_attn.py:1406-1440`

```python
def do_kv_cache_update(
    self,
    layer: torch.nn.Module,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    # 【说明】kv_cache 是当前 layer 的 [B, ...] view。
    key_cache, value_cache = kv_cache.transpose(1, 2).split(
        self.head_size, dim=-1
    )

    reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        # 【说明】同组所有 layer 使用相同 slot_mapping，所以它们分别写入
        # cache(layer_0, b)、cache(layer_1, b)……的相同 block/offset 位置。
        slot_mapping,
        self.kv_cache_dtype,
        layer._k_scale,
        layer._v_scale,
    )
```

因此，源码中的实际过程可以严格对应为：

```text
同一个 group slot/block b
    + layer_0 自己的 [B, ...] view -> 写 cache(layer_0, b)
    + layer_1 自己的 [B, ...] view -> 写 cache(layer_1, b)
    + ...
    + layer_n 自己的 [B, ...] view -> 写 cache(layer_n, b)

= logical_block(b)
```

这里的“一个逻辑 block 包含多层 cache”表示这些 layer payload 共享同一个 group-level block 身份和生命周期；并不要求存在一个 Python `list`/`dict` 在一次函数调用中把所有层打包写入。模型 forward 本来就是逐层执行，所以实际写入也是各层分别发生。

如果 group 中是 attention layers，每个 `cache(layer, b)` 通常是该层一个 token block 的 K/V；如果是 recurrent/state layer，则是对应 state payload。

vLLM 没有创建一个 Python 容器把这些 layer payload装进去。这个多层逻辑对象由两件事隐式定义：

1. manager 只分配一个 group-level block ID `b`；
2. group 内所有 layer 都用 `b` 索引自己的 cache view。

### 3.4 物理显存排布：block-outer 与 layer-outer 只改变物理展开顺序

本节中的 `block-outer` 和 `layer-outer` 是 **physical KV cache layout**，不是两种不同的逻辑 block 管理方式。

为避免混淆，下面明确区分：

- **逻辑 block `b`**：manager、block table、prefix match 使用的 group-level 编号。
- **物理 page `(layer, b)`**：某一层在逻辑 block `b` 下实际占用的 K/V 字节区域。
- **物理 block 区域**：仅在 block-outer 排布下，用于描述同一 `b` 的多层物理 pages 在 backing memory 中形成的连续 byte range；它不是额外创建的 manager/block-table 对象。

可以把 group cache 先理解成一个逻辑二维矩阵：

```text
                       逻辑 block 0    逻辑 block 1    逻辑 block 2
layer A 的物理 page        A0              A1              A2
layer B 的物理 page        B0              B1              B2
layer C 的物理 page        C0              C1              C2
```

无论使用哪一种物理 layout，逻辑 block 始终是一列：

```text
logical block 0 = {物理 page A0, 物理 page B0, 物理 page C0}
logical block 1 = {物理 page A1, 物理 page B1, 物理 page C1}
logical block 2 = {物理 page A2, 物理 page B2, 物理 page C2}
```

#### 每层 view 的含义

vLLM 最终会为每一层提供一个按 block ID 索引的 cache view。执行到 layer A 时，两种 layout 下访问方式都是：

```python
layer_A_view[p]
```

它表示“layer A 在 group block ID `p` 下的物理 cache page”。两种模式的 view shape 和索引语义相同，通常都可理解为 `[B, H, N, C]`；不同的是底层 `storage_offset` 和 B 轴 stride：

| per-layer view | 物理 block-outer | 物理 layer-outer |
|---|---|---|
| 访问形式 | `layer_A_view[p]` | `layer_A_view[p]` |
| B 轴相邻项 | A0、A1 之间跨过其他层的 pages | A0、A1 在 layer A 物理区内相邻 |
| B 轴 stride | 约为一个多层 packed 物理 block 区域 | 约为一个物理 page |
| 同一 `p` 的多层 pages | 物理相邻 | 物理分散 |

因此，view 的“内容含义”没有改变；layout 改变的是 `layer_A_view[p]` 最终通过 stride 定位到 backing memory 的哪个物理地址。

#### 物理 block-outer：同一逻辑 block 的多层物理 pages 相邻

```text
backing memory：

[A0 | B0 | C0] [A1 | B1 | C1] [A2 | B2 | C2]
└物理 block 区域 0┘ └物理 block 区域 1┘ └物理 block 区域 2┘
```

这里的“物理 block 区域 0”表示同一逻辑 block ID `0` 的多层 payload 恰好连续存放。它对应 `{A0, B0, C0}` 的连续显存范围，但逻辑所有权仍由 cache group 的 block `0` 决定。

#### 物理 layer-outer：每层自己的多个物理 pages 相邻

```text
backing memory：

[A0 | A1 | A2] [B0 | B1 | B2] [C0 | C1 | C2]
└ layer A 物理区 ┘ └ layer B 物理区 ┘ └ layer C 物理区 ┘
```

此时不存在一个跨层连续的“物理 block 区域 0”。逻辑 block `0` 仍然是 `{A0, B0, C0}`，只是这三个物理 pages 分散在三个 layer 物理区中。

因此：

```text
物理 block-outer：B 是外层物理轴，同一 b 的多层物理 pages 相邻。
物理 layer-outer：L 是外层物理轴，同一 layer 的多个物理 pages 相邻。

二者都不改变逻辑 block 的编号、匹配粒度、覆盖层集合和驱逐边界。
```

#### 源码解释：layout 如何真正变成物理显存地址

第一步，layout enum 明确声明自己描述的是 physical order。源码所说的 “logical shape `[L, B, H, N, C]`” 是 **tensor 的规范坐标轴顺序**，不是“一个 manager 逻辑 block 按 L→B 组织”。

这两个概念应写成：

```text
tensor 规范坐标：view_5d[layer_id, block_id, ...]，轴名固定为 [L, B, H, N, C]

manager 逻辑 block b：
    {view_5d[0, b, ...], view_5d[1, b, ...], ..., view_5d[L-1, b, ...]}
    等价记法：view_5d[:, b, ...]
```

因此 `[L, B, ...]` 中 L 写在 B 前面，只是在声明索引参数的顺序；它既不表示 L 是 manager 的上层对象，也不表示每层拥有独立逻辑 block。manager 如何把固定 `b` 解释为跨层逻辑 block，来自前文已经证明的 group-level block ID 和共享 slot mapping。

源码路径：`sources/vllm/vllm/v1/kv_cache_layout.py:15-32, 56-58`

```python
class KVCacheLayout(Enum):
    """Physical layout descriptor for a KV cache group.

    The logical shape is always [L, B, H, N, <content>].
    Each member's value is a stride permutation that maps logical axes
    to physical (memory) order.
    """

    # 【说明】上游 docstring 中的 logical shape 指 tensor 坐标轴，
    # 不指 cache manager 的 logical block ownership。

    # 【说明】L 在物理外层：一个 layer 的多个 B 位置先连续排列。
    LBNHC = (0, 1, 3, 2, 4)  # [L, B, N, H, C]

    # 【说明】B 在物理外层：一个 block ID 的多个 layer pages 先排列。
    BLNHC = (1, 0, 3, 2, 4)  # [B, L, N, H, C]

    @property
    def stride_order(self) -> tuple[int, ...]:
        return self.value

    @property
    def is_block_outermost(self) -> bool:
        return self.value[0] == _DIM_B
```

第二步，`compute_layout_strides()` 按物理轴顺序从内向外计算每个 **tensor 坐标轴** 的 byte stride。它始终保留相同的 `L` 和 `B` 两个索引轴，只改变 `(l, b)` 映射到哪个物理地址，因此得到不同的 `layer_stride` 与 `block_stride`。

源码路径：`sources/vllm/vllm/v1/kv_cache_interface.py:317-355`

```python
def compute_layout_strides(
    spec: KVCacheSpec,
    num_blocks: int,
    num_layers: int,
    layout: KVCacheLayout,
    kernel_block_size: int | None = None,
    fixed_strides: tuple[int | None, ...] = (None,) * 5,
) -> tuple[int, ...]:
    # 【说明】这是 tensor 的规范坐标 shape [L, B, H, N, C]；
    # manager 逻辑 block b 对应固定 B=b、遍历所有 L 的切片。
    shape = (
        num_layers,
        *compute_layer_kv_cache_shape_bytes(spec, num_blocks, kernel_block_size),
    )

    # 【说明】只有物理遍历顺序由 LBNHC / BLNHC 等 layout 决定。
    order = layout.stride_order

    strides = [0] * 5
    current_stride = 1
    for physical_idx, dim in reversed(tuple(enumerate(order))):
        if padded_page_size is not None and physical_idx == page_grid_end - 1:
            current_stride = max(current_stride, padded_page_size)
            assert current_stride % padded_page_size == 0
        strides[dim] = fixed_strides[dim] or current_stride
        current_stride = strides[dim] * shape[dim]
    return tuple(strides)
```

忽略 page 内部的 `H/N/C` 细节后，上述循环得到的关键关系是：

```text
物理 block-outer / BLNHC：
    layer_stride ≈ one_page_bytes
    block_stride ≈ num_layers × one_page_bytes
    address(layer=l, block=b) ≈ base + b×block_stride + l×layer_stride

物理 layer-outer / LBNHC：
    block_stride ≈ one_page_bytes
    layer_stride ≈ num_blocks × one_page_bytes
    address(layer=l, block=b) ≈ base + l×layer_stride + b×block_stride
```

第三步，实际分配代码把这两个 stride 保存进 `KVCacheTensor`，再由 `torch.as_strided()` 在同一个 raw backing allocation 上建立可按 `(layer_id, block_id)` 访问的五维 tensor view。

源码路径：`sources/vllm/vllm/v1/core/kv_cache_planning.py:650-664`；`sources/vllm/vllm/v1/kv_cache_interface.py:399-423`

```python
layer_stride, block_stride, _, _, _ = compute_layout_strides(
    spec,
    num_blocks,
    len(layer_names),
    layout,
    fixed_strides=(None, block_stride, None, None, None),
)
kv_cache_tensors.append(
    KVCacheTensor(
        size=size,
        layers=layer_names,
        layer_stride=layer_stride,
        block_stride=block_stride,
        offset=byte_offset,
    )
)

# 【说明】这里固定的是 tensor 坐标轴 [L, B, ...]，不是逻辑 block 的层次结构。
# 对 manager 逻辑 block b，应观察 view_5d[:, b, ...] 这一跨层切片。
logical_shape = (num_layers, *shape_bytes)
strides = compute_layout_strides(
    spec,
    num_blocks,
    num_layers,
    layout,
    kernel_block_size,
    fixed_strides=(layer_stride, block_stride, None, None, None),
)
view_5d = torch.as_strided(
    raw,
    size=logical_shape,
    stride=strides,
    storage_offset=raw.storage_offset() + kv_cache_tensor.offset,
)

views = []
for layer_idx in range(num_layers):
    # 【说明】每层取得一个第一维为 B 的 cache view；不同层对同一个 b
    # 的元素合在一起，才构成 group-level logical block b。
    cache_logical = view_5d[layer_idx]
    views.append(cache_logical)
```

这段执行代码把结论落实到了实际地址计算上：layout 改的是 `layer_stride` / `block_stride`，也就是物理 page 之间的距离。`view_5d` 只是统一的 tensor 坐标系；真正的 group-level 逻辑 block `b` 是其中固定 B、跨 L 的集合，且该 `b` 仍由共享的 group block ID / slot mapping 决定。

#### 默认选择与使用倾向

在当前 vLLM 源码中，**普通场景默认倾向使用物理 layer-outer**。当 backend 没有声明特殊偏好时，通用布局优先级首先是 `LBNHC`，其次是 `LBHNC`，两者都属于 layer-outer；`BLNHC`、`BLHNC` 等 block-outer 排在其后。

但这不是一个对所有模型和 backend 都固定不变的默认值。vLLM 启动时会汇总各 attention backend 支持且偏好的 layout，应用 Hybrid、DCP、speculative decoding 和 KV connector 等约束，最后选择共同候选中的第一项。因此：

- 一般路径、无特殊约束：通常选择 `LBNHC` 等 layer-outer；
- 特殊 backend 或混合 cache 约束：可能选择甚至强制使用 block-outer；
- 实际运行结果：以启动日志 `Using <layout> KV cache layout.` 为准；也可通过 `VLLM_KV_CACHE_LAYOUT` 显式指定，但指定值必须受当前 backend 支持。

对应源码：`sources/vllm/vllm/v1/attention/backends/utils.py` 中的 `_DEFAULT_LAYOUT_PREFERENCE` 与 `resolve_kv_cache_layout()`，以及 `sources/vllm/vllm/envs.py` 中默认值为 `None` 的 `VLLM_KV_CACHE_LAYOUT`。


### 3.6 Hybrid 对 vLLM 逻辑 block 的影响

Hybrid 只影响“哪些层属于同一个 cache group”。一个逻辑 block 覆盖的是**当前 group 中的层**，不是必然覆盖模型所有层。

```text
group 0 logical block b
= group 0 中所有 layer 的 b 号 payload

group 1 logical block b
= group 1 中所有 layer 的 b 号 payload
```

层类型会改变 payload 内容：

- attention layer：K/V 或量化后的 K/V 表示；
- MLA：latent/packed cache；
- Mamba/GDN/KDA：recurrent state。

但 `block-outer` / `layer-outer` 不会改变 group-level logical block 的含义。

---

## 4. SGLang：逻辑 page/slot ID 同样由多层共同使用

### 4.1 cache location 只分配一次，不包含 layer 维

decode 时 SGLang 只产生一份 `out_cache_loc`，并写入二维 `req_to_token` 表。该表的维度是 request × token position，没有 layer 维。

源码路径：`sources/sglang/python/sglang/srt/mem_cache/allocation.py:691-718`；`sources/sglang/python/sglang/srt/mem_cache/memory_pool.py:414-429`

```python
if _alloc_page_size(batch) == 1:
    out_cache_loc = alloc_token_slots(batch.tree_cache, bs * token_per_req)
else:
    last_loc = batch.req_to_token_pool.req_to_token[
        batch.req_pool_indices, seq_lens_gpu - 1
    ]
    seq_lens_next = seq_lens_gpu + token_per_req
    # 【说明】只分配一份逻辑 cache location/page location。
    out_cache_loc = alloc_paged_token_slots_decode(
        tree_cache=batch.tree_cache,
        seq_lens=seq_lens_next,
        seq_lens_cpu=batch.seq_lens_cpu + token_per_req,
        last_loc=last_loc,
        token_per_req=token_per_req,
        req_pool_indices=batch.req_pool_indices,
        batch=batch,
    )

# 【说明】同一 location 被记录到 request-token 映射中，不按 layer 复制表。
batch.req_to_token_pool.write(
    (batch.req_pool_indices, locs), out_cache_loc.to(torch.int32)
)
```

```python
self.size = size
self._alloc_size = size + 1
self.max_context_len = max_context_len
self.device = device

# 【说明】shape 中没有 layer 轴。
self.req_to_token = torch.zeros(
    (self._alloc_size, max_context_len), dtype=torch.int32, device=device
)

def write(self, indices, values):
    self.req_to_token[indices] = values
```

### 4.2 默认模式下，同一 page ID 分别索引各层 buffer

默认 MHA pool 的物理 buffer 是 per-layer 的。

源码路径：`sources/sglang/python/sglang/srt/mem_cache/memory_pool.py:2662-2673, 2746-2754`

```python
def _kv_buffer_shapes(self):
    if self.use_hnd:
        return (
            (self.num_pages, self.head_num, self.page_size, self.head_dim),
            (self.num_pages, self.head_num, self.page_size, self.v_head_dim),
        )
    rows = self.size + self.page_size
    return (
        (rows, self.head_num, self.head_dim),
        (rows, self.head_num, self.v_head_dim),
    )

# 【说明】每层拥有独立物理 K/V tensor。
self.k_buffer = [_zeros(k_shape) for _ in range(self.layer_num)]
self.v_buffer = [_zeros(v_shape) for _ in range(self.layer_num)]
```

但写入时，代码先按 `layer_id` 选择物理 buffer，再用所有层共同使用的 `loc` 选择 page 和 page 内 offset。

源码路径：`sources/sglang/python/sglang/srt/mem_cache/memory_pool.py:3093-3116`

```python
if self.use_hnd:
    # 【说明】先选择当前层的独立物理 buffer。
    k_buf = self.k_buffer[layer_id - self.start_layer]
    v_buf = self.v_buffer[layer_id - self.start_layer]

    # 【说明】loc/page ID 本身没有 layer 维，各层使用同一个 loc。
    pages = loc // self.page_size
    offs = loc % self.page_size

    # 【说明】相同 pages/offs 分别写入当前层 buffer。
    k_buf[pages, :, offs, :] = cache_k
    v_buf[pages, :, offs, :] = cache_v
    return

self._store_kv_layer(layer_id - self.start_layer, loc, cache_k, cache_v)
```

因此，默认模式下一个逻辑 page `p` 应理解为：

```text
logical page p
= {
    k_buffer[layer 0][p], v_buffer[layer 0][p],
    k_buffer[layer 1][p], v_buffer[layer 1][p],
    ...
  }
```

这些 payload 物理上分散，但 page ID 是 pool-level 的，不是每层独立分配的 page ID。

---

## 5. SGLang unified memory：逻辑 page 不变，只改变填充方式

### 5.1 unified entry 直接包含多层 payload

unified MHA layout 将同一 slot 的所有 layer rows 放进一个 entry。

源码路径：`sources/sglang/python/sglang/srt/mem_cache/kv/layout/mha.py:99-162`

```python
def k_row_bytes(self) -> int:
    return self.head_num * self.head_dim * self.store_dtype.itemsize

def v_row_bytes(self) -> int:
    return self.head_num * self.v_head_dim * self.store_dtype.itemsize

def host_entry_bytes(self) -> int:
    # 【说明】一个 slot entry 的大小直接乘以 layer_num。
    return self.layer_num * (self.k_row_bytes() + self.v_row_bytes())

def page_bytes(self, page_size: int) -> int:
    # 【说明】一个逻辑 page 是 page_size 个 multi-layer entries。
    return page_size * self.entry_bytes()

def _target_parts(self):
    layer_stride = self.k_row_bytes() + self.v_row_bytes()
    parts = (
        DensePart(
            name="k",
            offset_bytes=0,
            layer_stride_bytes=layer_stride,
            layer_num=self.layer_num,
            row_shape=(self.head_num, self.head_dim),
            dtype=self.store_dtype,
        ),
        DensePart(
            name="v",
            offset_bytes=self.k_row_bytes(),
            layer_stride_bytes=layer_stride,
            layer_num=self.layer_num,
            row_shape=(self.head_num, self.v_head_dim),
            dtype=self.store_dtype,
        ),
    )
    return parts
```

物理填充形式为：

```text
entry 0 = [K0 | V0 | K1 | V1 | K2 | V2 | ...]
entry 1 = [K0 | V0 | K1 | V1 | K2 | V2 | ...]
...

logical page p = page_size 个这样的 entries
```


### 5.2 default 与 unified 的逻辑 page 没有本质区别

假设 pool 中有 A、B、C 三层：

```text
logical page 0 = {A0, B0, C0}
logical page 1 = {A1, B1, C1}
```

默认模式的填充方式：

```text
A buffer: A0, A1, A2, ...
B buffer: B0, B1, B2, ...
C buffer: C0, C1, C2, ...
```

unified 的填充方式：

```text
entry/page 0: A0, B0, C0
entry/page 1: A1, B1, C1
entry/page 2: A2, B2, C2
```

因此：

```text
default：相同逻辑 page 的多层 payload 物理分散。
unified：相同逻辑 page 的多层 payload 物理打包。

逻辑 page ID 的多层含义不变。
```

---

## 6. Hybrid 模型中 logical block/page 覆盖哪些层

Hybrid 不会改变 logical block/page 的基本定义，只会改变它覆盖的 layer 集合。

### vLLM

一个 logical block 覆盖当前 `kv_cache_group` 中的层：

```text
group G 的 logical block b
= G 中所有层在 b 位置的 cache payload
```

如果模型被划分为多个 cache groups，则每个 group 有自己的 block table/manager 视角。`block-outer` 与 `layer-outer` 不改变 group 边界，也不改变 block 的逻辑匹配含义。

### SGLang

默认模式按实际 pool 管理 layer cache；unified memory 按 sub-pool/band 组织，例如 full attention、某个 window 长度或 recurrent state。

```text
sub-pool P 的 logical page p
= P 中所有层在 p 位置的 cache payload
```

不同 sub-pool 的 payload 不需要混进同一个 page envelope。Hybrid 主要决定 layer 被路由到哪个 pool/sub-pool。

---

## 7. 驱逐粒度：能否只驱逐 logical block/page 中的部分层

### 7.1 先给结论

| 框架与模式 | 当前驱逐单位 | 能否只驱逐其中部分层 |
|---|---|---|
| vLLM | 一个 cache-group-level `KVCacheBlock` | **不能在 block 内选层**。这个 block 覆盖 group 中多少层，就一起逻辑失效多少层 |
| SGLang 默认 `RadixCache` | 一组统一的 loc/page IDs | **不能在同一 pool page 内选层**。释放 ID 后，该 ID 在 pool 各层上的 payload 一起失去逻辑所有权 |
| SGLang unified cache | 一个 tree component 对应的 sub-pool unit/page | **可以按 component/sub-pool 分开驱逐**，因此 Full、不同 SWA window、Mamba state 可以分别回收；但**不能在同一个 sub-pool 内任意选择单层** |

两种“部分层”：

```text
支持：驱逐一个独立 group/component 所包含的整组层，保留其他 sub-pool。
不支持：在同一个 logical block/page/sub-pool unit 内，只驱逐 layer A，保留 layer B。
```

### 7.2 驱逐边界覆盖的层集合本身也可能不连续

部分层驱逐还有一个容易忽略的问题：vLLM 的 group 和 SGLang 的 sub-pool 都不等价于“模型中一段连续层”。因此，当前按 group/sub-pool 驱逐时，一次失效的可能是一组分散在模型深度不同位置的层。

仍以 `3 SWA + 1 Full` 重复两次为例：

```text
模型层顺序：
layer id: 0   1   2   3  | 4   5   6   7
type:     S0  S1  S2  F0 | S3  S4  S5  F1
```

#### vLLM：按 pattern 位置拆成多个 groups

vLLM 先形成同类型 layer bucket，再使用 `layers[i::num_groups]` 步进取层，而不是按连续模型层区间切分。

源码路径：`sources/vllm/vllm/v1/core/kv_cache_planning.py:466-520`

```python
layer_buckets = [list(bucket) for bucket in _bucket_layers(layer_configs)]
...
grouped_layers = []
for layers in layer_buckets:
    num_groups = cdiv(len(layers), group_size)
    for i in range(num_groups):
        # 【说明】跨 pattern repeat 取同一位置的层，不是连续切片。
        grouped_layers.append(layers[i::num_groups])

return [
    KVCacheGroup.from_layers({name: layer_configs[name] for name in names})
    for names in grouped_layers
]
```

这个例子通常形成：

```text
SWA group A = [0, 4]
SWA group B = [1, 5]
SWA group C = [2, 6]
Full group  = [3, 7]
```

具体 group ID 顺序由实际 bucket/config 决定，不能仅凭类型假定编号。驱逐 group A 的某个 logical block 时，一起失效的是 layer 0 和 layer 4 在该 block 上的 payload，而不是模型中的连续层 0～4。

不同 groups 的 cache hash 包含各自的 `group_id`，所以它们的 block 可以独立被驱逐：

源码路径：`sources/vllm/vllm/v1/core/kv_cache_utils.py:58-67`

```python
def make_block_hash_with_group_id(
    block_hash: BlockHash, group_id: int
) -> BlockHashWithGroupId:
    # 【说明】相同 token hash 在不同 group 下是不同 cache key。
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big", signed=False))
```

但是相同 manager policy 的多个 groups 在命中时会联合查询，任一 group 缺失就返回 miss：

源码路径：`sources/vllm/vllm/v1/core/block_pool.py:175-201`

```python
def get_cached_block(
    self, block_hash: BlockHash, kv_cache_group_ids: list[int]
) -> list[KVCacheBlock] | None:
    cached_blocks = []
    for group_id in kv_cache_group_ids:
        block_hash_with_group_id = make_block_hash_with_group_id(
            block_hash, group_id
        )
        block = self.cached_block_hash_to_block.get_one_block(
            block_hash_with_group_id
        )
        # 【说明】只要其中一个 group 的 block 已驱逐，联合命中失败。
        if not block:
            return None
        cached_blocks.append(block)
    return cached_blocks
```

因此 vLLM 可以只驱逐某个 group 的 block，其他 group 的 block 仍可能保留；但缺少任一相关 group 后，保留下来的 blocks 通常不能单独形成完整 prefix hit。要获得真正的任意 layer 级驱逐，仍需把 cache key、validity 和命中判断继续下沉到 layer/subgroup 维度。

#### SGLang：按 lifetime/window 汇总为较大的 sub-pool

SGLang unified planner 按 `(lifetime, allocation unit)` 建组，并把具有相同 geometry 的实际 `layer_id` 追加进去，同样没有要求 layer ID 连续。

源码路径：`sources/sglang/python/sglang/srt/mem_cache/kv/plan.py:432-453`

```python
_Groups = Dict[Tuple[KVLifetime, int], List[Tuple[PartGeometry, List[int]]]]

def _group_parts(decls: Sequence[KVLayerDecl], page_size: int) -> _Groups:
    groups: _Groups = {}
    for decl in decls:
        for part in decl.parts:
            key = (part.lifetime, _unit(part.lifetime, page_size))
            entries = groups.setdefault(key, [])
            for geometry, layer_ids in entries:
                if geometry == part.geometry:
                    # 【说明】直接追加实际 model layer_id，没有连续性检查。
                    layer_ids.append(decl.layer_id)
                    break
            else:
                entries.append((part.geometry, [decl.layer_id]))
    ...
    return groups
```

收集结果直接写入 `PlannedSubPool.layer_ids`：

源码路径：`sources/sglang/python/sglang/srt/mem_cache/kv/plan.py:713-752`

```python
sub_pools.append(
    PlannedSubPool(
        name=name,
        lifetime=lifetime,
        unit=unit,
        # 【说明】这个 sub-pool unit/page 共同覆盖的实际层集合。
        layer_ids=tuple(layer_ids),
        geometry=geometry,
        spec=spec,
        parts=parts,
        ...
    )
)
```

如果所有 SWA 层具有相同 window 和 geometry，上述例子通常是：

```text
SWA/window sub-pool = [0, 1, 2, 4, 5, 6]
Full sub-pool       = [3, 7]
```

这与 vLLM 的 pattern-position groups 不同：SGLang 通常把相同 lifetime/window 的层汇总进一个更大的 sub-pool。因此驱逐一个 SWA logical page/unit，会一起失效 layer 0、1、2、4、5、6 对应的 payload；它们同样不是一个连续模型层区间。

```text
vLLM 最细现有边界：一个 pattern-position cache group 的 block。
SGLang unified 最细现有边界：一个 lifetime/window sub-pool 的 page/unit。

二者都不能在这些边界内部再任选某个 layer 驱逐。
```

物理上是否连续不会改变上述结论：block-outer/unified entry 可以把非连续模型层的 payload 相邻打包；layer-outer/default per-layer buffer 可以让它们物理分散。驱逐能力取决于 metadata、ID 和 ownership 的边界，不取决于能否定位某一层的内存 slice。

---

## 8. 最终对比

| 问题 | vLLM | SGLang |
|---|---|---|
| 逻辑匹配编号属于谁 | cache group | pool/sub-pool |
| 是否为每层独立分配逻辑编号 | 否 | 否；同一 pool 内各层共同使用 loc/page ID |
| 一个逻辑单位是否覆盖多层 payload | 是，覆盖 group 中的层 | 是，覆盖 pool/sub-pool 中的层 |
| layer-specific 的是什么 | 物理 cache view/payload | 物理 buffer 或 unified entry offset |
| layout 改变什么 | `[Block, Layer]` 与 `[Layer, Block]` 的展开顺序 | per-layer buffer 与 multi-layer entry 的填充方式 |
| layout 是否改变逻辑匹配单位 | 否 | 否 |
| 最细的现有驱逐边界 | cache group 的 logical block | 默认模式为 pool page；unified 为 component/sub-pool unit |
| 能否在一个逻辑单位内部任选单层驱逐 | 否 | 否；unified 只能先按独立 sub-pool/component 拆开 |

一句话总结：

> vLLM 的 block-outer/layer-outer，以及 SGLang 的 default/unified，核心差异都是多层 cache payload 的物理填充方式；用于匹配和索引的 logical block/page 仍然是 group 或 pool 级对象，而不是 per-layer 独立对象。

---

## 9. 官方资料与源码索引

### 官方说明

- [vLLM Hybrid KV Cache Manager](https://github.com/vllm-project/vllm/blob/902364d152ff5cc204addd049c217e3bd68cc2fe/docs/design/hybrid_kv_cache_manager.md)
- [SGLang Unified Memory for Hybrid Models](https://github.com/sgl-project/sglang/blob/47361c57be9fcc23f11b9804633bfcfe9342f98e/docs/docs/advanced_features/unified_memory.mdx)

### vLLM 关键源码

- `sources/vllm/vllm/v1/core/kv_cache_coordinator.py`：每个 cache group 创建一个 manager。
- `sources/vllm/vllm/v1/core/single_type_kv_cache_manager.py`：group manager 分配 logical blocks。
- `sources/vllm/vllm/v1/core/block_pool.py`：唯一的一维 `KVCacheBlock` 对象池。
- `sources/vllm/vllm/v1/core/kv_cache_manager.py`：按 block ID 驱逐的公开入口。
- `sources/vllm/vllm/v1/worker/gpu/attn_utils.py`：同一 group mapping 展开给所有 layer。
- `sources/vllm/vllm/v1/kv_cache_layout.py`：block-outer/layer-outer 的 L/B 展开顺序。
- `sources/vllm/vllm/v1/kv_cache_interface.py`：共享 backing 上的 layer/block views。

### SGLang 关键源码

- `sources/sglang/python/sglang/srt/mem_cache/allocation.py`：统一 cache location/page location 的分配。
- `sources/sglang/python/sglang/srt/mem_cache/memory_pool.py`：默认 per-layer buffers 使用相同 loc/page ID。
- `sources/sglang/python/sglang/srt/mem_cache/kv/layout/mha.py`：unified multi-layer entry/page 定义与写入。
- `sources/sglang/python/sglang/srt/mem_cache/layout/token_major.py`：unified entry 上的 per-layer strided views。
- `sources/sglang/python/sglang/srt/mem_cache/radix_cache.py`：默认模式按统一 loc/page indices 驱逐。
- `sources/sglang/python/sglang/srt/mem_cache/allocator/paged.py`：把 locations 还原成 page IDs 并释放。
- `sources/sglang/python/sglang/srt/mem_cache/unified_radix_cache.py`：unified cache 按 component 路由驱逐。
- `sources/sglang/python/sglang/srt/mem_cache/unified_cache/components/`：Full、SWA、Mamba component 的独立释放动作。
- `sources/sglang/python/sglang/srt/mem_cache/kv/plan.py`：sub-pool 与其 `layer_ids` 的规划结构。
- `sources/sglang/python/sglang/srt/mem_cache/kv/storage.py`：最终以 `sub_pool + unit IDs` 为粒度释放。

以上结论固定于文首两个 commit。后续版本若修改 cache group、pool 或 unified-memory 实现，需要重新核对相关代码。
