$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$markdownPath = Join-Path $root 'hybrid_kv_cache_storage_research.md'
$outputPath = Join-Path $root 'hybrid_kv_cache_storage_research.html'

$converted = ConvertFrom-Markdown -LiteralPath $markdownPath
$body = $converted.Html

# The HTML has its own compact title and question section. Keep the report body
# from the first technical section onward so the Markdown remains the source of truth.
$firstSection = $body.IndexOf('<h2 ')
if ($firstSection -lt 0) {
    throw 'No level-2 section found in the Markdown report.'
}
$body = $body.Substring($firstSection)

# Heading numbers in the edited Markdown are intentionally not used as HTML
# navigation identifiers; the hierarchy is conveyed by the document outline.
$body = [regex]::Replace(
    $body,
    '(<h[2-4][^>]*>)\d+(?:\.\d+)*\.?\s+',
    '$1'
)

# Mark every source-backed Python excerpt first. These are grouped into their
# enclosing evidence section below, so no prose or citation is left orphaned.
$codePattern = '(?s)<p>源码路径：(.*?)</p>\s*<pre><code class="language-python">(.*?)</code></pre>'
$body = [regex]::Replace($body, $codePattern, {
    param($match)
    $source = $match.Groups[1].Value
    $code = $match.Groups[2].Value
    return @"
<div class="source-proof">
  <div class="source-path">$source</div>
  <pre><code class="language-python">$code</code></pre>
</div>
"@
})

# A source citation can occasionally cover two consecutive excerpts. In that
# case only the immediately following, still-unwrapped Python block is a
# continuation of the same proof.
$body = [regex]::Replace(
    $body,
    '(?s)</pre></div>\s*<pre><code class="language-python">(.*?)</code></pre>',
    '</pre></div><div class="source-proof"><div class="source-path">同一证明链的后续片段</div><pre><code class="language-python">$1</code></pre></div>'
)

# For evidence-oriented h4 sections, keep a concise conclusion visible and put
# the entire explanation, source paths, and code inside one disclosure block.
$proofSummaries = @{
    '代码证据 1：' = '同组多层被放入同一个 backing allocation；切出的每层 view 仍用同一逻辑 block 维度索引。'
    '代码证据 2：' = '同一 cache group 的各层取得相同 slot mapping，因此 block ID 和 block 内位置一致。'
    '代码证据 3：' = '每层用自己的 cache view 和共享 slot，将本层 K/V 写入同一逻辑 block 位置。'
    'vLLM：按 pattern 位置拆成多个 groups' = 'vLLM 的 group 可能覆盖模型深度上不连续的层，当前最细驱逐边界仍是 group block。'
    'SGLang：按 lifetime/window 汇总为较大的 sub-pool' = 'SGLang 的 sub-pool 可能汇总不连续层，当前最细边界仍是 sub-pool page/unit。'
}
$body = [regex]::Replace(
    $body,
    '(?s)(<h4[^>]*>(.*?)</h4>)(.*?)(?=<h[1-4][^>]*>|\z)',
    {
        param($match)
        $heading = $match.Groups[1].Value
        $headingText = [System.Net.WebUtility]::HtmlDecode(
            [regex]::Replace($match.Groups[2].Value, '<[^>]+>', '')
        )
        $content = $match.Groups[3].Value
        if (
            $content -notmatch 'class="source-proof"' -or
            $content -match 'class="evidence-details"'
        ) {
            return $match.Value
        }

        $summary = $null
        foreach ($key in $proofSummaries.Keys) {
            if ($headingText.StartsWith($key)) {
                $summary = $proofSummaries[$key]
                break
            }
        }
        if (-not $summary) {
            $summary = '本节给出对应实现路径与源码依据。'
        }

        $sourceCount = ([regex]::Matches($content, 'class="source-proof"')).Count
        return @"
$heading
<p class="evidence-summary"><strong>简要：</strong>$summary</p>
<details class="evidence-details">
  <summary>展开说明和源码证明（$sourceCount 处）</summary>
  <div class="evidence-content">
$content
  </div>
</details>
"@
    }
)

# Keep each dedicated evidence section as a single disclosure: once expanded,
# all of its explanations, paths, and source excerpts are visible together.
# Source snippets elsewhere remain individually collapsible.
$evidenceBlocks = [System.Collections.Generic.List[string]]::new()
$body = [regex]::Replace(
    $body,
    '(?s)<details class="evidence-details">.*?</details>',
    {
        param($match)
        $index = $evidenceBlocks.Count
        $evidenceBlocks.Add($match.Value)
        return "__EVIDENCE_BLOCK_$index`__"
    }
)

# Also fold source-backed h3 sections as a whole when they are not already
# inside a dedicated evidence subsection.
$sectionSummaries = @{
    'block 只按 cache group 分配一次' = '每个 cache group 申请自己的逻辑 block；manager 不按层重复分配。'
    '同一个 group block ID 被送给 group 中所有 layer' = '同组各层使用相同 block/slot 映射，而不是各自维护独立编号。'
    '物理显存排布：block-outer 与 layer-outer 只改变物理展开顺序' = '两种 layout 只改变物理 page 的相邻关系和 stride；逻辑 block 的定义不变。'
    'cache location 只分配一次，不包含 layer 维' = 'SGLang 对请求位置只保存一份逻辑 cache location，映射表没有 layer 轴。'
    '默认模式下，同一 page ID 分别索引各层 buffer' = '默认模式物理上每层分 buffer，但各层用同一 page ID 定位对应 payload。'
    'unified entry 直接包含多层 payload' = 'unified 模式把同一 slot 下多个 layer 的 rows 放入一个 entry。'
}
$sectionBlocks = [System.Collections.Generic.List[string]]::new()
$body = [regex]::Replace(
    $body,
    '(?s)(<h3[^>]*>(.*?)</h3>)(.*?)(?=<h[23][^>]*>|\z)',
    {
        param($match)
        $heading = $match.Groups[1].Value
        $headingText = [System.Net.WebUtility]::HtmlDecode(
            [regex]::Replace($match.Groups[2].Value, '<[^>]+>', '')
        )
        $content = $match.Groups[3].Value
        if (
            $content -notmatch 'class="source-proof"' -or
            $content -match '__EVIDENCE_BLOCK_|evidence-bundle'
        ) {
            return $match.Value
        }

        $summary = $null
        foreach ($key in $sectionSummaries.Keys) {
            if ($headingText.StartsWith($key)) {
                $summary = $sectionSummaries[$key]
                break
            }
        }
        if (-not $summary) {
            $summary = '本节给出对应实现过程与源码依据。'
        }

        $sourceCount = ([regex]::Matches($content, 'class="source-proof"')).Count
        $index = $sectionBlocks.Count
        $sectionBlocks.Add(@"
<p class="evidence-summary"><strong>简要：</strong>$summary</p>
<details class="evidence-details">
  <summary>展开说明与源码证明（$sourceCount 处）</summary>
$content
</details>
"@)
        return "$heading`n__SECTION_BLOCK_$index`__"
    }
)

$body = [regex]::Replace(
    $body,
    '(?s)<div class="source-proof">\s*<div class="source-path">(.*?)</div>\s*(<pre><code class="language-python">.*?</code></pre>)\s*</div>',
    '<details class="code-proof"><summary><span>展开源码证明</span><span class="source-path">$1</span></summary>$2</details>'
)

for ($index = 0; $index -lt $sectionBlocks.Count; $index++) {
    $body = $body.Replace("__SECTION_BLOCK_$index`__", $sectionBlocks[$index])
}

for ($index = 0; $index -lt $evidenceBlocks.Count; $index++) {
    $body = $body.Replace("__EVIDENCE_BLOCK_$index`__", $evidenceBlocks[$index])
}

# Present the vLLM logical-block evidence as one large, single-click disclosure.
# Keep the logical definition visible; place all three numbered explanations,
# source paths, and code snippets together inside the disclosure.
$body = [regex]::Replace(
    $body,
    '(?s)(<h3 id="vllm-block">.*?</h3>)(.*?)(?=<h3[^>]*>物理显存排布：block-outer 与 layer-outer 只改变物理展开顺序)',
    {
        param($match)
        $heading = $match.Groups[1].Value
        $section = $match.Groups[2].Value

        $firstProof = [regex]::Match($section, '<h4[^>]*>代码证据 1：.*?</h4>')
        if (-not $firstProof.Success) { return $match.Value }

        $visible = $section.Substring(0, $firstProof.Index)
        $proofs = $section.Substring($firstProof.Index)

        # Remove the generic wrapper created around the whole parent section,
        # while keeping the logical-block definition and its short lead-in.
        $visible = [regex]::Replace(
            $visible,
            '(?s)<p class="evidence-summary"><strong>简要：</strong>本节给出对应实现过程与源码依据。</p>\s*<details class="evidence-details">\s*<summary>.*?</summary>\s*',
            ''
        )

        # Remove the old per-proof disclosure controls from the inside, from
        # innermost to outermost, but preserve their headings and full contents.
        do {
            $previous = $proofs
            $proofs = [regex]::Replace(
                $proofs,
                '(?s)<details class="evidence-details(?: [^"]*)?">\s*<summary>.*?</summary>((?:(?!<details class="evidence-details).)*?)</details>',
                '$1'
            )
        } while ($proofs -ne $previous)

        # Source paths previously displayed in the small code summaries should
        # remain visible alongside their code once the large fold is opened.
        $proofs = [regex]::Replace(
            $proofs,
            '(?s)<details class="code-proof(?: [^"]*)?">\s*<summary>.*?<span class="source-path">(.*?)</span></summary>\s*(<pre><code class="language-python">.*?</code></pre>)\s*</details>',
            '<div class="source-proof"><div class="source-path">$1</div>$2</div>'
        )

        # The old parent wrapper is now the only remaining details close.
        $lastClose = $proofs.LastIndexOf('</details>')
        if ($lastClose -ge 0) {
            $proofs = $proofs.Remove($lastClose, '</details>'.Length)
        }

        return @"
$heading
$visible
<p class="evidence-summary"><strong>简要：</strong>三段证据连成一条链：同组 layers 共享逻辑 block/slot，各层使用自己的 cache view 分别写入对应 payload。</p>
<details class="evidence-details evidence-bundle">
  <summary>展开完整说明与源码证明（连续编号 1–3）</summary>
  <div class="evidence-content">
$proofs
  </div>
</details>
"@
    }
)

# The physical-layout discussion is a large outer disclosure. Its source-level
# stride explanation remains a nested disclosure inside it.
$body = [regex]::Replace(
    $body,
    '(?s)(<h3[^>]*>物理显存排布：block-outer 与 layer-outer 只改变物理展开顺序</h3>)(.*?)(?=<h3[^>]*>Hybrid 对 vLLM 逻辑 block 的影响)',
    {
        param($match)
        $heading = $match.Groups[1].Value
        $content = $match.Groups[2].Value
        return @"
$heading
<p class="evidence-summary"><strong>简要：</strong>物理 block-outer 让同一逻辑 block 的多层物理 pages 相邻；物理 layer-outer 让同一 layer 的多个物理 pages 相邻。两者不改变逻辑 block。</p>
<details class="evidence-details layout-bundle">
  <summary>展开物理显存排布说明</summary>
  <div class="evidence-content">
$content
  </div>
</details>
"@
    }
)

$vllmDiagram = @'
<figure class="architecture-figure" aria-labelledby="vllm-diagram-title">
  <figcaption id="vllm-diagram-title">
    <strong>vLLM 逻辑层次</strong>
    <span>cache group 直接管理逻辑 block；每个 block 是该 group 多层 payload 的共同逻辑单位。</span>
  </figcaption>
  <svg viewBox="0 0 1120 610" role="img" aria-labelledby="vllm-svg-title vllm-svg-desc">
    <title id="vllm-svg-title">vLLM cache group、block 与 layer cache 的层次结构</title>
    <desc id="vllm-svg-desc">两个 cache group 分别维护 block ID。每个逻辑 block 画成一个整体容器，内部列出该 group 中多个不一定连续的 layer cache payload。</desc>
    <defs>
      <marker id="v-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 Z" fill="#718096" />
      </marker>
    </defs>

    <g transform="translate(20 20)">
      <rect class="group-box group-a" width="1080" height="280" rx="6" />
      <text class="group-title" x="24" y="34">Cache group 0</text>
      <text class="group-note" x="24" y="57">一套逻辑 block ID；group 中的层可以不连续：Layer 0 / 4 / 8</text>

      <rect class="id-source" x="28" y="88" width="180" height="150" rx="5" />
      <text class="block-title" x="118" y="117" text-anchor="middle">group 0 block IDs</text>
      <text class="id-row" x="118" y="151" text-anchor="middle">0</text>
      <text class="id-row" x="118" y="181" text-anchor="middle">1</text>
      <text class="id-row" x="118" y="211" text-anchor="middle">…</text>
      <path class="connector" marker-end="url(#v-arrow)" d="M208 145 L257 145" />
      <path class="connector" marker-end="url(#v-arrow)" d="M208 180 L652 180" />

      <g transform="translate(270 77)">
        <rect class="logical-unit logical-a" width="350" height="178" rx="5" />
        <text class="logical-title" x="175" y="29" text-anchor="middle">逻辑 Block 0（一个整体）</text>
        <rect class="payload payload-a" x="18" y="54" width="96" height="88" rx="4" />
        <text class="payload-label" x="66" y="86" text-anchor="middle">Layer 0</text>
        <text class="payload-sub" x="66" y="109" text-anchor="middle">cache(0, 0)</text>
        <rect class="payload payload-a" x="127" y="54" width="96" height="88" rx="4" />
        <text class="payload-label" x="175" y="86" text-anchor="middle">Layer 4</text>
        <text class="payload-sub" x="175" y="109" text-anchor="middle">cache(4, 0)</text>
        <rect class="payload payload-a" x="236" y="54" width="96" height="88" rx="4" />
        <text class="payload-label" x="284" y="86" text-anchor="middle">Layer 8</text>
        <text class="payload-sub" x="284" y="109" text-anchor="middle">cache(8, 0)</text>
        <text class="unit-note" x="175" y="164" text-anchor="middle">同一个 block ID 0 覆盖三层 payload</text>
      </g>

      <g transform="translate(665 77)">
        <rect class="logical-unit logical-a" width="350" height="178" rx="5" />
        <text class="logical-title" x="175" y="29" text-anchor="middle">逻辑 Block 1（一个整体）</text>
        <rect class="payload payload-a" x="18" y="54" width="96" height="88" rx="4" />
        <text class="payload-label" x="66" y="86" text-anchor="middle">Layer 0</text>
        <text class="payload-sub" x="66" y="109" text-anchor="middle">cache(0, 1)</text>
        <rect class="payload payload-a" x="127" y="54" width="96" height="88" rx="4" />
        <text class="payload-label" x="175" y="86" text-anchor="middle">Layer 4</text>
        <text class="payload-sub" x="175" y="109" text-anchor="middle">cache(4, 1)</text>
        <rect class="payload payload-a" x="236" y="54" width="96" height="88" rx="4" />
        <text class="payload-label" x="284" y="86" text-anchor="middle">Layer 8</text>
        <text class="payload-sub" x="284" y="109" text-anchor="middle">cache(8, 1)</text>
        <text class="unit-note" x="175" y="164" text-anchor="middle">同一个 block ID 1 覆盖三层 payload</text>
      </g>
    </g>

    <g transform="translate(20 325)">
      <rect class="group-box group-b" width="1080" height="245" rx="6" />
      <text class="group-title" x="24" y="34">Cache group 1</text>
      <text class="group-note" x="24" y="57">另一套独立 block IDs；示例层为 Layer 3 / 7</text>

      <rect class="id-source" x="28" y="84" width="180" height="125" rx="5" />
      <text class="block-title" x="118" y="113" text-anchor="middle">group 1 block IDs</text>
      <text class="id-row" x="118" y="149" text-anchor="middle">0　　1　　…</text>
      <path class="connector" marker-end="url(#v-arrow)" d="M208 147 L257 147" />
      <path class="connector" marker-end="url(#v-arrow)" d="M208 168 L652 168" />

      <g transform="translate(270 76)">
        <rect class="logical-unit logical-b" width="350" height="146" rx="5" />
        <text class="logical-title" x="175" y="29" text-anchor="middle">逻辑 Block 0（一个整体）</text>
        <rect class="payload payload-b" x="44" y="52" width="116" height="68" rx="4" />
        <text class="payload-label" x="102" y="80" text-anchor="middle">Layer 3</text>
        <text class="payload-sub" x="102" y="102" text-anchor="middle">cache(3, 0)</text>
        <rect class="payload payload-b" x="190" y="52" width="116" height="68" rx="4" />
        <text class="payload-label" x="248" y="80" text-anchor="middle">Layer 7</text>
        <text class="payload-sub" x="248" y="102" text-anchor="middle">cache(7, 0)</text>
      </g>

      <g transform="translate(665 76)">
        <rect class="logical-unit logical-b" width="350" height="146" rx="5" />
        <text class="logical-title" x="175" y="29" text-anchor="middle">逻辑 Block 1（一个整体）</text>
        <rect class="payload payload-b" x="44" y="52" width="116" height="68" rx="4" />
        <text class="payload-label" x="102" y="80" text-anchor="middle">Layer 3</text>
        <text class="payload-sub" x="102" y="102" text-anchor="middle">cache(3, 1)</text>
        <rect class="payload payload-b" x="190" y="52" width="116" height="68" rx="4" />
        <text class="payload-label" x="248" y="80" text-anchor="middle">Layer 7</text>
        <text class="payload-sub" x="248" y="102" text-anchor="middle">cache(7, 1)</text>
      </g>
    </g>
  </svg>
  <p class="figure-note">vLLM 图把逻辑 block 画成明确的跨层容器：block ID 属于 group，块内是该 group 各层在同一 ID 下的 payload；物理 outer layout 不改变这个逻辑整体。</p>
</figure>
'@

$sglangDiagram = @'
<figure class="architecture-figure" aria-labelledby="sglang-diagram-title">
  <figcaption id="sglang-diagram-title">
    <strong>SGLang 逻辑 page/slot 层次</strong>
    <span>每个 pool/sub-pool 管理一套 page IDs；一个逻辑 page 是同一 ID 在该 pool 多层上的 payload 集合。</span>
  </figcaption>
  <svg viewBox="0 0 1120 610" role="img" aria-labelledby="sglang-svg-title sglang-svg-desc">
    <title id="sglang-svg-title">SGLang pool、逻辑 page ID 与多层 payload 的关系</title>
    <desc id="sglang-svg-desc">每个 pool 有独立的 page ID 空间。矩阵中的每一行是一个逻辑 page，同一行跨越该 pool 管理的多个 layer payload。</desc>

    <g transform="translate(20 20)">
      <rect class="group-box group-a" width="1080" height="320" rx="6" />
      <text class="group-title" x="24" y="36">Window / SWA pool</text>
      <text class="group-note" x="24" y="61">独立 page ID 空间；示例层为 Layer 0 / 1 / 4</text>

      <text class="matrix-head" x="352" y="92" text-anchor="middle">Layer 0 payload</text>
      <text class="matrix-head" x="582" y="92" text-anchor="middle">Layer 1 payload</text>
      <text class="matrix-head" x="812" y="92" text-anchor="middle">Layer 4 payload</text>

      <g transform="translate(38 112)">
        <rect class="page-row" width="1004" height="58" rx="4" />
        <text class="page-id" x="90" y="35" text-anchor="middle">逻辑 Page 0</text>
        <rect class="payload payload-a" x="230" y="9" width="168" height="40" rx="3" />
        <text class="payload-label" x="314" y="34" text-anchor="middle">payload(0, 0)</text>
        <rect class="payload payload-a" x="460" y="9" width="168" height="40" rx="3" />
        <text class="payload-label" x="544" y="34" text-anchor="middle">payload(1, 0)</text>
        <rect class="payload payload-a" x="690" y="9" width="168" height="40" rx="3" />
        <text class="payload-label" x="774" y="34" text-anchor="middle">payload(4, 0)</text>
        <text class="row-brace" x="930" y="35" text-anchor="middle">同一 ID 0</text>
      </g>

      <g transform="translate(38 180)">
        <rect class="page-row page-row-selected" width="1004" height="58" rx="4" />
        <text class="page-id" x="90" y="35" text-anchor="middle">逻辑 Page 1</text>
        <rect class="payload payload-a" x="230" y="9" width="168" height="40" rx="3" />
        <text class="payload-label" x="314" y="34" text-anchor="middle">payload(0, 1)</text>
        <rect class="payload payload-a" x="460" y="9" width="168" height="40" rx="3" />
        <text class="payload-label" x="544" y="34" text-anchor="middle">payload(1, 1)</text>
        <rect class="payload payload-a" x="690" y="9" width="168" height="40" rx="3" />
        <text class="payload-label" x="774" y="34" text-anchor="middle">payload(4, 1)</text>
        <text class="row-brace" x="930" y="35" text-anchor="middle">同一 ID 1</text>
      </g>

      <g transform="translate(38 248)">
        <rect class="page-row page-row-muted" width="1004" height="46" rx="4" />
        <text class="page-id" x="90" y="30" text-anchor="middle">逻辑 Page …</text>
        <text class="ellipsis" x="544" y="32" text-anchor="middle">…</text>
      </g>
    </g>

    <g transform="translate(20 365)">
      <rect class="group-box group-b" width="1080" height="205" rx="6" />
      <text class="group-title" x="24" y="36">Full-attention pool</text>
      <text class="group-note" x="24" y="61">另一套独立 page IDs；示例层为 Layer 3 / 7</text>

      <text class="matrix-head" x="455" y="88" text-anchor="middle">Layer 3 payload</text>
      <text class="matrix-head" x="705" y="88" text-anchor="middle">Layer 7 payload</text>

      <g transform="translate(38 105)">
        <rect class="page-row" width="1004" height="58" rx="4" />
        <text class="page-id" x="90" y="35" text-anchor="middle">逻辑 Page 0</text>
        <rect class="payload payload-b" x="300" y="9" width="210" height="40" rx="3" />
        <text class="payload-label" x="405" y="34" text-anchor="middle">payload(3, 0)</text>
        <rect class="payload payload-b" x="550" y="9" width="210" height="40" rx="3" />
        <text class="payload-label" x="655" y="34" text-anchor="middle">payload(7, 0)</text>
        <text class="row-brace" x="900" y="35" text-anchor="middle">同一 ID 0</text>
      </g>
    </g>
  </svg>
  <p class="figure-note">按逻辑 page 观察时，每一行才是一个完整 page：page ID 只分配一次，并共同标识该 pool 中各层对应的 payload。不同 pool 各自维护独立的 page ID 空间和生命周期。</p>
</figure>
'@

$vllmIntro = @'
<div class="section-summary">
  <p><strong>核心：</strong>vLLM 的逻辑 block 属于 cache group。group 内所有层复用同一个 block ID / slot mapping，各层分别把自己的 K/V 写入该 ID 对应的位置。</p>
  <p>因此，一个逻辑 block 可表示为 <code>{cache(layer_0, b), cache(layer_1, b), ...}</code>。block-outer 与 layer-outer 只改变这些 payload 的物理排列。</p>
</div>
'@

$sglangIntro = @'
<div class="section-summary">
  <p><strong>核心：</strong>SGLang 的 page/slot ID 由 pool/sub-pool 分配一次，并由该范围内的多层共同使用。</p>
  <p>因此，一个逻辑 page <code>p</code> 应理解为 <code>{payload(layer_0, p), payload(layer_1, p), ...}</code>。匹配、引用与释放面对的是这一跨层逻辑 page，而不是每层各自维护一套逻辑 page。</p>
</div>
'@

$body = [regex]::Replace(
    $body,
    '(<h2[^>]*>vLLM：.*?</h2>)',
    '$1' + "`n" + $vllmIntro + "`n" + $vllmDiagram,
    [System.Text.RegularExpressions.RegexOptions]::Singleline
)

$body = [regex]::Replace(
    $body,
    '(<h2[^>]*>SGLang：.*?</h2>)',
    '$1' + "`n" + $sglangIntro + "`n" + $sglangDiagram,
    [System.Text.RegularExpressions.RegexOptions]::Singleline
)

# Highlight report-authored explanation comments without changing the source
# excerpts themselves. Consecutive comment lines after “【说明】” are treated as
# one explanation block so wrapped explanations keep the same visual emphasis.
$body = [regex]::Replace(
    $body,
    '(?s)(<pre><code[^>]*>)(.*?)(</code></pre>)',
    {
        param($match)
        $code = [regex]::Replace(
            $match.Groups[2].Value,
            '(?m)^([ \t]*# 【说明】[^\r\n]*(?:\r?\n[ \t]*#[^\r\n]*)*)',
            '<span class="explanation-comment">$1</span>'
        )
        return $match.Groups[1].Value + $code + $match.Groups[3].Value
    }
)

$document = @"
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>vLLM 与 SGLang Hybrid KV Cache 调研</title>
  <style>
    :root {
      color-scheme: light;
      --page: #f6f7f9;
      --paper: #ffffff;
      --text: #20242a;
      --muted: #667085;
      --line: #d9dee7;
      --line-strong: #bcc5d2;
      --soft: #f2f4f7;
      --blue: #315f8c;
      --blue-soft: #eaf2f9;
      --green: #3c6e57;
      --green-soft: #eaf4ef;
      --amber: #855f22;
      --amber-soft: #faf2df;
      --code: #ffffff;
      --code-text: #17202a;
    }

    * { box-sizing: border-box; }

    html { scroll-behavior: smooth; }

    body {
      margin: 0;
      background: var(--page);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
      line-height: 1.72;
    }

    .page {
      width: min(1120px, calc(100% - 40px));
      margin: 28px auto 64px;
      padding: 52px 64px 72px;
      background: var(--paper);
      border: 1px solid var(--line);
      box-shadow: 0 8px 24px rgba(31, 38, 48, 0.06);
    }

    .document-header {
      padding-bottom: 28px;
      border-bottom: 1px solid var(--line);
    }

    h1, h2, h3, h4 { line-height: 1.35; font-weight: 600; color: #18202a; }
    h1 { margin: 0 0 12px; font-size: 32px; letter-spacing: 0; }
    h2 { margin: 56px 0 18px; padding-bottom: 10px; border-bottom: 1px solid var(--line); font-size: 25px; }
    h3 { margin: 38px 0 12px; font-size: 20px; }
    h4 { margin: 28px 0 10px; font-size: 17px; }
    p { margin: 10px 0; }
    ul, ol { padding-left: 24px; }
    li + li { margin-top: 6px; }
    strong { font-weight: 600; }
    a { color: var(--blue); text-underline-offset: 3px; }
    code { font-family: "Cascadia Code", "SFMono-Regular", Consolas, monospace; }
    p code, li code, td code { padding: 2px 5px; border-radius: 3px; background: var(--soft); font-size: 0.93em; }

    .subtitle { margin: 0; color: var(--muted); font-size: 16px; }
    .meta { margin-top: 12px; color: var(--muted); font-size: 14px; }

    .question-section { margin-top: 30px; }
    .question-section h2 { margin-top: 0; border: 0; padding: 0; font-size: 21px; }
    .question-list { display: grid; gap: 12px; margin-top: 14px; }
    .question {
      display: grid;
      grid-template-columns: 34px 1fr;
      gap: 12px;
      align-items: start;
      padding: 16px 18px;
      border: 1px solid var(--line);
      background: #fbfcfd;
    }
    .question-number {
      display: grid;
      place-items: center;
      width: 30px;
      height: 30px;
      color: var(--paper);
      background: #475467;
      border-radius: 3px;
      font-weight: 600;
    }
    .question p { margin: 1px 0 0; }

    .terms {
      margin-top: 18px;
      padding: 13px 16px;
      border-left: 3px solid var(--line-strong);
      background: var(--soft);
      color: #3d4652;
      font-size: 14px;
    }

    .section-summary {
      margin: 16px 0 24px;
      padding: 15px 18px;
      border-left: 4px solid var(--blue);
      background: var(--blue-soft);
    }
    .section-summary p { margin: 5px 0; }

    .architecture-figure {
      margin: 24px 0 34px;
      padding: 18px;
      border: 1px solid var(--line);
      background: #fcfcfd;
    }
    .architecture-figure figcaption { display: flex; flex-direction: column; gap: 2px; margin-bottom: 12px; }
    .architecture-figure figcaption strong { font-size: 17px; }
    .architecture-figure figcaption span, .figure-note { color: var(--muted); font-size: 14px; }
    .architecture-figure svg { display: block; width: 100%; height: auto; }
    .figure-note { margin: 10px 2px 0; }

    .diagram-root { fill: #eef1f5; stroke: #aeb8c5; stroke-width: 1.5; }
    .node-title { font-size: 20px; font-weight: 600; fill: #1f2937; }
    .connector { fill: none; stroke: #718096; stroke-width: 1.7; }
    .group-box { stroke-width: 1.5; }
    .group-a { fill: #f2f7fb; stroke: #9db8d0; }
    .group-b { fill: #f1f7f3; stroke: #9dbda9; }
    .group-title { font-size: 18px; font-weight: 600; fill: #25303c; }
    .group-note { font-size: 13px; fill: #667085; }
    .block-box { fill: #ffffff; stroke: #bbc5d1; stroke-width: 1.2; }
    .block-muted { fill: #f7f8fa; }
    .block-title { font-size: 15px; font-weight: 600; fill: #344054; }
    .payload { stroke-width: 1; }
    .payload-a { fill: #dfeef8; stroke: #8bb1cd; }
    .payload-b { fill: #e2f0e7; stroke: #8eb39b; }
    .payload-label { font-size: 14px; font-weight: 600; fill: #27313d; }
    .payload-sub { font-size: 11px; fill: #596575; }
    .ellipsis { font-size: 28px; fill: #98a2b3; }
    .id-source { fill: #ffffff; stroke: #aeb8c5; stroke-width: 1.2; }
    .id-row { font-size: 16px; font-weight: 600; fill: #315f8c; }
    .logical-unit { fill: #ffffff; stroke-width: 2; }
    .logical-a { stroke: #6f9fc3; }
    .logical-b { stroke: #72a184; }
    .logical-title { font-size: 16px; font-weight: 600; fill: #25303c; }
    .unit-note { font-size: 12px; fill: #667085; }
    .mode-panel { stroke-width: 1.5; }
    .default-panel { fill: #f7f9fc; stroke: #9db8d0; }
    .unified-panel { fill: #f4f8f5; stroke: #9dbda9; }
    .tensor-frame { fill: #ffffff; stroke: #aeb8c5; stroke-width: 1.2; }
    .tensor-label { font-size: 14px; font-weight: 600; fill: #344054; }
    .slot-box { fill: #f3f5f7; stroke: #bbc5d1; stroke-width: 1; }
    .slot-selected { fill: #dfeef8; stroke: #5f91b6; stroke-width: 1.5; }
    .slot-label { font-size: 12px; font-weight: 600; fill: #344054; }
    .shared-id-band { fill: rgba(49, 95, 140, 0.06); stroke: #5f91b6; stroke-width: 1.5; stroke-dasharray: 6 5; }
    .band-label { font-size: 13px; font-weight: 600; fill: #315f8c; }
    .unified-entry { fill: #ffffff; stroke: #8eb39b; stroke-width: 1.5; }
    .selected-entry { stroke: #3c6e57; stroke-width: 2.5; }
    .matrix-head { font-size: 13px; font-weight: 600; fill: #344054; }
    .page-row { fill: #ffffff; stroke: #aeb8c5; stroke-width: 1.2; }
    .page-row-selected { stroke: #5f91b6; stroke-width: 2.2; }
    .page-row-muted { fill: #f7f8fa; }
    .page-id { font-size: 15px; font-weight: 600; fill: #315f8c; }
    .row-brace { font-size: 12px; font-weight: 600; fill: #667085; }

    blockquote {
      margin: 16px 0;
      padding: 8px 16px;
      border-left: 3px solid var(--line-strong);
      color: var(--muted);
      background: var(--soft);
    }

    pre {
      margin: 0;
      padding: 17px 19px;
      overflow: auto;
      color: var(--code-text);
      background: var(--code);
      border: 1px solid var(--line);
      border-radius: 0 0 4px 4px;
      line-height: 1.55;
      tab-size: 4;
    }
    pre code { font-size: 13px; }
    pre code.language-text { color: var(--code-text); }
    .explanation-comment {
      display: block;
      margin: 0 -9px;
      padding: 1px 9px;
      color: #8a3f0a;
      background: #fff0c2;
      border-left: 4px solid #d97706;
      font-weight: 600;
    }

    details.code-proof, details.evidence-details {
      margin: 14px 0 20px;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #fbfcfd;
    }
    details.evidence-bundle { margin-top: 10px; }
    .evidence-bundle .evidence-content h4 { margin-top: 28px; }
    details.layout-bundle { margin-top: 10px; }
    .layout-bundle .evidence-details { margin-left: 0; margin-right: 0; }
    details.code-proof summary, details.evidence-details summary {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 14px;
      color: #344054;
      font-weight: 600;
      cursor: pointer;
      list-style-position: inside;
    }
    details.code-proof[open] summary, details.evidence-details[open] > summary { border-bottom: 1px solid var(--line); }
    details.evidence-details { padding-bottom: 2px; }
    details.evidence-details > :not(summary) { margin-inline: 14px; }
    details.evidence-details > :last-child { margin-bottom: 14px; }
    .evidence-summary { margin: 8px 0 12px; color: #3d4652; }
    .evidence-summary strong { color: #25364a; }
    .source-proof { margin: 14px 0 20px; }
    .source-proof .source-path { margin: 0 0 6px; }
    .source-path { color: var(--muted); font-size: 12px; font-weight: 400; text-align: right; overflow-wrap: anywhere; }
    .source-path code { background: transparent; padding: 0; }

    body > pre, .page > pre { margin: 14px 0 20px; border-radius: 4px; }

    table { width: 100%; border-collapse: collapse; margin: 18px 0 28px; font-size: 14px; }
    th, td { padding: 11px 12px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--line); }
    th { background: var(--soft); font-weight: 600; }

    hr { margin: 42px 0; border: 0; border-top: 1px solid var(--line); }

    .eviction-conclusion {
      margin-top: 18px;
      padding: 18px 20px;
      border: 1px solid #d5c39b;
      background: var(--amber-soft);
    }
    .eviction-conclusion strong { color: #684812; }
    .eviction-conclusion table { margin: 0; }
    .eviction-conclusion th { background: rgba(133, 95, 34, 0.08); }

    .footer-note { margin-top: 48px; padding-top: 18px; border-top: 1px solid var(--line); color: var(--muted); font-size: 13px; }

    @media (max-width: 820px) {
      .page { width: 100%; margin: 0; padding: 30px 22px 48px; border: 0; box-shadow: none; }
      h1 { font-size: 27px; }
      h2 { font-size: 22px; margin-top: 44px; }
      .architecture-figure { padding: 10px; margin-inline: -8px; overflow-x: auto; }
      .architecture-figure svg { min-width: 760px; }
      details.code-proof summary { align-items: flex-start; flex-direction: column; gap: 4px; }
      .source-path { text-align: left; }
    }

    @media print {
      body { background: white; }
      .page { width: 100%; margin: 0; padding: 0; border: 0; box-shadow: none; }
      details.code-proof:not([open]) > *:not(summary) { display: block; }
      details.code-proof summary { border-bottom: 1px solid var(--line); }
      .architecture-figure { break-inside: avoid; }
    }
  </style>
</head>
<body>
  <main class="page">
    <header class="document-header">
      <h1>vLLM 与 SGLang Hybrid KV Cache</h1>
      <p class="subtitle">逻辑 block/page 的多层组织方式与分层驱逐可行性</p>
      <p class="meta">基于本地源码调研 · 2026-11-18</p>

      <section class="question-section" aria-labelledby="main-questions">
        <h2 id="main-questions">主要问题</h2>
        <div class="question-list">
          <div class="question">
            <span class="question-number">1</span>
            <p><strong>块内部到底覆盖什么？</strong>vLLM 的一个逻辑 block、SGLang 的一个逻辑 page/slot，是只属于某一层，还是共同覆盖多层 cache payload？</p>
          </div>
          <div class="question">
            <span class="question-number">2</span>
            <p><strong>能否只驱逐部分层？</strong>在现有编号、匹配和 ownership 结构下，能否保留同一 block/page 中部分层的 cache，只释放另外一些层？</p>
          </div>
        </div>
        <div class="terms">
          <strong>本文口径：</strong>逻辑 block/page 指 manager、block table、请求映射或前缀匹配使用的编号对象；layer payload 指该编号在某一层对应的 K/V、latent 或 recurrent state。物理地址是否连续不是判断逻辑边界的依据。
        </div>
      </section>
    </header>

    $body

    <p class="footer-note">代码块来自正文所标注的本地源码路径。代码中的“【说明】”为报告添加的解释，不是上游源码注释。</p>
  </main>
</body>
</html>
"@

# Emphasize the already-written eviction conclusion without changing its text.
$document = [regex]::Replace(
    $document,
    '(<h3[^>]*>先给结论</h3>\s*)(<table>.*?</table>)',
    '$1<div class="eviction-conclusion">$2</div>',
    [System.Text.RegularExpressions.RegexOptions]::Singleline
)

[System.IO.File]::WriteAllText(
    $outputPath,
    $document,
    [System.Text.UTF8Encoding]::new($false)
)

Write-Output $outputPath
