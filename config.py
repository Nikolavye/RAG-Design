"""
RAGFlow 参数配置文件
====================
集中管理所有 RAGFlow 知识库、解析、检索、对话的参数。
修改参数后重新运行 ragflow_kb_manager.py 即可生效。

参数来源：RAGFlow v0.24 API  /v1/kb/update + ragflow-sdk
"""

# ══════════════════════════════════════════════════════════════════════════════
# 一、知识库基础配置
# ══════════════════════════════════════════════════════════════════════════════

KB_CONFIG = {

    # ── 知识库名称 ─────────────────────────────────────────────────────────────
    # 说明：创建知识库时使用的名称，同名会先删旧再建新
    # 取值：任意字符串
    # 示例："excavator_kb_v2"
    "dataset_name": "excavator_repair_kb",

    # ── 知识库描述 ─────────────────────────────────────────────────────────────
    # 说明：知识库的描述信息，仅用于标注，不影响检索
    # 取值：任意字符串
    "description": "挖掘机维修案例知识库，使用表格序列化预处理",

    # ── Embedding 模型 ─────────────────────────────────────────────────────────
    # 说明：将文字转成向量时使用的模型，直接影响语义检索质量
    # 取值（需在 RAGFlow 后台已配置对应 API key）：
    #   "text-embedding-3-small@OpenAI"     → OpenAI，性价比高，1536维
    #   "text-embedding-3-large@OpenAI"     → OpenAI，最高质量，3072维
    #   "gemini-embedding-001@Gemini"       → Google，免费配额大，768维
    #   "BAAI/bge-large-zh-v1.5@BAAI"      → 本地中文模型，无 API 费用
    #   "BAAI/bge-m3@BAAI"                 → 本地多语言模型
    # 建议：英文文档用 OpenAI；中文文档用 BAAI/bge-large-zh-v1.5
    "embedding_model": "gemini-embedding-001@Gemini",

    # ── 文档语言 ───────────────────────────────────────────────────────────────
    # 说明：影响分词策略，中文和英文处理逻辑不同
    # 取值："Chinese" | "English"
    "language": "English",

}


# ══════════════════════════════════════════════════════════════════════════════
# 二、文档解析 & 分块配置（parser_config）
# ══════════════════════════════════════════════════════════════════════════════

PARSER_CONFIG = {

    # ── 分块方法（最重要的参数）─────────────────────────────────────────────────
    # 说明：决定文档按什么逻辑切块，不同方法对应不同文档类型
    # 取值：
    #   "naive"          → 通用切块，按 token 数量切，适合大多数文档 ✅ 推荐
    #   "one"            → 整个文档作为一个 chunk，适合短文档或需要全文上下文
    #   "qa"             → LLM 从文档中生成 Q&A 对，用问题做向量，适合 FAQ 类文档
    #   "table"          → 专门处理结构化表格文件（Excel/CSV），不适合 PDF
    #   "paper"          → 学术论文专用，识别摘要/章节/参考文献
    #   "book"           → 长篇书籍，章节级分块
    #   "knowledge_graph"→ 提取实体关系，构建图谱
    # 示例：上传预处理后的 Markdown 文件 → 用 "naive"
    "chunk_method": "naive",

    # ── 布局识别引擎 ───────────────────────────────────────────────────────────
    # 说明：上传 PDF 时用哪个引擎识别文档结构（表格、标题、段落）
    #       上传 Markdown/TXT 时此参数无效，引擎自动跳过
    # 取值：
    #   "Naive"          → 纯文本提取，不识别布局，速度最快，表格效果差
    #   "DeepDOC"        → RAGFlow 内置 AI 引擎，适合普通 PDF，中等质量 ✅ 默认
    #   "Docling"        → IBM Docling，表格识别最准，需在容器内安装
    #   "MinerU"         → 学术 PDF 专用，公式/图表识别强
    # 建议：上传的是经过 table_serializer.py 处理的 .md 文件时，设 "Naive" 即可
    "layout_recognize": "Naive",

    # ── 每个 chunk 的 token 上限 ────────────────────────────────────────────────
    # 说明：控制每个 chunk 的最大长度（按 token 计算，约 1 token ≈ 0.75 个英文词）
    #       太小 → 上下文割裂，LLM 拿到的信息不完整
    #       太大 → 噪声增加，向量模糊，检索精度下降
    # 取值范围：128 ~ 4096（整数）
    # 建议值：
    #   纯文本文档        → 512
    #   含表格的文档      → 1024 ~ 2048
    #   已预处理的 .md    → 2048（每节"信息块+原始表格"正好一个 chunk）✅ 推荐
    #   整页式检索        → 4096
    "chunk_token_num": 2048,

    # ── 分块分隔符（父 chunk）──────────────────────────────────────────────────
    # 说明：naive 模式下，优先按这些字符切块，超过 token 上限才强制切
    #       对 Markdown 文件，用 "\n" 可以按段落自然切块
    # 取值：任意字符串，支持 \n（换行）\t（Tab）等转义
    # 示例："\n"（按行）| "\n\n"（按段落）| "---"（按 Markdown 分割线）
    "delimiter": "\n",

    # ── Parent-Child 分块（重要）────────────────────────────────────────────────
    # 说明：开启后，RAGFlow 会生成两级 chunk：
    #   Child chunk（小）→ 用于向量搜索，精准命中
    #   Parent chunk（大）→ 命中后返回给 LLM，提供完整上下文
    #   效果：搜索精度 ✅ + LLM 上下文完整性 ✅
    # 取值：True | False
    # 建议：表格类文档强烈推荐开启
    "enable_children": True,

    # ── 子 chunk 分隔符 ────────────────────────────────────────────────────────
    # 说明：RAGFlow 当前这套接口没有单独的 child_chunk_token_num，
    #       child 的粒度主要通过 children_delimiter 间接控制。
    #       如果按行（"\n"）切得太碎，可以改成按段/按语义块（"\n\n"）。
    # 取值：任意字符串
    # 示例："\n\n"（按段落/语义块，减少碎片化检索）
    "children_delimiter": "\n\n",

    # ── Chunk 重叠率 ───────────────────────────────────────────────────────────
    # 说明：相邻 chunk 之间重叠的比例，用于避免关键信息正好被切断
    #       增加重叠 → 减少边界信息丢失，但增加存储量和重复检索
    # 取值范围：0 ~ 50（百分比，整数）
    # 建议：
    #   已预处理的结构化文档 → 0（章节已自包含，不需要重叠）
    #   长篇连续文本         → 10 ~ 20
    "overlapped_percent": 0,

    # ── 自动关键词（近似 Contextual Retrieval）─────────────────────────────────
    # 说明：每个 chunk 额外让 LLM 生成 N 个关键词并加入索引
    #       效果类似 Anthropic 的 Contextual Retrieval，增强关键词搜索覆盖率
    #       会增加解析时间和 LLM token 消耗
    # 取值范围：0 ~ 10（整数，0 表示不开启）
    # 示例：5 → 每个 chunk 生成 5 个关键词
    #       chunk 内容："Replace engine rubber mounts"
    #       生成关键词：["rubber mount", "engine vibration", "idle shake", ...]
    "auto_keywords": 5,

    # ── 自动问题（近似 Q&A Chunking）──────────────────────────────────────────
    # 说明：每个 chunk 额外让 LLM 生成 N 个相关问题并加入索引
    #       用户提问时更容易与 chunk 匹配，特别适合问答场景
    #       会增加解析时间和 LLM token 消耗
    # 取值范围：0 ~ 5（整数，0 表示不开启）
    # 示例：3 → 生成 "What causes idle vibration?", "How to fix rubber mount?" 等
    "auto_questions": 3,

    # ── HTML 表格转 Excel 格式 ─────────────────────────────────────────────────
    # 说明：将 HTML 中的 <table> 标签内容转为 Excel 格式处理
    #       仅在上传含 HTML 表格的文档时有意义
    # 取值：True | False
    # 建议：上传 Markdown 文件时设 False
    "html4excel": False,

    # ── RAPTOR（层级摘要）─────────────────────────────────────────────────────
    # 说明：对所有 chunk 递归生成摘要，形成"知识树"
    #       优点：可以回答需要跨多个 chunk 综合推理的宏观问题
    #       缺点：解析时间极长，LLM 消耗大，小文档不值得开
    # 取值：{"use_raptor": True/False, "max_token": 512, "threshold": 0.3}
    # 建议：日常维修手册查询不需要，设 False
    "raptor": {
        "use_raptor": False,
        # "max_token": 512,    # RAPTOR 摘要节点的 token 上限
        # "threshold": 0.3,    # 相似度阈值，低于此值的 chunk 才合并
    },

    # ── GraphRAG（知识图谱）────────────────────────────────────────────────────
    # 说明：从文档中提取实体和关系，构建知识图谱，支持关联性查询
    #       适合：人物关系、产品依赖、流程关联等复杂关系型文档
    #       不适合：维修手册这类线性的案例文档
    # 取值：{"use_graphrag": True/False}
    # 建议：维修手册设 False，资源消耗大、效果提升不明显
    "graphrag": {
        "use_graphrag": False,
    },

}


# ══════════════════════════════════════════════════════════════════════════════
# 三、对话助手配置
# ══════════════════════════════════════════════════════════════════════════════

CHAT_CONFIG = {

    # ── 助手名称 ───────────────────────────────────────────────────────────────
    "assistant_name": "excavator_repair_assistant",

    # ── 检索返回的 chunk 数量（top_n）──────────────────────────────────────────
    # 说明：每次提问时，从知识库检索出 N 个最相关的 chunk 送给 LLM
    #       太少 → 可能漏掉相关信息
    #       太多 → LLM 上下文窗口被填满，推理变慢，且噪声增加
    # 取值范围：1 ~ 30（整数）
    # 建议：
    #   简单精确查询（"XX 故障怎么修"）→ 4 ~ 6
    #   需要综合多个案例的查询         → 8 ~ 12
    "top_n": 8,

    # ── 相似度阈值（similarity_threshold）─────────────────────────────────────
    # 说明：只有相似度高于此阈值的 chunk 才会被返回
    #       太高 → 容易没有结果（No relevant information found）
    #       太低 → 低质量 chunk 混入上下文，干扰 LLM
    # 取值范围：0.0 ~ 1.0（浮点数）
    # 建议：0.2（宽松）~ 0.4（严格）
    "similarity_threshold": 0.2,

    # ── 向量相似度权重 ─────────────────────────────────────────────────────────
    # 说明：混合检索时，向量搜索 vs 关键词搜索（BM25）的权重分配
    #       vector_similarity_weight = X → 关键词权重 = 1 - X
    #       向量权重高  → 语义理解强，但容易漏掉精确词匹配（型号、代码）
    #       关键词权重高 → 精确词匹配强，但语义理解弱
    # 取值范围：0.0 ~ 1.0（浮点数）
    # 建议：
    #   维修手册（含型号/代码）→ 0.5 ~ 0.7（向量和关键词都重要）
    #   纯语义查询场景         → 0.8 ~ 1.0
    "vector_similarity_weight": 0.7,

    # ── 显示引用来源 ───────────────────────────────────────────────────────────
    # 说明：回答中是否显示"来源于第X页/第X个文档"的引用标注
    # 取值：True | False
    "show_quote": True,

    # ── LLM 温度（temperature）─────────────────────────────────────────────────
    # 说明：控制 LLM 回答的随机性
    #       低温度 → 输出稳定、保守，适合精确的技术问答
    #       高温度 → 输出更有创意，适合开放性问题
    # 取值范围：0.0 ~ 1.0（浮点数）
    # 建议：技术维修手册 → 0.1 ~ 0.3
    "temperature": 0.2,

    # ── 回答最大 token 数 ──────────────────────────────────────────────────────
    # 说明：LLM 单次回答的最大长度
    # 取值范围：256 ~ 8192（整数，受所用 LLM 模型限制）
    # 建议：维修类问答 → 1024 ~ 2048
    "max_tokens": 2048,

    # ── System Prompt 模板 ─────────────────────────────────────────────────────
    # 说明：发给 LLM 的系统提示词，{knowledge} 是占位符，RAGFlow 会自动替换为检索到的 chunk
    #       好的 prompt 对回答质量影响巨大，比调参数更有效
    # 注意：必须包含 {knowledge} 占位符，否则检索结果不会被注入
    "prompt": """You are an experienced excavator maintenance expert.

Use ONLY the information provided in the knowledge base below to answer questions.
If the answer is not in the knowledge base, say so clearly — do not guess.

## Answer format
1. **Fault confirmation**: Identify the fault type and model
2. **Root cause**: State the most likely cause based on cases
3. **Troubleshooting steps**: List steps from simple to complex
4. **Solution**: Specific repair steps and parts needed
5. **Prevention**: How to avoid recurrence

## Rules
- Cite specific case details (machine model, machine no.) when available
- Keep answers concise and actionable
- If image links appear in the knowledge base, include them exactly as-is

Knowledge base:
{knowledge}
""",

}


# ══════════════════════════════════════════════════════════════════════════════
# 四、表格序列化配置（table_serializer.py 使用）
# ══════════════════════════════════════════════════════════════════════════════

SERIALIZER_CONFIG = {

    # ── 使用的 LLM 模型 ────────────────────────────────────────────────────────
    # 说明：表格序列化时调用的 LLM 模型
    # 取值（Google AI Studio）：
    #   "gemini-2.5-pro"           → 最高质量，理解复杂表格，推荐 ✅
    #   "gemini-2.0-flash"         → 速度快，成本低，质量略低
    #   "gemini-1.5-flash"         → 最便宜，简单表格够用
    "model": "gemini-2.5-pro",

    # ── 每个信息块的最大 token 数 ──────────────────────────────────────────────
    # 说明：LLM 生成的每张表格序列化内容的最大长度
    # 取值范围：500 ~ 3000（整数）
    # 建议：1500 够用，太大会超出向量模型的最优输入长度
    "max_output_tokens": 1500,

    # ── 上下文抓取字符数 ───────────────────────────────────────────────────────
    # 说明：抓取表格前后各多少字符作为上下文提供给 LLM
    #       太少 → LLM 不知道表格的主题
    #       太多 → 干扰信息增加，token 浪费
    # 取值范围：100 ~ 800（整数）
    "context_max_chars": 400,

    # ── 表格后最多抓取的文本段落数 ────────────────────────────────────────────
    # 说明：表格之后最多读几个文本块作为"after context"
    # 取值范围：1 ~ 5（整数）
    "context_after_max_blocks": 3,

    # ── LLM 温度 ───────────────────────────────────────────────────────────────
    # 说明：序列化时 LLM 的输出随机性，设 0 保证每次输出稳定一致
    # 取值范围：0.0 ~ 1.0
    "temperature": 0,

}


# ══════════════════════════════════════════════════════════════════════════════
# 五、快速调参指南
# ══════════════════════════════════════════════════════════════════════════════
"""
场景一：回答总是说"没有相关信息"
  → 降低 similarity_threshold（如 0.2 → 0.1）
  → 增加 top_n（如 8 → 12）
  → 检查 embedding_model 是否与文档语言匹配

场景二：回答不够准确，混入了不相关内容
  → 提高 similarity_threshold（如 0.2 → 0.35）
  → 减少 top_n（如 8 → 5）
  → 检查 chunk_token_num 是否过大

场景三：表格内容检索不到
  → 确认已用 table_serializer.py 预处理
  → 开启 auto_keywords（5）和 auto_questions（3）
  → 降低 vector_similarity_weight（如 0.7 → 0.5），增加关键词搜索权重

场景四：解析很慢
  → 关闭 raptor 和 graphrag（保持 False）
  → 关闭 auto_keywords 和 auto_questions（设 0）
  → 上传 .md 文件而不是 PDF（跳过 layout_recognize）

场景五：回答太短/太长
  → 调整 max_tokens（短 → 加大；长 → 减小）
  → 在 prompt 中明确说明回答长度要求
"""
