# pdf-ocrer

> 本项目修改自 [leon80148/pdf-ocrer](https://github.com/leon80148/pdf-ocrer)，
> 衷心感谢原作者的出色工作。此分支由 [nizhiyi](https://github.com/nizhiyi) 维护，
> 主要改动：OCR 引擎常驻子进程（模型只加载一次）、无文字文件不再产出、
> 界面与日志体验优化。后续更新发布在
> [nizhiyi/pdf-ocrer](https://github.com/nizhiyi/pdf-ocrer)。

批量 OCR 扫描版 PDF / 图片，生成可搜索、可复制文字的 PDF。输出文件固定命名为
`原文件名_OCR.pdf`，保存在原文件所在目录；原始文件绝不修改。

适合把扫描件、拍照件、图片型 PDF 一次性批量转成带文字层的档案，方便全文检索与复制。

## 功能

- **批量处理**：一次拖入/选择多个文件和多个文件夹，混合处理。
- **同名输出**：`合同.pdf → 合同_OCR.pdf`，落在原文件夹内。
- **智能跳过**：已有文字层的页面不重复 OCR；输出文件已存在时自动跳过（可勾选
  「全部重新处理」强制覆盖）。
- **图片输入**：支持 jpg / jpeg / png / tif / tiff，自动转成可搜索 PDF。
- **加密检测**：带密码的 PDF 自动跳过并在结果中标注。
- **独立引擎进程**：OCR 在独立子进程中一次处理一个文件，界面始终流畅；
  模型只在首个批次加载一次，之后常驻复用到程序关闭。
- **引擎依赖自动安装**：源码 / pip 安装模式下，所选 OCR 引擎缺依赖时开始处理前自动
  `pip install`（需联网）；打包安装版内置 rapidocr，无需安装。
- **现代卡片式界面**，跟随系统深色/浅色外观，全简体中文。

## 安装

### 方式一：Windows 安装包（推荐普通用户）

从 [Releases](https://github.com/nizhiyi/pdf-ocrer/releases) 下载
`pdf-ocrer-setup-<版本>.exe` 安装。安装版内置 RapidOCR 引擎与 PP-OCRv6 small 模型（离线可用）。

### 方式二：pip 安装（源码）

```bash
pip install pdf-ocrer                 # 内置 RapidOCR 引擎与默认依赖
```

Python 3.11+。默认使用 PP-OCRv6 small 模型（离线可用）；其它模型版本/大小首次使用需联网下载。

GPU 加速（仅源码/pip 安装，安装包为纯 CPU）：

```bash
pip install "pdf-ocrer[rapidocr-gpu-cuda]"   # NVIDIA，配合 device = "cuda"
pip install "pdf-ocrer[rapidocr-gpu-dml]"    # AMD/Intel 核显，配合 device = "dml"
```

## 使用

### 图形界面

直接运行 `pdf-ocrer`（无参数）或安装版中的 `pdf-ocrer-gui`：

1. 点「选择文件（夹）」→ 选择 PDF 文件（可多选）或文件夹；也可以把文件/文件夹
   直接拖进窗口，可多次拖入累积（例如先拖 2 个文件再拖 3 个文件夹）。
2. 点「开始」。输出文件 `原文件名_OCR.pdf` 出现在每个源文件所在目录。
3. 处理过程中可点「取消」；「重新处理」勾选后忽略已存在的 `_OCR` 输出强制重做；
   「覆盖原文件」勾选后 OCR 结果直接替换原文件（原文件先备份到 `<下载文件夹>\pdf-ocrer备份\`，
   备份或替换失败时自动保留 `_OCR` 文件，不丢数据）。

### 命令行

```bash
pdf-ocrer C:\Scans                        # 处理整个文件夹（默认仅顶层）
pdf-ocrer C:\Scans --recursive            # 递归扫描子文件夹
pdf-ocrer a.pdf b.pdf C:\Scans            # 文件与文件夹混合，可传多个
pdf-ocrer C:\Scans --force                # 输出已存在也全部重新处理
pdf-ocrer C:\Scans --dpi 300              # 覆盖渲染分辨率
pdf-ocrer --version
```

不带路径运行 `pdf-ocrer` 会启动图形界面。

## 输出规则

| 输入 | 输出 | 说明 |
|---|---|---|
| `合同.pdf`（扫描件） | `合同_OCR.pdf` | OCR 识别并植入文字层 |
| `报告.pdf`（已有文字层） | 无输出 | 全部页均已含文字层，状态「已OCR-跳过」 |
| `模糊.pdf`（OCR 无结果） | 无输出 | 没有识别出任何文字，状态「无文本-跳过」，不生成无文字层的拷贝 |
| `扫描.png` | `扫描_OCR.pdf` | 图片转可搜索 PDF |
| 加密 / 损坏的 PDF | 无输出 | 状态标注「加密-跳过」/「失败」 |

- 输出永远与源文件同目录；递归模式下子文件夹里的文件也输出到各自所在目录。
- 再次运行时，输出文件已存在的源文件自动跳过；用「全部重新处理」/ `--force` 覆盖。
- `*_OCR.pdf` 是本工具的输出文件：无论拖入、选择还是文件夹扫描都会自动
  忽略，绝不作为输入，覆盖模式下也绝不覆盖它们，避免重复识别与输出被改写。

## 配置

首次运行后可在「编辑设置」里调整常用项；完整选项见 `config.example.toml`，
配置文件位于 `config.toml`（安装版在 `%APPDATA%\pdf_ocrer\config.toml`）。

主要选项（节选）：

| 段 | 键 | 说明 |
|---|---|---|
| `[ocr]` | `ocr_version` | 模型版本：`PP-OCRv6`（默认）/ `PP-OCRv5` / `PP-OCRv4` |
| `[ocr]` | `dpi` | 渲染分辨率，默认 200（72–600） |
| `[ocr]` | `device` | `cpu` / `cuda` / `dml`（GPU 需另装对应套件） |
| `[ocr]` | `model_type` | 模型大小：v6 → tiny/small/medium；v4/v5 → mobile/server；`small` 默认内置 |
| `[output]` | `overwrite_original` | 覆盖原文件模式（GUI 可勾选）：OCR 结果替换原文件，原文件备份到下载文件夹，默认 false |
| `[input]` | `recursive` | 递归扫描子文件夹，默认 false |
| `[input]` | `image_extensions` | 支持的图片扩展名 |
| `[logging]` | `dir` | 日志目录（默认 `%LOCALAPPDATA%\pdf_ocrer\logs`） |

旧版本配置中的 `[naming]`、`[llm]`、`[watch]`、`[performance]` 段以及 `[ocr]` 里的
`engine`/`lang`/`enable_mkldnn`/`det_model_name`/`rec_model_name` 键已废弃，
加载时自动忽略（`[performance]` 在保存设置时自动移除）。

## 开发与测试

```bash
pip install -e .[dev]
pytest                            # 单元测试（无需网络与 OCR 模型）
pytest -o python_files="integration_smoke.py" tests/integration_smoke.py
                                  # 集成测试（需真实 OCR 模型，文件名不带 test_ 前缀，默认收集会跳过）
ruff check src tests
```

单元测试不依赖真实 OCR 模型；集成测试需真实模型，运行方式见上（它们在
`integration_smoke.py` 中，默认收集会跳过，因此单元测试输出没有 "deselected"）。

## License

MIT
