# Windows 安装与首次运行

## 环境要求

- Windows 10/11 x64。
- Python 3.11 或 3.12。项目当前依赖声明为 Python >=3.11；为与现有锁文件及依赖轮子保持一致，优先 Python 3.12。
- 安装时需要访问 Python 包索引以下载 uv 和项目依赖。
- 运行本地 Paper 需要网络访问 Binance 公共 USDⓈ-M 行情接口。
- 本包不提供 Python、依赖轮子或自包含 EXE，不适用于无网络安装。

## 安装

将包解压到新的、独立目录，不要解压到另一个已有机器人状态目录。PowerShell 当前目录切换到解压出的项目根目录，然后运行：

~~~powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
~~~

该命令只对本次 PowerShell 进程临时绕过脚本执行策略，不修改系统策略。安装脚本会：

1. 检查 Python 3.12 或 3.11。
2. 在项目目录创建 .venv。
3. 仅在该 .venv 内安装 uv。
4. 依据 uv.lock 安装锁定的运行依赖。
5. 加载 YAML 做配置导入检查。

安装不会启动机器人、创建 .env、读取凭据、访问私有 API、创建账本、发送委托、注册系统服务或启动 watchdog。若目标目录已有 .venv，脚本会停止以避免复用不明环境；请在全新目录解压，不要删除旧目录来绕过此检查。

## 启动本地 Paper

~~~powershell
.\run-local-paper.ps1
~~~

脚本显式设置 BOT_MODE=paper 和 BINANCE_TESTNET=false，之后运行 scripts/run_multi_paper.py。Paper 只调用公开行情接口，在本地创建模拟状态；默认模拟权益为 10,000 USDT，状态文件会在首次启动后写入 data/paper/portfolio.json，运行日志写入 logs/。按 Ctrl+C 停止。

首次启动会扫描市场并生成候选池；只有闭合 1H 决策周期才会产生逐币策略路由记录。候选池刷新不等于路由评估或交易信号。

可传递原 Runner 支持的参数，例如只做一次行情周期：

~~~powershell
.\run-local-paper.ps1 --once
~~~

本包没有预置历史账本。不要把其他电脑上的 Demo 账本直接复制进来充当同步状态；如要迁移已有仓位或待单，必须先做独立状态核对和人工迁移方案。

## USDⓈ-M Demo（仅在主动选择后）

Demo 使用测试环境私有 API，并会真实发送测试委托、创建保护止损。它不是本地模拟。不要在不了解当前 Demo 账户仓位和挂单状态时启动；同一 Demo 账户不得在旧 Runner 状态未知时由第二台电脑并行管理。

启用前至少完成以下准备：

1. 在 Binance USDⓈ-M Demo 控制台独立核对当前仓位、普通订单和 Algo 保护订单。
2. 确认任何其他电脑上的 Runner／watchdog 不会继续管理同一账户。
3. 检查对应账本与 Demo 账户能否一致恢复；本包不包含任何账本或状态迁移工具。
4. 只在 Windows Credential Manager 中录入 USDⓈ-M Demo 专用 API Key；不要使用正式账户 Key。
5. 在单独 PowerShell 会话中主动启动 Demo Runner。启动脚本从当前 Windows 用户的 Credential Manager 读取凭据，只在进程环境中临时提供，并将 Demo 账本与本地 Paper 账本分开保存。

~~~powershell
.\scripts\setup_demo_credentials.ps1
.\run-demo-paper.ps1 --once  # 仅运行一轮
.\run-demo-paper.ps1         # 持续运行，按 Ctrl+C 停止
~~~

Demo 状态默认写入 `data/paper/demo_portfolio.json`；当前启动脚本使用 EMA50 偏离候选策略：4H 收盘价相对 EMA50 偏离至少 1%，再由已收盘 1H K 线方向确认；持仓按 EMA20 中轨保护及结构止损管理。此操作会向 Binance USDⓈ-M Demo 发送测试委托，不会启动正式盘交易。候选策略仍属前瞻验证，不代表已证明有稳定收益。

Demo watchdog 文件随项目源码提供。本机 Startup 快捷方式已配置为启动该 watchdog；它会重启 EMA50 偏离候选 Runner，并检查 Demo 账本与最近交易所快照的同步状态。其他电脑不会因安装项目自动注册该启动项。

## 配置与状态

- configs/default.yaml 是策略参数唯一来源。编辑配置属于策略变更，需要单独设计和离线验证。
- 不要将 .env、API Key、Secret、账本、日志或账户导出放入 ZIP、代码仓库或共享目录。
- BOT_MODE=live 或 --live 当前被程序拒绝。不要据此尝试正式盘交易。
- 包含测试源码但本次打包没有运行测试套件；最近项目记录中的测试状态见策略复盘。
- 下载 ZIP 后，在 ZIP 所在目录运行 Get-FileHash .\alt-meanrev-bot-windows-20261023.zip -Algorithm SHA256，并与同目录的 .sha256 文件核对。

