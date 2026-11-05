# GitHub 同步与跨电脑恢复

私有仓库保存机器人源码、策略配置、依赖锁文件、安装脚本、测试与维护文档。首个提交建立当前源码基线；本地尚无 Git 历史，因此无法从该提交恢复更早的源码版本。此后每次提交都可作为回滚点。

仓库不会保存 `.env`、API 密钥、Windows Credential Manager 内容、`.venv`、运行账本、交易所状态、行情缓存、日志、临时文件或发布 ZIP。GitHub 同步的是机器人程序，不是模拟账户状态。提交前仍要检查 `git status`，不要用 `git add -f` 强行加入忽略文件。

## 在另一台 Windows 电脑恢复

安装 Git，并安装 Python 3.11 或 3.12。然后克隆仓库并安装锁定依赖：

```powershell
git clone https://github.com/eadison419-stack/alt-meanrev-bot.git
Set-Location .\alt-meanrev-bot
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

检查源码和配置：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

8766 使用 USDⓈ-M Demo。每台 Windows 电脑都需要在本机的 Windows Credential Manager 单独录入 Demo 专用密钥：

```powershell
.\scripts\setup_demo_credentials.ps1
```

录入后，用项目现有的看护脚本启动 Demo Runner；需要时另行启动读取同一账本的 Dashboard。安装程序不会自动启动 Runner、Dashboard 或看护器。不要在两台电脑上同时管理同一个 Demo 账户。仓库不包含账本；启动已有 Demo 账户前，必须先核对该账户仓位、订单和本机账本。

## 日常同步与回滚

开始修改前先同步远端，再创建分支：

```powershell
git switch main
git pull --ff-only
git switch -c fix/short-description
```

完成修改并验证后，提交并推送分支：

```powershell
git status --short
git add <本次修改的文件>
git commit -m "fix: 简要说明"
git push -u origin fix/short-description
```

合并前检查变更，合并后再从 `main` 同步。若某次修改需要撤销，先用 `git log --oneline` 找到对应提交，再用 `git revert <提交号>` 创建反向提交并推送；这样会保留同步历史，也可在之后恢复被撤销的提交。
