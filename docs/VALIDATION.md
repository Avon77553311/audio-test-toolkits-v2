# 简化版验证记录

验证对象：D:\tools\audio-toolkit-short，2026-11-25。

- Vue TypeScript 检查与 Vite 生产构建：通过。
- Java 17 服务端构建：通过。
- Python 三工具边界、模型路径及完整性单元测试：26 项通过。
- Java Host/Origin/局域网网络策略：21 项通过。
- 最终 Java → 私有 Python → 算法子进程端到端：通过。真实生成单音 MIDI、音符与合成试听；五折本地模型识别 C–F–G–Am；Demucs 六轨分离及 EfficientAT 乐器分析成功。结果文件及 ZIP 下载可用，其他浏览器会话被拒绝访问。
- 其他九工具的任务提交被拒绝，和弦工作室、钢琴、选区编辑接口返回 404。CSRF、Host、Origin、管理员入口保护检查通过。
- Windows Docker Desktop 的 Linux 容器：Compose 配置通过；复用基础依赖镜像，以只读目录挂载本版 dist、JAR 和 worker；实际 MIDI 转录和下载通过，完整和弦与乐器模型可用。
- 交付的 Start-Docker.ps1：在独立 audio-toolkit-short-qa 项目上实际执行通过。修复 Windows PowerShell 5.1 下异步进程退出码误判；Docker 检查有 20 秒超时。测试容器和网络已停止并移除，未使用完整版容器或数据卷。
- 源码 Python 语法、中文编码及 PowerShell 脚本语法：通过。

界面实现了 2008 年桌面音频软件风格、浅色/深色/跟随系统、三个工具切换、导入、波形、试听、任务与结果页面。自动截图检查未完成：当前聊天的内置浏览器不可用，Chrome 自动化连接也未成功取得页面。未将此项记为通过。可在 http://127.0.0.1:8081/ 查看实际效果。

端到端测试脚本：tests/short_integration.py。最终本机证据位于忽略目录 .local-tools/short-qa-4194b085；Docker MIDI 测试记录位于 .local-tools/docker-short-result.json。测试只使用合成音频，未调用付费服务。
