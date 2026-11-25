# 简化版范围与架构

根目录：D:\tools\audio-toolkit-short。

浏览器 Vue 3 → Java 17 同源 API、访客会话与资源队列 → 私有 Python Flask worker → 每任务独立算法子进程。Python 只在 loopback 监听，使用 Java 随机生成的内部令牌。

允许的工具 ID 精确为 stems、midi、chords。Python 参数校验和执行器均使用白名单；Java 不再提供 /api/studio/*、/api/piano/*、/api/tasks/{id}/edit。其他九个工具没有页面入口、任务注册或可执行 API。

UI 为经典桌面工作站，保留文件导入、双声道波形、播放定位、音量、三工具参数、后台任务、下载、分轨混音、乐器证据、MIDI 预览、和弦时间轴与管理调度。只处理完整源音频，不提供裁剪或选区编辑。

业务数据仅写入本版 runtime。模型复制到本版 runtime/models，Demucs 检查点在 runtime/torch。开发用 Python 第三方依赖目前通过 .pth 复用，安装脚本可切换为完全独立的依赖环境。Docker 通过独立 Compose 项目和数据卷运行，挂载本版代码，不更新旧容器。

本地 Java 默认 8081、Vite 5174；Docker 默认 9174。保留深浅主题、局域网免登录使用、管理员权限、CSRF、来源与任务文件所有权校验。云端付费引擎默认关闭。
