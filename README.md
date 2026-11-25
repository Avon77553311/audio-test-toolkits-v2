# Audio Toolkit Short

独立简化版，仅提供 **音轨分离、音频转 MIDI、和弦分析**。源码根目录为 `D:\tools\audio-toolkit-short`；完整版保持在原目录。

界面采用 2008 年前后的经典桌面软件风格：蓝色标题栏、菜单与工具条、树形工具箱、渐变按钮、双声道波形和凹陷时间显示。提供经典浅色、经典深色和跟随系统三种外观；外观选择会保存。

## 本机使用

当前机器已构建好前后端。双击 `Start-Local.cmd`，访问 **http://127.0.0.1:8081/**。局域网使用 `Start-LAN.cmd`，按终端打印的本机局域网地址访问，端口同为 8081；需要时以管理员身份运行 `Allow-LAN-Firewall.cmd`。端口可通过 `Start-Local.cmd --port 8082` 覆盖。

要求 Java 17+、Python 3.11、FFmpeg，以及 `worker/requirements.txt`、CPU PyTorch 和 Demucs。新机器可运行 `Setup-Tools.cmd` 安装。前端开发执行 `pnpm install --frozen-lockfile`、`pnpm dev`（5174 端口）；发布构建执行 `pnpm build` 和 `powershell -ExecutionPolicy Bypass -File scripts/Build-Server.ps1`。

当前本地 `.venv` 是独立环境，通过一个 `.pth` 文件复用完整版已安装的第三方 Python 包，未引用完整版应用源码、配置或任务。运行 `Setup-Tools.cmd` 会移除此捷径并在本目录安装完整依赖。复制到另一台机器必须重新安装依赖，不能直接复制 Windows 虚拟环境。

## 三个工具

- 音轨分离：Demucs 四轨/六轨，本地 CPU 执行；支持独立分轨下载、同步试听、静音/独奏、音量和混音导出。可选 EfficientAT 本地乐器识别。
- 音频转 MIDI：单音 pYIN、多音 CQT 估计、均衡/人声转音/快速吉他预设；提供参数控制、MIDI 下载、音符预览、合成音频试听。
- 和弦分析：DSP 均衡分析及完整五折神经模型；支持和声、低音、混合分析，输出时间轴和 JSON 报告。

本目录的 `runtime/models` 已复制完整五折和弦模型和 EfficientAT 权重，`runtime/torch/hub/checkpoints` 已复制 Demucs 权重。模型、原生二进制不进 Git 或源码包，授权和来源见 `worker/licenses`。算法共用的底层模块保留，其他工具的页面、任务注册和独立 API 已禁用。

## Windows Docker Desktop

使用 **Linux 容器**。本版复用原 `audio-toolkit:linux-amd64-20261023` 依赖镜像，将本目录构建产物和算法目录以只读方式挂载，修改应用后不必重做整个依赖镜像。新机器需先导入该基础镜像。

1. 将 `deploy/deploy.env.example` 复制到根目录 `deploy.env`，填写目标 IP。默认示例为 **127.0.0.1:9174**，与完整版 9173 分开。
2. 保留 `dist`、`server/audio-studio-server.jar`、`worker`、`runtime/models`、`runtime/torch/hub/checkpoints`，以及 `compose.yaml` 和启动脚本。
3. 双击 `Start-Docker.cmd`。脚本先做限时 Docker 引擎检查，再验证 Compose 和启动容器。
4. 查看日志：`docker compose --env-file deploy.env logs -f studio`；停止：`docker compose --env-file deploy.env down`。

Compose 项目为 `audio-toolkit-short`，数据卷为本项目的 `short-data`，不会使用完整版数据卷。升级时替换本版构建产物和源码后执行 `docker compose --env-file deploy.env up -d --force-recreate`。不要把简化版挂载或补丁应用到完整版容器。修改 IP/端口后重建本版容器即可。

## 安全与管理

普通局域网访客免登录，任务和文件按浏览器会话隔离；保留 Host/Origin、CSRF、任务所有权校验、排队与取消。管理员入口用于调度、审计和密码管理，初始化令牌见本版服务器生成的 `runtime/setup-token.txt`（以终端提示为准）。`.env.local.example` 为配置模板，无账号、密钥或任务从完整版迁入。付费能力默认关闭，普通访客不能调用；无需 OpenAI API。

## 验证与源码归档

`tests/short_integration.py` 验证真实 Java/Python 三工具处理、模型、私有下载、CSRF 和禁用工具。单元检查：`.venv\Scripts\python.exe -m pytest tests/test_short_api.py tests/test_chord_models.py tests/test_instrument_model.py`。

运行 `.venv\Scripts\python.exe scripts/package_source.py` 可生成带 SHA256 校验的源码包；不包含账号、密钥、任务、虚拟环境、模型、缓存和构建产物。
