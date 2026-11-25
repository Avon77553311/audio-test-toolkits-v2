"""
工具注册表 - 可扩展的工具管理系统
每个工具继承 BaseTool，实现 process 方法即可自动注册到平台
"""

import os

def _convert_with_tool(input_path, output_path, mono=True):
    """尝试用 ffmpeg 或 afconvert (macOS 内置) 将音频转为 WAV"""
    import subprocess
    # 检查 ffmpeg 是否安装
    ffmpeg_available = True
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
    except FileNotFoundError:
        ffmpeg_available = False
    except Exception:
        ffmpeg_available = False

    # 尝试 ffmpeg (跨平台)
    if ffmpeg_available:
        try:
            command = ['ffmpeg', '-y', '-i', input_path]
            if mono:
                command.extend(['-ac', '1'])
            command.extend(['-sample_fmt', 's16', output_path])
            subprocess.run(command, check=True, capture_output=True, timeout=120)
            return True
        except subprocess.CalledProcessError:
            pass

    # 尝试 afconvert (macOS 自带)
    try:
        command = [
            'afconvert', input_path, output_path,
            '-d', 'LEI16',   # 16-bit little-endian integer PCM
            '-f', 'WAVE',    # WAV container
        ]
        if mono:
            command.extend(['-c', '1'])
        subprocess.run(
            command, check=True, capture_output=True, timeout=120,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    if not ffmpeg_available:
        raise RuntimeError(
            "无法解析该音频格式，且 ffmpeg 未安装。请安装 ffmpeg: brew install ffmpeg (macOS) 或 apt install ffmpeg (Linux)"
        )
    raise RuntimeError(
        "无法解析该音频格式。请安装 ffmpeg: brew install ffmpeg"
    )


def load_audio(input_path, mono=True):
    """
    加载音频文件，降混为单声道。
    优先使用 soundfile；若格式不被支持 (如 .m4a/.aac)，
    自动回退到 ffmpeg / afconvert (macOS) 转换为临时 WAV 后再加载。

    Returns:
        (audio_data, sample_rate) — mono=True 返回一维数组，否则返回 [frames, channels]。
    """
    import numpy as np
    import soundfile as sf
    import tempfile
    try:
        audio_data, sample_rate = sf.read(input_path, always_2d=True)
    except Exception as e:
        # soundfile 无法解析 → 检查是否文件不存在
        if isinstance(e, FileNotFoundError):
            raise FileNotFoundError(f"音频文件不存在: {input_path}")
        # soundfile 无法解析 → 用 ffmpeg/afconvert 转换
        tmp_wav = tempfile.NamedTemporaryFile(
            suffix='.wav', delete=False, dir=os.path.dirname(input_path)
        )
        tmp_wav.close()
        try:
            _convert_with_tool(input_path, tmp_wav.name, mono=mono)
            audio_data, sample_rate = sf.read(tmp_wav.name, always_2d=True)
        finally:
            if os.path.exists(tmp_wav.name):
                os.remove(tmp_wav.name)

    if mono:
        if audio_data.shape[1] > 1:
            audio_data = audio_data.mean(axis=1)
        else:
            audio_data = audio_data[:, 0]
    return audio_data.astype(np.float64), sample_rate

class BaseTool:
    """工具基类 - 所有工具必须继承此类"""
    name = "base-tool"
    display_name = "基础工具"
    description = "基础工具描述"
    short_desc = "基础工具"
    accepted_extensions = []

    def process(self, input_path, output_dir, **kwargs):
        """
        处理输入文件，输出到指定目录

        Args:
            input_path: 输入文件路径
            output_dir: 输出目录路径
            **kwargs: 额外参数

        Returns:
            dict: {
                "success": bool,
                "output_files": [str],  # 输出文件路径列表
                "message": str,
                "metadata": dict        # 额外信息
            }
        """
        raise NotImplementedError("子类必须实现 process 方法")

    def get_info(self):
        """返回工具信息"""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "short_desc": getattr(self, 'short_desc', self.description),
            "accepted_extensions": self.accepted_extensions,
        }


class ToolRegistry:
    """工具注册表 - 管理所有已注册的工具"""

    def __init__(self):
        self._tools = {}

    def register(self, tool):
        """注册工具"""
        if not isinstance(tool, BaseTool):
            raise TypeError(f"工具必须继承 BaseTool，得到的是 {type(tool)}")
        self._tools[tool.name] = tool

    def get(self, name):
        """按名称获取工具"""
        return self._tools.get(name)

    def list_tools(self):
        """列出所有工具"""
        return [tool.get_info() for tool in self._tools.values()]

    def __contains__(self, name):
        return name in self._tools


# 全局注册表实例
registry = ToolRegistry()
