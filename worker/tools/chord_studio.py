"""
和弦工作室 (Chord Studio)

功能:
1. 完整和弦库 (名称 + 音名, e.g. Cmaj9 = C E G B D)
2. 完整音阶库 (自然大调, Lydian 等)
3. 和弦进行模板 (流行 1645, 爵士 251 等)
4. 自由组合和弦 (选调 → 选级数 → 选和弦类型 → 播放 → 导出 MIDI)

依赖: numpy, soundfile, mido
"""

import functools
import os
from pathlib import Path

# ─── 常量 ───

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MIDI_C4 = 60  # C4 (middle C) 的 MIDI 音符号
A4_FREQ = 440.0
A4_MIDI = 69
MIN_TEMPO_BPM = 20
MAX_TEMPO_BPM = 300

# ─── 和弦公式 (从根音的半音间隔) ───


def _chord(abbr, intervals):
    """建立以标准和弦符号显示的和弦定义。"""
    return {
        'display': f"X{abbr}" if abbr else "X",
        'intervals': intervals,
        'abbr': abbr,
    }


CHORD_FORMULAS = {
    # 三和弦
    'maj':     _chord('', [0, 4, 7]),
    'm':       _chord('m', [0, 3, 7]),
    'dim':     _chord('dim', [0, 3, 6]),
    'aug':     _chord('aug', [0, 4, 8]),
    'sus2':    _chord('sus2', [0, 2, 7]),
    'sus4':    _chord('sus4', [0, 5, 7]),
    '5':       _chord('5', [0, 7]),
    # 七和弦
    '7':       _chord('7', [0, 4, 7, 10]),
    'maj7':    _chord('maj7', [0, 4, 7, 11]),
    'm7':      _chord('m7', [0, 3, 7, 10]),
    'mmaj7':   _chord('mMaj7', [0, 3, 7, 11]),
    'dim7':    _chord('dim7', [0, 3, 6, 9]),
    'm7b5':    _chord('m7b5', [0, 3, 6, 10]),
    'aug7':    _chord('7#5', [0, 4, 8, 10]),
    'maj7#5':  _chord('maj7#5', [0, 4, 8, 11]),
    '7sus4':   _chord('7sus4', [0, 5, 7, 10]),
    '7sus2':   _chord('7sus2', [0, 2, 7, 10]),
    '7b5':     _chord('7b5', [0, 4, 6, 10]),
    # 六和弦
    '6':       _chord('6', [0, 4, 7, 9]),
    'm6':      _chord('m6', [0, 3, 7, 9]),
    '69':      _chord('6/9', [0, 4, 7, 9, 14]),
    # 九和弦
    '9':       _chord('9', [0, 4, 7, 10, 14]),
    'maj9':    _chord('maj9', [0, 4, 7, 11, 14]),
    'm9':      _chord('m9', [0, 3, 7, 10, 14]),
    'mmaj9':   _chord('mMaj9', [0, 3, 7, 11, 14]),
    '9sus4':   _chord('9sus4', [0, 5, 7, 10, 14]),
    '7#9':     _chord('7#9', [0, 4, 7, 10, 15]),
    '7b9':     _chord('7b9', [0, 4, 7, 10, 13]),
    'add9':    _chord('add9', [0, 4, 7, 14]),
    'madd9':   _chord('madd9', [0, 3, 7, 14]),
    'add4':    _chord('add4', [0, 4, 5, 7]),
    # 十一和弦
    '11':      _chord('11', [0, 4, 7, 10, 14, 17]),
    'maj11':   _chord('maj11', [0, 4, 7, 11, 14, 17]),
    'm11':     _chord('m11', [0, 3, 7, 10, 14, 17]),
    '7#11':    _chord('7#11', [0, 4, 7, 10, 18]),
    # 十三和弦
    '13':      _chord('13', [0, 4, 7, 10, 14, 17, 21]),
    'maj13':   _chord('maj13', [0, 4, 7, 11, 14, 17, 21]),
    'm13':     _chord('m13', [0, 3, 7, 10, 14, 17, 21]),
    '13b9':    _chord('13b9', [0, 4, 7, 10, 13, 21]),
    # 变音与扩展和弦
    'm69':     _chord('m6/9', [0, 3, 7, 9, 14]),
    'maj7b5':  _chord('maj7b5', [0, 4, 6, 11]),
    'dimMaj7': _chord('dimMaj7', [0, 3, 6, 11]),
    '7alt':    _chord('7alt', [0, 4, 6, 10, 13, 20]),
    '7b5b9':   _chord('7b5b9', [0, 4, 6, 10, 13]),
    '7#5#9':   _chord('7#5#9', [0, 4, 8, 10, 15]),
    # 非标准音阶按三度堆叠时的安全展示类型
    'other':   _chord('(stack)', [0, 4, 7]),
    'other7':  _chord('7(stack)', [0, 4, 7, 10]),
    # 单音 (用于音阶播放)
    'single':  {'display': 'Note', 'intervals': [0], 'abbr': ''},
}

BROWSABLE_CHORD_TYPES = tuple(
    chord_type
    for chord_type in CHORD_FORMULAS
    if chord_type not in {'other', 'other7', 'single'}
)

# ─── 音阶公式 (从根音的半音间隔) ───

def _scale(display, pitch_classes, descending_pitch_classes=None, description=''):
    """建立音阶定义；遍历公式包含首尾主音，且允许方向相关的合法例外。"""
    descending = descending_pitch_classes or list(reversed(pitch_classes))
    return {
        'display': display,
        # intervals 保留为和弦级数推导所用的上行音高集合（不含八度主音）。
        'intervals': pitch_classes,
        'ascending_intervals': pitch_classes + [12],
        'descending_intervals': [12] + descending,
        'description': description,
    }


SCALE_FORMULAS = {
    'major':          _scale('自然大调', [0, 2, 4, 5, 7, 9, 11]),
    'natural_minor':  _scale('自然小调', [0, 2, 3, 5, 7, 8, 10]),
    'harmonic_minor': _scale('和声小调', [0, 2, 3, 5, 7, 8, 11]),
    'melodic_minor':  _scale(
        '旋律小调（古典）',
        [0, 2, 3, 5, 7, 9, 11],
        [10, 8, 7, 5, 3, 2, 0],
        '上行升高第 6、7 级；下行还原为自然小调。',
    ),
    'ionian':         _scale('伊奥尼亚', [0, 2, 4, 5, 7, 9, 11]),
    'dorian':         _scale('多利亚', [0, 2, 3, 5, 7, 9, 10]),
    'phrygian':       _scale('弗利吉亚', [0, 1, 3, 5, 7, 8, 10]),
    'lydian':         _scale('利底亚', [0, 2, 4, 6, 7, 9, 11]),
    'mixolydian':     _scale('混合利底亚', [0, 2, 4, 5, 7, 9, 10]),
    'aeolian':        _scale('爱奥利亚', [0, 2, 3, 5, 7, 8, 10]),
    'locrian':        _scale('洛克利亚', [0, 1, 3, 5, 6, 8, 10]),
    'major_pent':     _scale('大调五声', [0, 2, 4, 7, 9]),
    'minor_pent':     _scale('小调五声', [0, 3, 5, 7, 10]),
    'blues':          _scale('蓝调音阶', [0, 3, 5, 6, 7, 10]),
    'whole_tone':     _scale('全音音阶', [0, 2, 4, 6, 8, 10]),
    'diminished':     _scale(
        '减音阶（全音-半音）', [0, 2, 3, 5, 6, 8, 9, 11],
        description='采用 whole-half octatonic（全音—半音交替）公式。',
    ),
    'diminished_half_whole': _scale(
        '减音阶（半音-全音）', [0, 1, 3, 4, 6, 7, 9, 10],
        description='采用 half-whole octatonic（半音—全音交替）公式。',
    ),
}


def get_scale_traversal_intervals(scale_type):
    """返回音阶实际播放所需的上、下行绝对半音序列。"""
    scale = SCALE_FORMULAS[scale_type]
    return scale['ascending_intervals'][:], scale['descending_intervals'][:]

# ─── 和弦进行模板 ───

PROGRESSION_TEMPLATES = {
    'pop': [
        {'id': 'pop_axis', 'name': 'I–V–vi–IV', 'degrees': [1, 5, 6, 4], 'scale': 'major', 'seventh': False, 'description': '开阔、稳定的四和弦循环，常用于副歌', 'tags': ['大调', '4和弦', '循环', '基础']},
        {'id': 'pop_singer_songwriter', 'name': 'vi–IV–I–V', 'degrees': [6, 4, 1, 5], 'scale': 'major', 'seventh': False, 'description': '从相对小调起步的抒情循环', 'tags': ['小调中心', '4和弦', '主歌/副歌', '基础']},
        {'id': 'pop_50s', 'name': 'I–vi–IV–V', 'degrees': [1, 6, 4, 5], 'scale': 'major', 'seventh': False, 'description': '50s / doo-wop 回转，适合复古流行', 'tags': ['大调', '4和弦', '段尾回转', '基础']},
        {'id': 'pop_hopscotch', 'name': 'IV–V–vi–I', 'degrees': [4, 5, 6, 1], 'scale': 'major', 'seventh': False, 'description': '具有持续上扬感的现代流行循环', 'tags': ['大调', '4和弦', '循环', '基础']},
    ],
    'rock': [
        {'id': 'rock_three_chord', 'name': 'I–IV–V–IV', 'degrees': [1, 4, 5, 4], 'scale': 'major', 'seventh': False, 'description': '经典三和弦摇滚与 riff 骨架', 'tags': ['大调', '3和弦', '循环', '基础']},
        {'id': 'rock_mixolydian', 'name': 'I–♭VII–IV–I', 'degrees': [1, 7, 4, 1], 'scale': 'mixolydian', 'seventh': False, 'description': '带 ♭VII 的 Mixolydian 开放式进行', 'tags': ['调式', '借用和弦', 'riff', '进阶']},
        {'id': 'rock_aeolian', 'name': 'i–♭VII–♭VI–♭VII', 'degrees': [1, 7, 6, 7], 'scale': 'natural_minor', 'seventh': False, 'description': '重型与另类摇滚常用的小调循环', 'tags': ['小调', '4和弦', '副歌', '基础']},
        {'id': 'rock_pachelbel_desc', 'name': 'I–V–vi–iii–IV–I–IV–V', 'degrees': [1, 5, 6, 3, 4, 1, 4, 5], 'scale': 'major', 'seventh': False, 'description': '下行低音感的长线条叙事进行', 'tags': ['大调', '8和弦', '叙事', '进阶']},
    ],
    'folk': [
        {'id': 'folk_primary', 'name': 'I–IV–I–V', 'degrees': [1, 4, 1, 5], 'scale': 'major', 'seventh': False, 'description': '传统民谣与叙事主歌的主和弦骨架', 'tags': ['大调', '开放和弦', '主歌', '基础']},
        {'id': 'folk_pop', 'name': 'I–V–vi–IV', 'degrees': [1, 5, 6, 4], 'scale': 'major', 'seventh': False, 'description': '适合木吉他与 singer-songwriter 的循环', 'tags': ['大调', '4和弦', '循环', '基础']},
        {'id': 'folk_minor_descent', 'name': 'i–♭VII–♭VI–V', 'degrees': [1, 7, 6, 5], 'scale': 'harmonic_minor', 'seventh': False, 'description': '暗色叙事与小调段落收束', 'tags': ['小调', '下行低音', '终止', '进阶']},
        {'id': 'folk_plagal', 'name': 'I–IV–I', 'degrees': [1, 4, 1], 'scale': 'major', 'seventh': False, 'description': '简洁、温暖的变格回应', 'tags': ['大调', '3和弦', '副歌', '基础']},
    ],
    'jazz': [
        {'id': 'jazz_major_251', 'name': 'ii7–V7–Imaj7', 'degrees': [2, 5, 1], 'scale': 'major', 'seventh': True, 'description': '爵士标准曲最核心的大调终止', 'tags': ['大调', '七和弦', '终止', '进阶']},
        {'id': 'jazz_minor_251', 'name': 'iiø7–V7–i', 'degrees': [2, 5, 1], 'scale': 'harmonic_minor', 'seventh': True, 'description': '小调 ii–V–i，适合暗色终止', 'tags': ['小调', '半减七', '终止', '高级']},
        {'id': 'jazz_1625_turnaround', 'name': 'Imaj7–VI7–ii7–V7', 'degrees': [1, 6, 2, 5], 'scale': 'major', 'seventh': True, 'description': '段尾返回开头的经典 turnaround', 'tags': ['大调', '七和弦', '段尾回转', '进阶']},
        {'id': 'jazz_circle', 'name': 'iii7–VI7–ii7–V7–Imaj7', 'degrees': [3, 6, 2, 5, 1], 'scale': 'major', 'seventh': True, 'description': '五度循环延长终止感', 'tags': ['大调', '五度循环', '终止', '高级']},
    ],
    'rnb': [
        {'id': 'rnb_251_color', 'name': 'ii9–V13–Imaj9', 'degrees': [2, 5, 1], 'scale': 'major', 'seventh': True, 'description': '以扩展音色彩演奏的慢歌终止骨架', 'tags': ['大调', '扩展音', '终止', '进阶']},
        {'id': 'rnb_minor_251', 'name': 'iiø7–V7♭9–i9', 'degrees': [2, 5, 1], 'scale': 'harmonic_minor', 'seventh': True, 'description': '暗色 R&B / Neo-Soul 回环', 'tags': ['小调', '扩展音', '终止', '高级']},
        {'id': 'rnb_1645', 'name': 'Imaj7–vi7–IVmaj7–V7sus', 'degrees': [1, 6, 4, 5], 'scale': 'major', 'seventh': True, 'description': '复古与当代 Soul ballad 骨架', 'tags': ['大调', '七和弦', '抒情', '进阶']},
        {'id': 'neo_minor_vamp', 'name': 'i9–IV13', 'degrees': [1, 4], 'scale': 'dorian', 'seventh': True, 'description': 'Dorian 色彩的慢速 pocket', 'tags': ['调式', '2和弦', '即兴Vamp', '进阶']},
    ],
    'blues': [
        {'id': 'blues_12_standard', 'name': '标准 12-Bar Blues', 'degrees': [1, 1, 1, 1, 4, 4, 1, 1, 5, 4, 1, 5], 'scale': 'mixolydian', 'seventh': True, 'description': '标准 12 小节；末格 V7 便于继续循环', 'tags': ['12小节', '属七和弦', '循环', '基础']},
        {'id': 'blues_12_quick_change', 'name': 'Quick Change 12-Bar', 'degrees': [1, 4, 1, 1, 4, 4, 1, 1, 5, 4, 1, 5], 'scale': 'mixolydian', 'seventh': True, 'description': '第 2 小节提前进入 IV7 的经典变化', 'tags': ['12小节', '属七和弦', 'Quick Change', '进阶']},
        {'id': 'blues_8_bar', 'name': '8-Bar Blues', 'degrees': [1, 5, 4, 4, 1, 5, 1, 5], 'scale': 'mixolydian', 'seventh': True, 'description': '紧凑的 8 小节蓝调循环', 'tags': ['8小节', '属七和弦', '循环', '基础']},
    ],
    'funk': [
        {'id': 'funk_one_chord', 'name': 'I9 Vamp', 'degrees': [1], 'scale': 'mixolydian', 'seventh': True, 'description': '以节奏与切分为核心的单和弦 groove', 'tags': ['调式', '1和弦', '即兴Vamp', '基础']},
        {'id': 'funk_dorian_vamp', 'name': 'i7–IV7', 'degrees': [1, 4], 'scale': 'dorian', 'seventh': True, 'description': 'Dorian pocket 两和弦循环', 'tags': ['调式', '2和弦', '即兴Vamp', '进阶']},
        {'id': 'funk_blues_vamp', 'name': 'I9–IV9', 'degrees': [1, 4], 'scale': 'mixolydian', 'seventh': True, 'description': '适合吉他与铜管 stab 的 blues-funk', 'tags': ['属和弦', '2和弦', '节奏优先', '进阶']},
    ],
    'soul': [
        {'id': 'soul_plagal', 'name': 'I7–IV7–I7', 'degrees': [1, 4, 1], 'scale': 'mixolydian', 'seventh': True, 'description': 'Gospel / Soul 式呼应与 riff', 'tags': ['属七和弦', '3和弦', '呼应', '进阶']},
        {'id': 'soul_1645', 'name': 'Imaj7–vi7–IVmaj7–V7', 'degrees': [1, 6, 4, 5], 'scale': 'major', 'seventh': True, 'description': '抒情 Soul 与复古 R&B 循环', 'tags': ['大调', '七和弦', '抒情', '进阶']},
        {'id': 'soul_minor_cadence', 'name': 'iv7–V7–i7', 'degrees': [4, 5, 1], 'scale': 'harmonic_minor', 'seventh': True, 'description': '小调情绪高潮与收束', 'tags': ['小调', '七和弦', '终止', '进阶']},
    ],
    'city_pop': [
        {'id': 'citypop_royal_road', 'name': 'IVmaj7–V7–iii7–vi7', 'degrees': [4, 5, 3, 6], 'scale': 'major', 'seventh': True, 'description': '王道进行：抬升情绪并避免直接落 I', 'tags': ['大调', '七和弦', '副歌', '进阶']},
        {'id': 'citypop_2516', 'name': 'ii7–V7–Imaj7–VI7', 'degrees': [2, 5, 1, 6], 'scale': 'major', 'seventh': True, 'description': '循环结尾与进入下一段的回转', 'tags': ['大调', '七和弦', '段尾回转', '高级']},
        {'id': 'citypop_circle', 'name': 'IVmaj7–VII7–iii7–VI7–ii7–V7–Imaj7', 'degrees': [4, 7, 3, 6, 2, 5, 1], 'scale': 'major', 'seventh': True, 'description': '适合 verse / pre-chorus 的长线条五度循环', 'tags': ['大调', '五度循环', '次属和弦', '高级']},
    ],
}

TEMPLATE_QUALITY_OVERRIDES = {
    'folk_minor_descent': ['m', 'maj', 'maj', 'maj'],
    'jazz_minor_251': ['m7b5', '7', 'm'],
    'jazz_1625_turnaround': ['maj7', '7', 'm7', '7'],
    'jazz_circle': ['m7', '7', 'm7', '7', 'maj7'],
    'rnb_251_color': ['m9', '13', 'maj9'],
    'rnb_minor_251': ['m7b5', '7', 'm9'],
    'rnb_1645': ['maj7', 'm7', 'maj7', '7sus4'],
    'neo_minor_vamp': ['m9', '13'],
    'blues_12_standard': ['7'] * 12,
    'blues_12_quick_change': ['7'] * 12,
    'blues_8_bar': ['7'] * 8,
    'funk_one_chord': ['9'],
    'funk_dorian_vamp': ['m7', '7'],
    'funk_blues_vamp': ['9', '9'],
    'soul_plagal': ['7', '7', '7'],
    'soul_1645': ['maj7', 'm7', 'maj7', '7'],
    'soul_minor_cadence': ['m7', '7', 'm7'],
    'citypop_royal_road': ['maj7', '7', 'm7', 'm7'],
    'citypop_2516': ['m7', '7', 'maj7', '7'],
    'citypop_circle': ['maj7', '7', 'm7', '7', 'm7', '7', 'maj7'],
}

TEMPLATE_ROOT_OFFSET_OVERRIDES = {
    'folk_minor_descent': [0, 10, 8, 7],
}


# ─── 核心函数: 音名 ↔ MIDI ───

def note_to_index(note_name):
    """音名 → 半音索引 (0-11)"""
    return NOTE_NAMES.index(note_name)


def midi_to_freq(midi_note):
    """MIDI 音符号 → 频率 (Hz)"""
    return A4_FREQ * (2.0 ** ((midi_note - A4_MIDI) / 12.0))


def validate_tempo(tempo):
    """将 BPM 转为浮点数，并限制在可用且合理的范围内。"""
    import numpy as np
    try:
        tempo = float(tempo)
    except (TypeError, ValueError) as exc:
        raise ValueError("tempo 必须是数字") from exc
    if not np.isfinite(tempo) or not MIN_TEMPO_BPM <= tempo <= MAX_TEMPO_BPM:
        raise ValueError(
            f"tempo 必须在 {MIN_TEMPO_BPM}-{MAX_TEMPO_BPM} BPM 之间"
        )
    return tempo


def get_chord_notes(root, chord_type):
    """
    获取和弦音名列表

    Args:
        root: 根音名 ('C', 'D#' 等)
        chord_type: 和弦类型键 ('maj', 'm7', 'maj9' 等)

    Returns:
        list[str]: 音名列表, e.g. ['C', 'E', 'G', 'B', 'D'] for Cmaj9
    """
    root_idx = note_to_index(root)
    intervals = CHORD_FORMULAS[chord_type]['intervals']
    return [NOTE_NAMES[(root_idx + iv) % 12] for iv in intervals]


def get_chord_midi_notes(root, chord_type, octave=4):
    """
    获取和弦 MIDI 音符号列表

    Args:
        root: 根音名
        chord_type: 和弦类型键
        octave: 根音所在八度 (4 = C4 中音区)

    Returns:
        list[int]: MIDI 音符号列表
    """
    root_idx = note_to_index(root)
    root_midi = MIDI_C4 + (octave - 4) * 12 + root_idx
    intervals = CHORD_FORMULAS[chord_type]['intervals']
    return [root_midi + iv for iv in intervals]


def get_chord_name(root, chord_type):
    """获取和弦显示名 (e.g. 'C', 'Cm', 'Cmaj9')"""
    abbr = CHORD_FORMULAS[chord_type]['abbr']
    return f"{root}{abbr}"


def get_scale_notes(root, scale_type):
    """
    获取音阶音名列表

    Args:
        root: 调性根音名
        scale_type: 音阶类型键

    Returns:
        list[str]: 音名列表, e.g. ['C', 'D', 'E', 'F#', 'G', 'A', 'B'] for C Lydian
    """
    root_idx = note_to_index(root)
    intervals = SCALE_FORMULAS[scale_type]['intervals']
    return [NOTE_NAMES[(root_idx + iv) % 12] for iv in intervals]


def _classify_triad(root_pitch, third_pitch, fifth_pitch):
    """根据音程判断三和弦类型"""
    i1 = (third_pitch - root_pitch) % 12
    i2 = (fifth_pitch - third_pitch) % 12
    if i1 == 4 and i2 == 3:
        return 'maj'
    elif i1 == 3 and i2 == 4:
        return 'm'
    elif i1 == 3 and i2 == 3:
        return 'dim'
    elif i1 == 4 and i2 == 4:
        return 'aug'
    elif i1 == 5 and i2 == 2:
        return 'sus4'
    elif i1 == 2 and i2 == 5:
        return 'sus2'
    # 五声音阶、全音音阶等并不总能形成标准三和弦；返回安全类型供 API 展示。
    return 'other'


def _classify_seventh(root_pitch, third_pitch, fifth_pitch, seventh_pitch):
    """根据音程判断七和弦类型"""
    triad = _classify_triad(root_pitch, third_pitch, fifth_pitch)
    seventh_interval = (seventh_pitch - root_pitch) % 12
    if triad == 'maj' and seventh_interval == 11:
        return 'maj7'
    elif triad == 'maj' and seventh_interval == 10:
        return '7'
    elif triad == 'm' and seventh_interval == 10:
        return 'm7'
    elif triad == 'dim' and seventh_interval == 10:
        return 'm7b5'
    elif triad == 'dim' and seventh_interval == 9:
        return 'dim7'
    elif triad == 'm' and seventh_interval == 11:
        return 'mmaj7'
    elif triad == 'aug' and seventh_interval == 10:
        return 'aug7'
    elif triad == 'aug' and seventh_interval == 11:
        return 'maj7#5'
    elif triad == 'other':
        return 'other7'
    return '7'


def get_scale_degree_info(root, scale_type, degree):
    """
    获取指定音阶中某级数的和弦信息

    通过堆叠三度音 (1st, 3rd, 5th, 7th) 自动推导和弦类型

    Args:
        root: 调性根音名
        scale_type: 音阶类型键
        degree: 级数 (1-7, 1-indexed)

    Returns:
        dict: {
            'degree': int,
            'root_note': str,        # 该级数根音名
            'triad_type': str,       # 三和弦类型键 (在 CHORD_FORMULAS 中)
            'triad_notes': list,     # 三和弦音名
            'seventh_type': str,    # 七和弦类型键
            'seventh_notes': list,  # 七和弦音名
        }
    """
    root_idx = note_to_index(root)
    intervals = SCALE_FORMULAS[scale_type]['intervals']
    n = len(intervals)
    if not isinstance(degree, int) or not 1 <= degree <= n:
        raise ValueError(f"degree 必须是 1-{n} 之间的整数")

    # 构建跨三个八度的音阶 pitch 列表
    scale_pitches = []
    for octave in range(3):
        for i in range(n):
            scale_pitches.append(root_idx + intervals[i] + octave * 12)

    d = degree - 1  # 0-indexed

    # 三和弦: 1st, 3rd, 5th (跳过每相邻一个音)
    triad_pitches = [scale_pitches[d], scale_pitches[d + 2], scale_pitches[d + 4]]
    triad_notes = [NOTE_NAMES[p % 12] for p in triad_pitches]
    triad_type = _classify_triad(*triad_pitches)

    # 七和弦: 1st, 3rd, 5th, 7th
    seventh_pitches = triad_pitches + [scale_pitches[d + 6]]
    seventh_notes = [NOTE_NAMES[p % 12] for p in seventh_pitches]
    seventh_type = _classify_seventh(*seventh_pitches)

    return {
        'degree': degree,
        'root_note': triad_notes[0],
        'triad_type': triad_type,
        'triad_notes': triad_notes,
        'seventh_type': seventh_type,
        'seventh_notes': seventh_notes,
    }


def build_template_progression(template_id, root='C', octave=4):
    """Resolve one offline template into the same chord records used for audio/MIDI."""
    template = next(
        (
            item
            for templates in PROGRESSION_TEMPLATES.values()
            for item in templates
            if item['id'] == template_id
        ),
        None,
    )
    if template is None:
        raise ValueError(f"未知和弦进行模板: {template_id}")
    qualities = TEMPLATE_QUALITY_OVERRIDES.get(template_id)
    root_offsets = TEMPLATE_ROOT_OFFSET_OVERRIDES.get(template_id)
    if qualities is not None and len(qualities) != len(template['degrees']):
        raise ValueError(f"和弦进行模板配置损坏: {template_id}")
    if root_offsets is not None and len(root_offsets) != len(template['degrees']):
        raise ValueError(f"和弦进行模板根音配置损坏: {template_id}")

    progression = []
    for index, degree in enumerate(template['degrees']):
        info = get_scale_degree_info(root, template['scale'], degree)
        root_note = (
            NOTE_NAMES[(note_to_index(root) + root_offsets[index]) % 12]
            if root_offsets is not None
            else info['root_note']
        )
        chord_type = (
            qualities[index]
            if qualities is not None
            else info['seventh_type'] if template['seventh'] else info['triad_type']
        )
        if chord_type not in CHORD_FORMULAS:
            raise ValueError(f"和弦进行模板包含未知和弦类型: {chord_type}")
        progression.append(
            {
                'root': root_note,
                'type': chord_type,
                'octave': int(octave),
                'name': get_chord_name(root_note, chord_type),
            }
        )
    return progression


# ─── 音频合成 ───

def _generate_tone(freq, duration_sec, sample_rate=44100):
    """
    生成钢琴音色 (基频 + 谐波 + ADSR 包络)

    使用正弦波叠加 4 次谐波模拟钢琴音色, 配合 ADSR 包络
    """
    import numpy as np
    n_samples = int(sample_rate * duration_sec)
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    # 基频 + 谐波 (模拟钢琴泛音列)
    wave = np.sin(2 * np.pi * freq * t)
    wave += 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    wave += 0.15 * np.sin(2 * np.pi * 3 * freq * t)
    wave += 0.08 * np.sin(2 * np.pi * 4 * freq * t)

    # 归一化
    max_amp = np.max(np.abs(wave))
    if max_amp > 0:
        wave = wave / max_amp * 0.5

    # ADSR 包络
    attack_samples = min(int(sample_rate * 0.005), n_samples)    # 5ms
    decay_samples = min(int(sample_rate * 0.1), n_samples - attack_samples)  # 100ms
    release_samples = min(int(sample_rate * 0.05), max(1, n_samples - attack_samples - decay_samples))  # 50ms, 不超过剩余样本
    sustain_level = 0.6

    envelope = np.ones(n_samples, dtype=np.float64) * sustain_level
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    if attack_samples + decay_samples < n_samples:
        envelope[attack_samples:attack_samples + decay_samples] = \
            np.linspace(1, sustain_level, decay_samples)
    # 始终应用 release, 防止短音结尾产生 "啪" 声
    if release_samples > 0 and release_samples < n_samples:
        envelope[-release_samples:] = np.linspace(sustain_level, 0, release_samples)

    return (wave * envelope).astype(np.float32)


DEFAULT_ARPEGGIO_INTERVAL_SEC = 0.25
ARPEGGIO_TAIL_SEC = 0.35


def _arpeggio_timing(note_count, duration_sec, interval_sec=None):
    """返回可完整容纳在片段内的起音间隔、单音时长和总时长。"""
    if note_count <= 0:
        return 0.0, 0.0, 0.0
    duration_sec = max(float(duration_sec), 0.001)
    requested_interval = interval_sec or DEFAULT_ARPEGGIO_INTERVAL_SEC
    requested_interval = max(float(requested_interval), 0.001)
    if note_count == 1:
        return 0.0, duration_sec, duration_sec
    tail = min(ARPEGGIO_TAIL_SEC, duration_sec)
    interval = min(
        requested_interval,
        max(0.0, duration_sec - tail) / (note_count - 1),
    )
    last_onset = interval * (note_count - 1)
    tone_duration = duration_sec - last_onset
    return interval, tone_duration, duration_sec


def _mode_sections(mode, duration_sec):
    if mode == "both":
        half = duration_sec / 2
        return [("arpeggiated", half), ("block", half)]
    if mode in {"arpeggiated", "block"}:
        return [(mode, duration_sec)]
    raise ValueError("演奏模式仅支持 arpeggiated、block 或 both")


def _note_schedule(note_count, duration_sec, mode, interval_sec=None):
    if note_count <= 0:
        return []
    if mode == "block":
        return [(0.0, duration_sec)] * note_count
    interval, tone_duration, _output_duration = _arpeggio_timing(
        note_count,
        duration_sec,
        interval_sec,
    )
    return [
        (index * interval, tone_duration)
        for index in range(note_count)
    ]


def _synthesize_chord(midi_notes, duration_sec, mode, sample_rate=44100, note_duration_sec=None):
    """合成和弦；分解模式的相邻起音间隔不随和弦音数变化。"""
    import numpy as np
    if not midi_notes:
        return np.zeros(0, dtype=np.float32)

    output = np.zeros(int(sample_rate * duration_sec), dtype=np.float32)
    schedule = _note_schedule(
        len(midi_notes),
        duration_sec,
        mode,
        note_duration_sec,
    )
    for note, (onset, tone_duration) in zip(midi_notes, schedule):
        tone = _generate_tone(midi_to_freq(note), tone_duration, sample_rate)
        start = int(onset * sample_rate)
        end = min(start + len(tone), len(output))
        output[start:end] += tone[:end - start]

    max_val = np.max(np.abs(output)) if len(output) else 0
    if max_val > 0.9:
        output = output / max_val * 0.9
    return output


def synthesize_progression(chord_midi_list, mode='both', tempo=120, sample_rate=44100, note_duration_sec=None):
    """
    合成和弦进行音频

    Args:
        chord_midi_list: list[list[int]], 每个内层列表是一个和弦的 MIDI 音符号
        mode: 'arpeggiated' | 'block' | 'both' (分解 + 柱式)
        tempo: BPM
        sample_rate: 采样率
        note_duration_sec: 分解模式下相邻音符的固定起音间隔 (秒)

    Returns:
        (np.ndarray, int): (音频信号, 采样率)
    """
    import numpy as np
    tempo = validate_tempo(tempo)
    beat_duration = 60.0 / tempo
    chord_beats = 2  # 每个和弦占 2 拍

    parts = []
    for midi_notes in chord_midi_list:
        if not midi_notes:
            continue

        duration = beat_duration * chord_beats
        parts.append(
            np.concatenate(
                [
                    _synthesize_chord(
                        midi_notes,
                        section_duration,
                        section_mode,
                        sample_rate,
                        note_duration_sec=note_duration_sec,
                    )
                    for section_mode, section_duration in _mode_sections(
                        mode,
                        duration,
                    )
                ]
            )
        )

    if not parts:
        return np.zeros(0, dtype=np.float32), sample_rate

    audio = np.concatenate(parts)

    # 最终安全归一化
    max_val = np.max(np.abs(audio))
    if max_val > 0.95:
        audio = audio / max_val * 0.95

    return audio.astype(np.float32), sample_rate


# ─── MIDI 导出 ───

def export_midi(
    chord_midi_list,
    output_path,
    mode='both',
    tempo=120,
    note_duration_sec=None,
):
    """
    导出和弦进行为 MIDI 文件

    Args:
        chord_midi_list: list[list[int]], 每个内层列表是一个和弦的 MIDI 音符号
        output_path: 输出文件路径
        mode: 'arpeggiated' | 'block' | 'both'
        tempo: BPM
    """
    import mido
    from mido import Message, MetaMessage, MidiFile, MidiTrack
    tempo = validate_tempo(tempo)
    mid = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))
    track.append(Message('program_change', program=0))  # Acoustic Grand Piano

    ticks_per_beat = 480
    chord_beats = 2

    seconds_per_beat = 60.0 / tempo
    chord_duration = seconds_per_beat * chord_beats
    events = []
    chord_start = 0
    for midi_notes in chord_midi_list:
        if not midi_notes:
            continue

        section_start = 0.0
        for section_mode, section_duration in _mode_sections(
            mode,
            chord_duration,
        ):
            schedule = _note_schedule(
                len(midi_notes),
                section_duration,
                section_mode,
                note_duration_sec,
            )
            for note, (onset, note_duration) in zip(midi_notes, schedule):
                start_tick = chord_start + round(
                    (section_start + onset)
                    / seconds_per_beat
                    * ticks_per_beat
                )
                end_tick = chord_start + round(
                    (section_start + onset + note_duration)
                    / seconds_per_beat
                    * ticks_per_beat
                )
                events.append((start_tick, 1, note))
                events.append((max(start_tick + 1, end_tick), 0, note))
            section_start += section_duration
        chord_start += ticks_per_beat * chord_beats

    cursor = 0
    for tick, is_on, note in sorted(
        events,
        key=lambda event: (event[0], event[1]),
    ):
        track.append(
            Message(
                "note_on" if is_on else "note_off",
                note=note,
                velocity=80 if is_on else 0,
                time=max(0, tick - cursor),
            )
        )
        cursor = tick

    target = Path(output_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        mid.save(temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


# ─── 结构化数据导出 ───

@functools.lru_cache(maxsize=1)
def get_studio_data():
    """返回和弦工作室的全部静态数据。"""
    return {
        'notes': NOTE_NAMES,
        'chord_types': [
            {'key': k, 'display': v['display'], 'abbr': v['abbr'],
             'intervals': v['intervals'],
             'example': f"C{v['abbr']} = {' '.join(get_chord_notes('C', k))}"}
            for k, v in CHORD_FORMULAS.items()
            if k in BROWSABLE_CHORD_TYPES
        ],
        'scale_types': [
            {
                'key': k,
                'display': v['display'],
                'description': v['description'],
                'intervals': v['intervals'],
                'ascending_intervals': v['ascending_intervals'],
                'descending_intervals': v['descending_intervals'],
                'example': f"C {v['display']} = {' '.join(get_scale_notes('C', k))}",
            }
            for k, v in SCALE_FORMULAS.items()
        ],
        'progressions': {
            genre: [
                {key: p[key] for key in (
                    'id', 'name', 'degrees', 'scale', 'seventh',
                    'description', 'tags'
                )}
                for p in items
            ]
            for genre, items in PROGRESSION_TEMPLATES.items()
        },
    }
