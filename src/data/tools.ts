export type Tool = { id: string; name: string; en: string; icon: string; color: string; category: string; desc: string; tag?: string }
export const tools: Tool[] = [
  { id: 'stems', name: '音轨分离', en: 'STEM SEPARATION', icon: 'layers', color: 'mint', category: '处理与转换', desc: '拆解每一层声音，让人声与乐器各自闪耀。', tag: 'AI' },
  { id: 'midi', name: '音频转 MIDI', en: 'AUDIO TO MIDI', icon: 'piano', color: 'purple', category: '分析与转录', desc: '捕捉旋律中的每一个音符，延续你的创作。', tag: 'AI' },
  { id: 'chords', name: '和弦分析', en: 'CHORD ANALYSIS', icon: 'music', color: 'blue', category: '分析与转录', desc: '读懂和声、调性与低音，找到音乐的脉络。', tag: 'AI' },
]
