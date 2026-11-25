export function makeDemo(context: AudioContext): AudioBuffer {
  const rate = 22050, seconds = 32
  const buffer = context.createBuffer(2, rate * seconds, rate)
  const chords = [[130.81, 164.81, 196], [110, 130.81, 164.81], [87.31, 110, 130.81], [98, 123.47, 146.83]]
  for (let channel = 0; channel < 2; channel++) {
    const data = buffer.getChannelData(channel)
    for (let i = 0; i < data.length; i++) {
      const t = i / rate, beat = t % 0.5, bar = t % 4, chord = chords[Math.floor(t / 4) % 4]
      let sample = chord.reduce((a, f, j) => a + Math.sin(2 * Math.PI * f * (1 + channel * 0.0008) * t) * (0.055 + 0.014 * Math.sin(t * 0.8 + j)), 0)
      sample *= Math.min(1, bar * 4) * Math.exp(-bar * 0.2)
      sample += Math.sin(2 * Math.PI * (45 * beat + 11 * (1 - Math.exp(-beat * 25)))) * Math.exp(-beat * 22) * 0.27
      const noteTime = t % 0.25
      sample += Math.sin(2 * Math.PI * chord[Math.floor(t * 4) % 3] * 4 * t) * Math.exp(-noteTime * 22) * 0.055
      sample += Math.sin(i * 1.789 + Math.sin(i * 0.123) * 23) * Math.exp(-noteTime * 90) * 0.025
      data[i] = sample * Math.min(1, t / 1.2, (seconds - t) / 1.8)
    }
  }
  return buffer
}

export function waveform(buffer: AudioBuffer, channel: number, count = 300): number[] {
  const data = buffer.getChannelData(Math.min(channel, buffer.numberOfChannels - 1))
  return Array.from({ length: count }, (_, index) => {
    const start = Math.floor(index * data.length / count), end = Math.floor((index + 1) * data.length / count)
    let peak = 0
    for (let j = start; j < end; j += Math.max(1, Math.floor((end - start) / 80))) peak = Math.max(peak, Math.abs(data[j]))
    return Math.max(0.015, peak)
  })
}

export function encodeWav(buffer: AudioBuffer, start: number, end: number, fadeIn: number, fadeOut: number): Blob {
  const rate = buffer.sampleRate, channels = buffer.numberOfChannels
  const first = Math.floor(start * rate), last = Math.min(buffer.length, Math.floor(end * rate)), frames = last - first
  if (frames <= 0) throw new Error('请先选择有效的音频区间')
  const bytes = frames * channels * 2, data = new DataView(new ArrayBuffer(44 + bytes))
  const str = (offset: number, value: string) => { for (let i = 0; i < value.length; i++) data.setUint8(offset + i, value.charCodeAt(i)) }
  str(0, 'RIFF'); data.setUint32(4, 36 + bytes, true); str(8, 'WAVE'); str(12, 'fmt ')
  data.setUint32(16, 16, true); data.setUint16(20, 1, true); data.setUint16(22, channels, true)
  data.setUint32(24, rate, true); data.setUint32(28, rate * channels * 2, true); data.setUint16(32, channels * 2, true); data.setUint16(34, 16, true)
  str(36, 'data'); data.setUint32(40, bytes, true)
  const inputs = Array.from({ length: channels }, (_, i) => buffer.getChannelData(i))
  for (let i = 0; i < frames; i++) {
    const gain = Math.min(1, fadeIn > 0 ? i / (fadeIn * rate) : 1, fadeOut > 0 ? (frames - i - 1) / (fadeOut * rate) : 1)
    for (let c = 0; c < channels; c++) {
      const sample = Math.max(-1, Math.min(1, inputs[c][first + i] * gain))
      data.setInt16(44 + (i * channels + c) * 2, Math.round(sample < 0 ? sample * 32768 : sample * 32767), true)
    }
  }
  return new Blob([data], { type: 'audio/wav' })
}

export function clock(seconds: number, precise = false): string {
  if (!Number.isFinite(seconds)) return '00:00'
  const minutes = Math.floor(seconds / 60).toString().padStart(2, '0')
  const rest = Math.floor(seconds % 60).toString().padStart(2, '0')
  return `${minutes}:${rest}${precise ? '.' + Math.floor(seconds % 1 * 100).toString().padStart(2, '0') : ''}`
}
