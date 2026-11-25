/** Normalized autocorrelation for the practice-room tuner; null means no stable pitch. */
export function detectPitch(samples: Float32Array, sampleRate: number): number | null {
  let rms = 0
  for (const value of samples) rms += value * value
  if (Math.sqrt(rms / samples.length) < .008) return null
  const minLag = Math.floor(sampleRate / 1400), maxLag = Math.min(Math.floor(sampleRate / 60), Math.floor(samples.length / 2))
  const correlations = new Float32Array(maxLag + 1)
  for (let lag = minLag; lag <= maxLag; lag++) {
    let sum = 0, left = 0, right = 0
    for (let i = 0; i < samples.length - maxLag; i++) { sum += samples[i] * samples[i + lag]; left += samples[i] ** 2; right += samples[i + lag] ** 2 }
    correlations[lag] = sum / Math.sqrt(left * right || 1)
  }
  let best = 0
  for (let lag = minLag + 1; lag < maxLag; lag++) {
    if (correlations[lag] > .88 && correlations[lag] > correlations[lag - 1] && correlations[lag] >= correlations[lag + 1]) { best = lag; break }
  }
  if (!best) return null
  const before = correlations[best - 1], center = correlations[best], after = correlations[best + 1]
  const correction = (before - after) / (2 * (before - 2 * center + after) || 1)
  return sampleRate / (best + correction)
}
