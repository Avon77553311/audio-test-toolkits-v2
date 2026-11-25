import { test } from 'node:test'
import assert from 'node:assert/strict'
import { encodeWav, clock, waveform } from '../src/lib/audio.ts'

function audio(channels, sampleRate = 100) {
  return { sampleRate, numberOfChannels: channels.length, length: channels[0].length, getChannelData: i => channels[i] }
}

test('WAV keeps sample rate, stereo interleaving and selected duration', async () => {
  const input = audio([Float32Array.from({ length: 1000 }, () => .5), Float32Array.from({ length: 1000 }, () => -.5)])
  const bytes = await encodeWav(input, 2, 5, 0, 0).arrayBuffer(), view = new DataView(bytes)
  assert.equal(new TextDecoder().decode(bytes.slice(0, 4)), 'RIFF')
  assert.equal(view.getUint16(22, true), 2)
  assert.equal(view.getUint32(24, true), 100)
  assert.equal(view.getUint16(34, true), 16)
  assert.equal(bytes.byteLength, 44 + 300 * 2 * 2)
  assert.equal(view.getInt16(44, true), 16384)
  assert.equal(view.getInt16(46, true), -16384)
})

test('export starts at requested source frame and clips samples', async () => {
  const values = new Float32Array(1000); values[200] = 2; values[201] = -2
  const view = new DataView(await encodeWav(audio([values]), 2, 3, 0, 0).arrayBuffer())
  assert.equal(view.getInt16(44, true), 32767)
  assert.equal(view.getInt16(46, true), -32768)
})

test('fade-in and fade-out silence the exact first and last sample', async () => {
  const view = new DataView(await encodeWav(audio([new Float32Array(1000).fill(.8)]), 0, 5, 1, 1).arrayBuffer())
  assert.equal(view.getInt16(44, true), 0)
  assert.equal(view.getInt16(view.byteLength - 2, true), 0)
  assert.ok(view.getInt16(44 + 250 * 2, true) > 26000)
})

test('empty and reversed ranges are rejected', () => {
  const input = audio([new Float32Array(1000)])
  assert.throws(() => encodeWav(input, 5, 2, 0, 0))
  assert.throws(() => encodeWav(input, 2, 2, 0, 0))
})

test('mono waveform can be rendered as the second display channel', () => {
  const result = waveform(audio([new Float32Array(1000).fill(.4)]), 1, 50)
  assert.equal(result.length, 50)
  assert.ok(result.every(value => Math.abs(value - .4) < .001))
})

test('clock handles duration, precision and invalid numbers', () => {
  assert.equal(clock(125), '02:05')
  assert.equal(clock(8, true), '00:08.00')
  assert.equal(clock(NaN), '00:00')
})
