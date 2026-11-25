import test from 'node:test'
import assert from 'node:assert/strict'
import { exportId } from '../src/lib/id.ts'

test('export history works on LAN HTTP without randomUUID',()=>{
  const original=Object.getOwnPropertyDescriptor(globalThis,'crypto')
  const random=globalThis.crypto.getRandomValues.bind(globalThis.crypto)
  Object.defineProperty(globalThis,'crypto',{configurable:true,value:{getRandomValues:random}})
  try {
    const ids=Array.from({length:100},()=>exportId())
    assert.equal(new Set(ids).size,100)
    assert.ok(ids.every(id=>/^[a-f0-9]{32}$/.test(id)))
  } finally { Object.defineProperty(globalThis,'crypto',original) }
})
