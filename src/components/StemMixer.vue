<script setup lang="ts">
import { computed, onBeforeUnmount, ref, shallowRef } from 'vue'
import type { ResultFile } from '../services/tools'
import { encodeWav, clock } from '../lib/audio'
const props=defineProps<{files:ResultFile[]}>()
const tracks=shallowRef<{name:string;buffer:AudioBuffer}[]>([])
const levels=ref<number[]>([]),muted=ref<boolean[]>([]),solo=ref<number|null>(null)
const context=shallowRef<AudioContext>(),playing=ref(false),busy=ref(false),error=ref(''),position=ref(0)
let sources:AudioBufferSourceNode[]=[],gains:GainNode[]=[],frame=0,started=0,offset=0,disposed=false
const duration=computed(()=>tracks.value[0]?.buffer.duration||0)
function level(index:number){return muted.value[index]||(solo.value!==null&&solo.value!==index)?0:levels.value[index]!/100}
function update(){gains.forEach((gain,index)=>gain.gain.setTargetAtTime(level(index),context.value!.currentTime,.01))}
function stop(){cancelAnimationFrame(frame);sources.forEach(source=>{source.onended=null;try{source.stop()}catch{}source.disconnect()});gains.forEach(gain=>gain.disconnect());sources=[];gains=[];playing.value=false}
async function load(){busy.value=true;error.value='';try{
  context.value ||=new AudioContext()
  const audioFiles=props.files.filter(file=>/^(vocals|drums|bass|other|piano|guitar)\.wav$/.test(file.name))
  const loaded:{name:string;buffer:AudioBuffer}[]=[]
  for(const file of audioFiles){const response=await fetch(file.url);if(!response.ok)throw new Error('音轨读取失败');loaded.push({name:file.name,buffer:await context.value.decodeAudioData(await response.arrayBuffer())})}
  if(disposed)return
  if(!loaded.length||loaded.some(track=>Math.abs(track.buffer.duration-loaded[0]!.buffer.duration)>.05))throw new Error('轨道时长不一致，无法同步混音')
  tracks.value=loaded;levels.value=loaded.map(()=>80);muted.value=loaded.map(()=>false)
}catch(cause){error.value=(cause as Error).message}finally{busy.value=false}}
function animate(){position.value=Math.max(0,Math.min(duration.value,offset+context.value!.currentTime-started));if(position.value>=duration.value){stop();position.value=0}else frame=requestAnimationFrame(animate)}
async function play(){if(playing.value){stop();return}if(!context.value||!tracks.value.length)return;await context.value.resume();if(disposed)return
  offset=position.value>=duration.value?0:position.value;started=context.value.currentTime
  tracks.value.forEach((track,index)=>{const source=context.value!.createBufferSource(),gain=context.value!.createGain();source.buffer=track.buffer;gain.gain.value=level(index);source.connect(gain);gain.connect(context.value!.destination);sources.push(source);gains.push(gain)})
  const when=context.value.currentTime+.05;started=when;sources.forEach(source=>source.start(when,offset));playing.value=true;animate()
}
function seek(){const wasPlaying=playing.value;stop();if(wasPlaying)play()}
function exportMix(){if(!tracks.value.length)return
  const first=tracks.value[0]!.buffer, mixed=new AudioBuffer({length:first.length,sampleRate:first.sampleRate,numberOfChannels:2})
  tracks.value.forEach((track,index)=>{const gain=level(index);for(let channel=0;channel<2;channel++){const input=track.buffer.getChannelData(Math.min(channel,track.buffer.numberOfChannels-1)),target=mixed.getChannelData(channel);for(let sample=0;sample<Math.min(input.length,target.length);sample++)target[sample]!+=input[sample]!*gain}})
  let peak=1;for(let channel=0;channel<2;channel++){const data=mixed.getChannelData(channel);for(const value of data)peak=Math.max(peak,Math.abs(value))}if(peak>1)for(let channel=0;channel<2;channel++){const data=mixed.getChannelData(channel);for(let i=0;i<data.length;i++)data[i]!/=peak}
  const url=URL.createObjectURL(encodeWav(mixed,0,mixed.duration,0,0)),link=document.createElement('a');link.href=url;link.download='stem_mix.wav';link.click();setTimeout(()=>URL.revokeObjectURL(url),10000)
}
onBeforeUnmount(()=>{disposed=true;stop();context.value?.close()})
</script>
<template><div class="stem-mixer"><h3>同步分轨混音</h3><button v-if="!tracks.length" class="button secondary" :disabled="busy" @click="load">{{busy?'正在载入音轨…':'载入音轨，同步试听与混音'}}</button><p v-if="error" role="alert" class="field-error">{{error}}</p><template v-if="tracks.length"><div class="mixer-transport"><button class="button secondary" @click="play">{{playing?'暂停':'同步播放'}}</button><span>{{clock(position)}} / {{clock(duration)}}</span><button class="button secondary" @click="exportMix">导出混音 WAV</button></div><input v-model.number="position" type="range" min="0" :max="duration" step=".01" aria-label="混音播放位置" @change="seek"/><div v-for="(track,index) in tracks" :key="track.name" class="mixer-track"><strong>{{track.name}}</strong><button class="text-button" :aria-pressed="muted[index]" @click="muted[index]=!muted[index];update()">{{muted[index]?'已静音':'静音'}}</button><button class="text-button" :aria-pressed="solo===index" @click="solo=solo===index?null:index;update()">{{solo===index?'正在独奏':'独奏'}}</button><input v-model.number="levels[index]" type="range" min="0" max="100" :aria-label="track.name+' 音量'" @input="update"/></div></template></div></template>
