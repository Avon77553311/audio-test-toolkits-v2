<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue'
import Icon from './components/Icon.vue'
import ToolPanel from './components/ToolPanel.vue'
import TaskResults from './components/TaskResults.vue'
import AccountCenter from './components/AccountCenter.vue'
import { authUser, authLoading, toolUser, loadSession } from './services/auth'
import { capabilities, tasks, connectTools, stopPolling } from './services/tools'
import { tools } from './data/tools'
import { clock, makeDemo, waveform } from './lib/audio'
import { useTheme } from './lib/theme'
const { preference: theme } = useTheme()
const page = ref(location.pathname.replace(/\/+$/,'') === '/admin' ? 'admin' : 'workspace')
const selected = ref('stems'), fileInput = ref<HTMLInputElement>(), sourceFile = shallowRef<File|null>(null)
const buffer = shallowRef<AudioBuffer|null>(null), fileName = ref('演示音频.wav'), busy = ref(false), toast = ref(''), playing = ref(false), position = ref(0), volume = ref(75)
let context: AudioContext|undefined, player: AudioBufferSourceNode|undefined, gain: GainNode|undefined, offset = 0, started = 0, timer = 0, noticeTimer: ReturnType<typeof setTimeout>|undefined
const duration = computed(() => buffer.value?.duration || 0)
const current = computed(() => tools.find(t => t.id === selected.value)!)
const selection = computed(() => ({start:0,end:duration.value}))
const waves = computed(() => buffer.value ? [waveform(buffer.value,0,240),waveform(buffer.value,1,240)] : [])
const latest = computed(() => tasks.value.find(t => t.tool === selected.value))
const activeCount = computed(() => tasks.value.filter(t => ['queued','running','uploading'].includes(t.status)).length)
function audio() { return context ||= new AudioContext() }
function notice(message:string) { toast.value=message; clearTimeout(noticeTimer); noticeTimer=setTimeout(()=>toast.value='',6000) }
function pause() { if(player){player.onended=null;player.stop();player=undefined}; playing.value=false; cancelAnimationFrame(timer) }
function resetDemo() { pause();sourceFile.value=null;buffer.value=makeDemo(audio());fileName.value='演示音频.wav';position.value=0 }
async function openFile(file?:File) {
  if(!file)return
  if(file.size>100*1024*1024){notice('音频文件不能超过 100 MB');return}
  busy.value=true;pause();position.value=0;sourceFile.value=file;fileName.value=file.name;buffer.value=null
  try {buffer.value=await audio().decodeAudioData(await file.arrayBuffer());notice('音频已导入，选择工具后开始处理。')}catch{notice('浏览器无法预览此格式，仍可提交给服务器处理。')}finally{busy.value=false}
}
function choose(event:Event) { const input=event.target as HTMLInputElement;void openFile(input.files?.[0]);input.value='' }
function drop(event:DragEvent) {event.preventDefault();void openFile(event.dataTransfer?.files[0])}
async function play() {
  if(playing.value){pause();return};if(!buffer.value)return
  await audio().resume();offset=position.value>=duration.value?0:position.value;started=audio().currentTime
  player=audio().createBufferSource();gain=audio().createGain();gain.gain.value=volume.value/100
  player.buffer=buffer.value;player.connect(gain);gain.connect(audio().destination);player.onended=()=>{pause();position.value=0};player.start(0,offset);playing.value=true
  const tick=()=>{if(playing.value){position.value=Math.min(duration.value,offset+audio().currentTime-started);timer=requestAnimationFrame(tick)}};tick()
}
function seek(event:Event) {const resume=playing.value;pause();position.value=Number((event.target as HTMLInputElement).value);if(resume)void play()}
async function preview(url:string,name:string) {try{const r=await fetch(url);if(!r.ok)throw new Error('结果读取失败');await openFile(new File([await r.blob()],name));page.value='workspace'}catch(e){notice((e as Error).message)}}
watch(volume,value=>{if(gain)gain.gain.value=value/100})
watch(()=>toolUser.value?.id,(id,old)=>{if(id!==old){stopPolling();tasks.value=[];resetDemo()};if(id)void connectTools()})
onMounted(async()=>{resetDemo();try{await loadSession()}catch(e){notice((e as Error).message)}})
onBeforeUnmount(()=>{pause();stopPolling();clearTimeout(noticeTimer);void context?.close()})
</script>

<template>
  <div class="retro-app" @dragover.prevent @drop="drop">
    <header class="window-caption"><div><span class="app-emblem"><Icon name="audio" :size="20"/></span><strong>Audio Toolkit</strong><span>简化版 · 三合一音频工作站</span></div><span class="edition-label">CLASSIC EDITION</span></header>
    <nav class="menu-bar" aria-label="应用菜单"><button @click="fileInput?.click()">打开音频 (O)</button><button @click="page='workspace'">工作区 (W)</button><button @click="page='history'">任务记录 (T)</button><button @click="resetDemo();page='workspace'">载入示例 (D)</button><div class="menu-spacer"></div><label class="theme-select">外观 <select v-model="theme" aria-label="界面外观"><option value="light">经典浅色</option><option value="dark">经典深色</option><option value="system">跟随系统</option></select></label></nav>
    <div class="command-bar"><button class="button primary" @click="fileInput?.click()"><Icon name="upload" :size="17"/>打开音频</button><span class="toolbar-separator"></span><button class="button" :disabled="!buffer" @click="play"><Icon :name="playing?'pause':'play'" :size="16"/>{{playing?'暂停':'试听'}}</button><button class="button" @click="page='history'"><Icon name="tasks" :size="16"/>任务记录 <b>{{tasks.length}}</b></button><span class="toolbar-note">支持拖入音频文件 · 最大 100 MB</span><span class="connection-lamp" :class="{online:capabilities.connected}"></span><span class="connection-text">{{capabilities.connected?'处理引擎已连接':'正在连接引擎'}}</span></div>
    <div class="desktop-layout">
      <aside class="tool-browser"><h2><Icon name="grid" :size="15"/>工具箱</h2><p class="tree-heading">音频处理</p><button v-for="tool in tools" :key="tool.id" class="tree-tool" :class="{selected:selected===tool.id&&page==='workspace'}" @click="selected=tool.id;page='workspace'"><Icon :name="tool.icon" :size="21"/><span>{{tool.name}}<small>{{tool.en}}</small></span></button><div class="sidebar-info"><strong>使用步骤</strong><ol><li>打开或拖入音频</li><li>选择工具并设置参数</li><li>开始处理，下载结果</li></ol><p>局域网共享使用<br/>任务按浏览器会话隔离</p></div><button class="admin-link" @click="page='admin'"><Icon name="settings" :size="15"/>{{authUser?'管理员控制台':'管理员入口'}}</button></aside>
      <main class="desktop-content">
        <div v-if="authLoading" class="loading-box" role="status">正在初始化本地会话…</div>
        <AccountCenter v-else-if="page==='admin'" admin @changed="connectTools" @workspace="page='workspace'"/>
        <template v-else-if="page==='workspace'">
          <div class="workspace-heading"><div><span class="path-label">我的工作区 / {{current.name}}</span><h1>{{current.name}}</h1><p>{{current.desc}}</p></div><span class="version-plaque">AUDIO TOOLKIT <strong>SHORT</strong><small>LOCAL AUDIO WORKSTATION</small></span></div>
          <div class="tool-tabs" role="tablist" aria-label="音频工具"><button v-for="tool in tools" :id="'tab-'+tool.id" :key="tool.id" role="tab" :aria-selected="selected===tool.id" :class="{active:selected===tool.id}" @click="selected=tool.id"><Icon :name="tool.icon" :size="16"/>{{tool.name}}</button></div>
          <div class="workspace-grid">
            <section class="source-panel panel" :aria-label="'音频预览：'+fileName"><div class="panel-caption"><span><Icon name="audio" :size="16"/>源音频</span><button class="text-button" @click="fileInput?.click()">更换文件…</button></div><div class="source-name"><Icon name="file" :size="27"/><div><strong>{{fileName}}</strong><small>{{sourceFile?'已导入音频':'内置演示 · 可直接测试'}} · {{buffer?clock(duration)+' / '+buffer.sampleRate+' Hz / '+buffer.numberOfChannels+' 声道':'服务器解码'}}</small></div></div><div class="wave-screen"><div class="wave-ruler"><span v-for="i in 5" :key="i">{{clock(duration*(i-1)/4)}}</span></div><div v-if="waves.length" class="wave-tracks"><div v-for="(bars,index) in waves" :key="index" class="wave-track"><span class="channel-name">{{index?'R':'L'}}</span><svg viewBox="0 0 720 90" preserveAspectRatio="none" aria-hidden="true"><line x1="0" y1="45" x2="720" y2="45"/><rect v-for="(amplitude,i) in bars" :key="i" :x="i*3" :y="45-amplitude*41" width="2" :height="Math.max(1,amplitude*82)"/></svg></div><div class="playhead" :style="{left:(duration?position/duration*100:0)+'%'}"></div></div><div v-else class="wave-empty">{{busy?'正在读取波形…':'此格式将在服务器处理，预览暂不可用'}}</div></div><label class="seek-control">播放位置<input type="range" aria-label="播放位置" min="0" :max="duration||1" step=".01" :value="position" :disabled="!buffer" @input="seek"/></label><div class="transport"><button class="button" :disabled="!buffer" @click="pause();position=0"><Icon name="back" :size="17"/>归零</button><button class="button primary" :disabled="!buffer" @click="play"><Icon :name="playing?'pause':'play'" :size="17"/>{{playing?'暂停':'播放'}}</button><output class="time-display">{{clock(position,true)}} <small>/ {{clock(duration,true)}}</small></output><label class="volume-control"><Icon name="volume" :size="17"/><input v-model.number="volume" aria-label="播放音量" type="range" min="0" max="100"/></label></div><div class="source-footnote"><Icon name="help" :size="14"/>播放音量仅用于试听；处理任务使用完整源音频。</div><div class="workflow-note"><strong>当前工具：{{current.name}}</strong><p>在右侧设置参数并开始处理。任务可在后台继续，完成后在下方试听和下载。</p></div></section>
            <section class="inspector panel" role="tabpanel" :aria-labelledby="'tab-'+selected"><div class="panel-caption"><span><Icon name="sliders" :size="16"/>处理参数</span><small>{{current.name}}</small></div><ToolPanel :key="selected" :tool-id="selected" :buffer="buffer" :source-file="sourceFile" :file-name="fileName" :selection="selection" @notice="notice"/></section>
          </div>
          <TaskResults v-if="latest" :task="latest" @preview="preview"/>
          <section v-else class="result-placeholder panel"><Icon name="tasks" :size="25"/><div><strong>处理结果</strong><p>完成任务后，分轨试听、MIDI 音符或和弦时间轴会显示在这里。</p></div></section>
        </template>
        <template v-else><div class="workspace-heading"><div><span class="path-label">我的工作区 / 任务记录</span><h1>任务记录</h1><p>仅显示当前浏览器会话提交的任务。</p></div><button class="button" @click="page='workspace'">返回工作区</button></div><div v-if="!tasks.length" class="result-placeholder panel">暂无任务，打开音频即可开始。</div><TaskResults v-for="task in tasks" :key="task.id" :task="task" @preview="preview"/></template>
      </main>
    </div>
    <footer class="status-bar"><span><Icon name="success" :size="13"/>{{busy?'正在读取音频':'就绪'}}</span><span>{{activeCount}} 个后台任务</span><span class="status-right">Vue + Java + Python · 局域网版本</span></footer>
    <input ref="fileInput" class="visually-hidden" type="file" accept="audio/*,.mp3,.wav,.flac,.m4a,.ogg,.aac" aria-label="打开音频文件" @change="choose"/>
    <div v-if="toast" class="toast-message" role="status"><Icon name="help" :size="17"/>{{toast}}<button aria-label="关闭提示" @click="toast=''">×</button></div>
  </div>
</template>
