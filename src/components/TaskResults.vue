<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import Icon from './Icon.vue'
import StemMixer from './StemMixer.vue'
import { cancelTask, type AudioTask } from '../services/tools'
import { clock } from '../lib/audio'
import { tools } from '../data/tools'
const props = defineProps<{task:AudioTask}>()
const emit = defineEmits<{preview:[url:string,name:string]}>()
const error = ref('')
watch(()=>props.task.id,()=>{error.value=''})
const metadata = computed(() => props.task.result?.metadata || {})
const chordEvents = computed(() => props.task.tool==='chords' ? metadata.value.chord_progression || metadata.value.progression || [] : [])
const notes = computed(() => metadata.value.notes || [])
const isAudio = (name:string) => /\.(wav|mp3|flac|m4a|aac|ogg)$/i.test(name)
const title = computed(() => tools.find(item=>item.id===props.task.tool)?.name || '音频编辑')
async function cancel() { try { await cancelTask(props.task) } catch(cause) {error.value=(cause as Error).message} }
</script>
<template>
  <section class="task-result-panel">
    <div class="section-title"><h2><Icon :name="task.status==='completed'?'success':task.status==='failed'?'help':'clock'" :size="18"/>{{ title }}<span>{{ task.name }}</span></h2><span class="job-status" :class="task.status">{{ {uploading:'接收中',queued:'排队中',running:'处理中',completed:'已完成',failed:'失败',cancelled:'已取消'}[task.status] }}</span></div>
    <div class="task-message"><span>{{ task.message }}</span><button v-if="task.cancelable" class="text-button" @click="cancel">取消任务</button></div>
    <div v-if="['queued','running'].includes(task.status)" class="job-progress"><i :style="{width:task.progress+'%'}"></i><span>{{ task.progress }}%</span></div>
    <p v-if="error" class="field-error">{{ error }}</p>
    <template v-if="task.status==='completed'&&task.result">
      <StemMixer v-if="task.tool==='stems'" :key="task.id" :files="task.result.files"/>
      <p v-for="warning in task.result.warnings||[]" :key="warning" class="result-warning">{{ warning }}</p>
      <div v-if="task.result.files.length" class="result-files"><article v-for="file in task.result.files" :key="file.url"><div class="result-file-title"><Icon :name="isAudio(file.name)?'audio':'file'" :size="17"/><strong>{{ file.name }}</strong><small>{{ (file.size/1024).toFixed(1) }} KB</small><a :href="file.url+'?download=1'" :download="file.name" class="icon-button" :aria-label="'下载 '+file.name"><Icon name="download" :size="17"/></a></div><audio v-if="isAudio(file.name)" :src="file.url" preload="none" controls/><button v-if="isAudio(file.name)" class="text-button" @click="emit('preview',file.url,file.name)">载入工作区试听<Icon name="arrow" :size="13"/></button></article></div>
      <div v-if="chordEvents.length" class="event-timeline"><h3>识别事件 · {{ chordEvents.length }}</h3><div><article v-for="(event,index) in chordEvents.slice(0,300)" :key="index"><strong>{{ event.chord||event.note_name||event.note||'N' }}</strong><span>{{ Number(event.start??event.start_sec??event.onset_sec??0).toFixed(2) }}–{{ Number(event.end??event.end_sec??0).toFixed(2) }}s</span></article></div></div>
      <div v-if="notes.length" class="midi-notes"><h3>音符预览 · {{ metadata.total_notes??notes.length }}</h3><div><span v-for="(note,index) in notes.slice(0,120)" :key="index">{{ note.note_name||note.name||note.midi_note||note.pitch }}<small>{{ Number(note.onset_sec??note.start??0).toFixed(2) }}s</small></span></div></div>
      <div v-if="metadata.instrument_analysis" class="event-timeline"><h3>乐器识别</h3><p v-if="!metadata.instrument_analysis.instruments?.length" class="form-help">当前音频未检测到足够的乐器证据。</p><p v-else class="form-help">以下为模型证据等级，不代表识别准确率；完整时间段和编曲角色见分析数据或 JSON 报告。</p><div><article v-for="(item,index) in metadata.instrument_analysis.instruments||[]" :key="index"><strong>{{item.label||item.instrument_id||item.name}}</strong><span>{{item.confidence==='high'?'强证据':item.confidence==='medium'?'较强证据':'可能'}}{{typeof item.score==='number'?' · '+item.score.toFixed(3):''}}</span></article></div></div>
      <details class="metadata-details"><summary>完整分析数据</summary><pre>{{ JSON.stringify(metadata,null,2) }}</pre></details>
    </template>
  </section>
</template>
