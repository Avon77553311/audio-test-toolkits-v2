<script setup lang="ts">
import { computed, reactive, ref, watch } from 'vue'
import Icon from './Icon.vue'
import { capabilities, connectTools, serviceError, submitting, submitAudioTask, tasks } from '../services/tools'
const props = defineProps<{toolId:string; buffer:AudioBuffer|null; sourceFile:File|null; fileName:string; selection:{start:number;end:number}}>()
const emit = defineEmits<{notice:[message:string]}>()
const options = reactive<Record<string,any>>({})
const reference = ref<File|null>(null), paidConsent = ref(false), error = ref('')
const defaults: Record<string,Record<string,any>> = {
  stems:{model:'htdemucs_6s',execution:'cpu'}, midi:{transcription_mode:'mono',transcription_profile:'balanced',quality_mode:'balanced',bpm:120,fmin:50,fmax:1000,min_note_duration_ms:80,activation_level:.21,time_window_ms:10,use_runpod:false},
  chords:{analysis_mode:'harmony',quality_mode:'balanced',confidence_threshold:.15,time_window_ms:500,min_chord_duration_ms:1100,use_runpod:false},
}
watch(() => props.toolId, id => { Object.keys(options).forEach(key => delete options[key]); Object.assign(options,defaults[id]||{}); error.value=''; reference.value=null; paidConsent.value=false },{immediate:true})
const paid = computed(() => options.execution === 'gpu' || options.use_runpod)
const available = computed(() => capabilities.value.connected && (options.execution === 'gpu' ? capabilities.value.features.cloud_stems : capabilities.value.tools[props.toolId]?.available))
const running = computed(() => tasks.value.some(task => task.tool === props.toolId && ['queued','running'].includes(task.status)))
async function submit() { error.value=''; try { await submitAudioTask(props.toolId,{...options,...(paid.value?{paid_consent:paidConsent.value}:{})},props.sourceFile,props.buffer,props.fileName,reference.value); emit('notice','任务已提交，处理结果会显示在工作区下方。') } catch(cause) { error.value=(cause as Error).message } }
watch(() => [options.quality_mode,options.analysis_mode], () => { if(['midi','chords'].includes(props.toolId) && (options.quality_mode!=='high' || options.analysis_mode==='bass'))options.use_runpod=false })
watch(() => options.execution, value => { if(value==='gpu'){options.instrument_analysis=false} })
</script>
<template>
  <div class="inspector-content tool-form">
    <span class="preview-inline">{{ available ? paid ? '已配置 · 云端付费' : '本地算法已连接' : '需要配置运行环境' }}</span>
    <template v-if="toolId==='stems'"><label class="form-label">分离模型</label><select v-model="options.model"><option value="htdemucs_6s">Demucs · 六轨分离</option><option value="htdemucs_ft">Demucs · 四轨分离</option></select><label class="form-label">执行方式</label><select v-model="options.execution"><option value="cpu">本地 CPU</option><option v-if="capabilities.features.cloud_stems" value="gpu">RunPod GPU · 付费</option></select><div class="stem-tags"><span v-for="name in options.model==='htdemucs_6s'?['人声','鼓','贝斯','吉他','钢琴','其他']:['人声','鼓','贝斯','其他']" :key="name">{{ name }}</span></div><p class="form-help">本地模型就绪后可离线处理。CPU 分轨需要较长时间，任务在后台执行，可在任务列表恢复查看。</p></template>
    <template v-if="toolId==='midi'"><label class="form-label">转录模式</label><select v-model="options.transcription_mode"><option value="mono">单音旋律 · pYIN</option><option value="poly">多音估计 · CQT</option></select><label class="form-label">演奏预设</label><select v-model="options.transcription_profile"><option value="balanced">均衡</option><option value="vocal_melisma">人声转音</option><option value="fast_guitar">快速吉他</option></select><label class="form-label">质量</label><select v-model="options.quality_mode"><option value="balanced">均衡</option><option value="high">多候选质量优先</option></select><label class="form-label">BPM</label><input v-model.number="options.bpm" type="number" min="1" max="999" aria-label="转录 BPM"/><details class="advanced-options"><summary>音域与识别参数</summary><label>最低频率 Hz<input v-model.number="options.fmin" type="number" min="8.18" max="12543.85"/></label><label>最高频率 Hz<input v-model.number="options.fmax" type="number" min="8.18" max="12543.85"/></label><label>最短音符 ms<input v-model.number="options.min_note_duration_ms" type="number" min="1" max="60000"/></label><label>置信度<input v-model.number="options.activation_level" type="number" min="0" max="1" step=".01"/></label><label>帧间隔 ms<input v-model.number="options.time_window_ms" type="number" min="1" max="500"/></label></details><p class="form-help">多音模式为估计结果。完成后可下载 MIDI、查看音符和试听合成预览。</p></template>
    <template v-if="toolId==='chords'"><label class="form-label">分析内容</label><select v-model="options.analysis_mode"><option value="harmony">和声 / 和弦</option><option value="bass">低音旋律</option><option value="mix">混合分析</option></select><label class="form-label">质量</label><select v-model="options.quality_mode"><option value="balanced">均衡 · DSP 本地分析</option><option v-if="capabilities.features.neural_chords || capabilities.features.cloud_chords" value="high">高质量 · 神经模型</option></select><label class="form-label">最短分段 {{ options.min_chord_duration_ms }} ms</label><input v-model.number="options.min_chord_duration_ms" type="range" :min="options.analysis_mode==='harmony'?500:50" max="5000" step="50"/><label class="form-label">置信度阈值</label><input v-model.number="options.confidence_threshold" type="number" min="0" max="1" step=".01"/><p class="form-help" v-if="!capabilities.features.neural_chords">尚未检测到可用的完整五折本地模型；当前可用 DSP 均衡分析，管理员安装模型补充包后开放本地高质量分析。</p></template>
    <label v-if="['midi','chords'].includes(toolId) && capabilities.features[toolId==='midi'?'cloud_stems':'cloud_chords'] && options.quality_mode==='high' && options.analysis_mode!=='bass'" class="check-label"><input v-model="options.use_runpod" type="checkbox"/>{{toolId==='midi'?'GPU 分轨辅助转录':'GPU 神经和弦分析'}} · 计入 GPU 配额 / 余额</label>
    <label v-if="toolId==='stems' && options.execution==='cpu' && capabilities.features.instrument_analysis" class="check-label"><input v-model="options.instrument_analysis" type="checkbox"/>分轨后识别乐器 · 本地模型</label>
    <label v-if="paid" class="check-label"><input v-model="paidConsent" type="checkbox"/>我确认本次会调用外部付费服务</label>
    <div v-if="!available" class="connection-note"><Icon name="help" :size="15"/><span>{{ serviceError || capabilities.tools[toolId]?.reason || '对应能力尚未配置' }}。可运行 Setup-Tools.cmd 后重启网站。</span></div>
    <p v-if="error" class="field-error" role="alert">{{ error }}</p>
  </div>
  <div class="inspector-footer"><button class="button primary" :disabled="!available||submitting||running||(paid&&!paidConsent)" @click="submit"><Icon :name="submitting?'loader':'play'" :size="16"/>{{ submitting?'正在上传…':running?'任务处理中':'开始处理' }}</button><button v-if="!available" class="text-button reconnect-button" @click="connectTools">重新检查连接</button><p>{{ paid?'需显式确认 · 可能产生服务商费用':'真实本地处理 · 结果保留在工作室服务器' }}</p></div>
</template>
