import { computed, ref } from 'vue'
import { encodeWav } from '../lib/audio'
import { csrfToken,toolUser,loadSession,submissionId } from './auth'

export type ResultFile = { name: string; size: number; url: string }
export type AudioTask = { id: string; tool: string; name: string; status: 'uploading'|'queued'|'running'|'completed'|'failed'|'cancelled'; progress: number; message: string; created: string; updated: string; cancelable: boolean; paid: boolean; result: { message?: string; files: ResultFile[]; metadata?: Record<string, any>; warnings?: string[] } | null }
export type Capabilities = { connected: boolean; tools: Record<string, {available: boolean; reason: string}>; features: Record<string, boolean>; speech_models?: Record<string, boolean> }
export const capabilities = ref<Capabilities>({connected:false, tools:{},features:{}})
export const serviceError = ref('')
export const tasks = ref<AudioTask[]>([])
export const submitting = ref(false)
let pollTimer: ReturnType<typeof setTimeout> | undefined
let stopped = false
export async function api<T = any>(url: string, options: RequestInit = {}): Promise<T> {
  const userId=toolUser.value?.id
  const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 95000)
  try {
    const response = await fetch(url, {...options, headers:{...(csrfToken.value?{'X-CSRF-Token':csrfToken.value}:{}),...options.headers}, signal:controller.signal, credentials:'same-origin'})
    const result = await response.json()
    if(response.status===401 && toolUser.value?.id===userId){tasks.value=[];await loadSession()}
    if (!response.ok || result.success === false) throw new Error(result.message || '处理请求失败')
    return result as T
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw new Error('本地服务响应超时，请在任务列表检查是否已提交。')
    throw error
  } finally { clearTimeout(timeout) }
}
export function upsertTask(task: AudioTask) { tasks.value = [task, ...tasks.value.filter(item => item.id !== task.id)].sort((a,b) => b.created.localeCompare(a.created)); schedulePolling() }
function schedulePolling() { clearTimeout(pollTimer); if (!stopped && tasks.value.some(task => ['queued','running'].includes(task.status))) pollTimer = setTimeout(refreshTasks, 1200) }
export async function refreshTasks() {
  const userId=toolUser.value?.id
  try { const result = await api('/api/tasks'); if(toolUser.value?.id===userId){tasks.value = result.tasks; serviceError.value = ''} }
  catch (error) { serviceError.value = (error as Error).message }
  finally { schedulePolling() }
}
export async function connectTools() {
  stopped = false
  try { capabilities.value = await api('/api/capabilities'); serviceError.value = ''; await refreshTasks() }
  catch (error) { capabilities.value.connected = false; serviceError.value = (error as Error).message }
}
export function stopPolling() { stopped = true; clearTimeout(pollTimer) }
export function toolAvailable(id: string) { return ['stems','midi','chords'].includes(id) && !!capabilities.value.tools[id]?.available }
export function currentTask(tool: string) { return computed(() => tasks.value.find(task => task.tool === tool)) }
export async function submitAudioTask(tool: string, options: Record<string, any>, source: File | null, buffer: AudioBuffer | null, name: string, reference?: File | null) {
  if (submitting.value) throw new Error('请等待当前上传完成')
  if (!source && !buffer) throw new Error('请先导入音频')
  submitting.value = true
  try {
    const form = new FormData(); form.append('tool',tool); form.append('options',JSON.stringify(options))
    const file = source || new File([encodeWav(buffer!,0,buffer!.duration,0,0)], name + '.wav', {type:'audio/wav'})
    form.append('file',file); if (reference) form.append('reference',reference)
    const result = await api('/api/tasks',{method:'POST',headers:{'X-Submission-ID':submissionId()},body:form}); upsertTask(result.task); return result.task as AudioTask
  } finally { submitting.value = false }
}
export async function cancelTask(task: AudioTask) { const result = await api(`/api/tasks/${task.id}`,{method:'DELETE'}); upsertTask(result.task) }
