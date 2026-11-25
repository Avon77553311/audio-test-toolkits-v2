import { computed,ref } from 'vue'
export type StudioUser={id:string;login:string;name:string;role:'admin'|'user'|'guest'}
export const authUser=ref<StudioUser|null>(null), csrfToken=ref(''), authLoading=ref(true), setupRequired=ref(false), googleAvailable=ref(false)
export const visitorUser=ref<StudioUser|null>(null),accessMode=ref<'lan-public'|'accounts'>('lan-public')
export const publicLan=computed(()=>accessMode.value==='lan-public'),toolUser=computed(()=>authUser.value||visitorUser.value)
export function submissionId(){const bytes=crypto.getRandomValues(new Uint8Array(16));bytes[6]=(bytes[6]!&15)|64;bytes[8]=(bytes[8]!&63)|128;const h=Array.from(bytes,b=>b.toString(16).padStart(2,'0')).join('');return `${h.slice(0,8)}-${h.slice(8,12)}-${h.slice(12,16)}-${h.slice(16,20)}-${h.slice(20)}`}
export async function authRequest(path:string,options:RequestInit={}) {
  const response=await fetch(path,{...options,credentials:'same-origin',headers:{...(options.body?{'Content-Type':'application/json'}:{}),...(csrfToken.value?{'X-CSRF-Token':csrfToken.value}:{}),...options.headers}})
  const data=await response.json()
  if(response.status===401 && !path.startsWith('/api/auth/'))await loadSession()
  if(!response.ok||data.success===false)throw new Error(data.message||'账户请求失败')
  return data
}
export async function loadSession(){authLoading.value=true;try{const data=await authRequest('/api/me');accessMode.value=data.accessMode||'accounts';csrfToken.value=data.csrf;visitorUser.value=data.visitor||null;authUser.value=data.user||null;setupRequired.value=data.setupRequired;googleAvailable.value=data.googleAvailable}finally{authLoading.value=false}}
export async function signIn(login:string,password:string,setupToken?:string){const data=await authRequest(setupRequired.value?'/api/auth/bootstrap':'/api/auth/login',{method:'POST',body:JSON.stringify({login,password,setupToken})});authUser.value=data.user;csrfToken.value=data.csrf;setupRequired.value=false}
export async function signOut(){await authRequest('/api/auth/logout',{method:'POST'});await loadSession()}
