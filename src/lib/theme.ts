import { computed, onBeforeUnmount, ref, watch } from 'vue'

export type ThemePreference = 'dark' | 'light' | 'system'

export function useTheme() {
  const media = window.matchMedia('(prefers-color-scheme: dark)')
  const systemDark = ref(media.matches)
  const initial = document.documentElement.dataset.themePreference
  const preference = ref<ThemePreference>(initial === 'dark' || initial === 'system' ? initial : 'light')
  const resolved = computed(() => preference.value === 'system' ? (systemDark.value ? 'dark' : 'light') : preference.value)
  const onSystemChange = (event: MediaQueryListEvent) => { systemDark.value = event.matches }
  media.addEventListener('change', onSystemChange)
  watch([preference, resolved], () => {
    document.documentElement.dataset.theme = resolved.value
    document.documentElement.dataset.themePreference = preference.value
    document.documentElement.style.colorScheme = resolved.value
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', resolved.value === 'dark' ? '#252e37' : '#dce3eb')
    try { localStorage.setItem('audio-studio-theme', preference.value) } catch {}
  }, { immediate: true })
  onBeforeUnmount(() => media.removeEventListener('change', onSystemChange))
  return { preference, resolved }
}
