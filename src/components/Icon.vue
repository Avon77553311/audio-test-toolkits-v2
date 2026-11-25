<script setup lang="ts">
import { computed } from 'vue'
import { AudioLines, LayoutDashboard, Grid2X2, FolderClock, ChevronRight, ChevronDown, Upload, Search, Bell, Settings2, Plus, Scissors, Mic2, Piano, Music2, FileAudio, SlidersHorizontal, ArrowUpRight, Download, Play, Pause, SkipBack, Repeat2, Volume2, VolumeX, ZoomIn, ZoomOut, X, Check, CircleHelp, WandSparkles, Split, Headphones, Layers3, Clock3, MoreHorizontal, Keyboard, Command, Menu, ArrowLeft, CircleCheck, LoaderCircle, Trash2, Radio, Mail, ExternalLink, Sun, Moon, Monitor } from 'lucide-vue-next'
const props = withDefaults(defineProps<{ name: string; size?: number }>(), { size: 20 })
const icons: Record<string, any> = { audio: AudioLines, dashboard: LayoutDashboard, grid: Grid2X2, tasks: FolderClock, right: ChevronRight, down: ChevronDown, upload: Upload, search: Search, bell: Bell, settings: Settings2, plus: Plus, scissors: Scissors, mic: Mic2, piano: Piano, music: Music2, file: FileAudio, sliders: SlidersHorizontal, arrow: ArrowUpRight, download: Download, play: Play, pause: Pause, back: SkipBack, repeat: Repeat2, volume: Volume2, mute: VolumeX, zoomIn: ZoomIn, zoomOut: ZoomOut, close: X, check: Check, help: CircleHelp, wand: WandSparkles, split: Split, headphones: Headphones, layers: Layers3, clock: Clock3, more: MoreHorizontal, keyboard: Keyboard, command: Command, menu: Menu, left: ArrowLeft, success: CircleCheck, loader: LoaderCircle, trash: Trash2, radio: Radio, mail: Mail, external: ExternalLink }
icons.sun = Sun; icons.moon = Moon; icons.monitor = Monitor
const component = computed(() => icons[props.name] || AudioLines)
</script>
<template><component :is="component" :size="size" :stroke-width="1.65" aria-hidden="true" /></template>
