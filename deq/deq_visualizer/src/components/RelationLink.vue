<script setup lang="ts">
import { inject } from 'vue'
import { VisualizerDataInjectionKey } from '@/misc/VisualizerData'
import { SelectionManagerInjectionKey } from '@/misc/SelectionManager'
import * as vis_pb from '@/proto/visualizer'

const props = defineProps<{
  target: vis_pb.Selectable
  label: string
}>()

const data = inject(VisualizerDataInjectionKey)!
const selectionManager = inject(SelectionManagerInjectionKey)!

function onMouseEnter() {
  // Highlight the target in the 3D scene and show tooltip at object location
  selectionManager.hover({ elements: [props.target] }, true)
}

function onMouseLeave() {
  // Clear hover
  selectionManager.hover({ elements: [] }, false)
}

function onClick(event: MouseEvent) {
  if (event.shiftKey) {
    // Shift+click: spawn a new info pane at cursor position
    data.spawnInfoPane(event.clientX, event.clientY, props.target)
  } else {
    // Regular click: select the element
    selectionManager.select({ elements: [props.target] })
  }
}
</script>

<template>
  <span 
    class="relation-link" 
    @mouseenter="onMouseEnter" 
    @mouseleave="onMouseLeave"
    @click="onClick"
  >
    {{ label }}
  </span>
</template>

<style scoped>
.relation-link {
  color: #6af;
  cursor: pointer;
  text-decoration: none;
  transition: color 0.15s ease;
}

.relation-link:hover {
  color: #9cf;
  text-decoration: underline;
}
</style>
