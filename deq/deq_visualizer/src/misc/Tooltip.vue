<script lang="ts" setup>
import { ref, watch } from 'vue'

export interface Props {
  element: HTMLElement
  text: string | undefined
  // Optional override position (relative to element). When set, tooltip shows at this position instead of mouse
  overrideX?: number
  overrideY?: number
}
const props = withDefaults(defineProps<Props>(), {
  overrideX: undefined,
  overrideY: undefined,
})

const clientX = ref(0)
const clientY = ref(0)

props.element.addEventListener('mousemove', (event) => {
  // Only update from mouse if no override is set
  if (props.overrideX === undefined || props.overrideY === undefined) {
    const rect = props.element.getBoundingClientRect()
    clientX.value = event.clientX - rect.left
    clientY.value = event.clientY - rect.top
  }
})

// Watch for override position changes
watch(
  () => [props.overrideX, props.overrideY],
  ([ox, oy]) => {
    if (ox !== undefined && oy !== undefined) {
      clientX.value = ox
      clientY.value = oy
    }
  },
  { immediate: true }
)
</script>

<template>
  <div v-if="text" :style="{ top: clientY - 30 + 'px', left: clientX + 'px' }">{{ text }}</div>
</template>

<style lang="css" scoped>
div {
  position: absolute;
  background-color: white;
  border: 1px solid black;
  padding: 5px;
  border-radius: 3px;
  box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
  pointer-events: none;
  font-size: 12px;
  z-index: 1000;
}
</style>
