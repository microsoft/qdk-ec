<script setup lang="ts">
import { computed, onMounted, reactive, ref, type Reactive, useTemplateRef, watchEffect, provide, type ShallowRef, shallowRef, watch, onBeforeUnmount } from 'vue'
import SharedRenderer from '@/misc/SharedRenderer.ts' // optimization: share a single WebGL renderer across all the instances
import { PerspectiveCamera as ThreePerspectiveCamera, OrthographicCamera as ThreeOrthographicCamera, Vector3 } from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { PerspectiveCamera, Scene, AmbientLight } from 'troisjs'
import SelectionManager from '@/misc/SelectionManager'
import * as pb2 from '@/proto/deq_bin'
import * as vis_pb2 from '@/proto/visualizer'
import { VisualizerData, VisualizerDataInjectionKey } from '@/misc/VisualizerData'
// @ts-expect-error the Stats module does not have a declaration file
import Stats from 'troisjs/src/components/misc/Stats'
import { assert, jsonValueRef, VisualizerInjectionKey, type PartialDisplayMode } from './util'
import Blocks from '@/components/Blocks.vue'
import Realization from '@/components/Realization.vue'
import CheckModel from '@/components/CheckModel.vue'
import ErrorModel from '@/components/ErrorModel.vue'
import ConfigPane from '@/components/ConfigPane.vue'
import FloatingInfoPane from '@/components/FloatingInfoPane.vue'

// props are for one-directional data flow (parent to child)
// use `defineModel` for bi-directional data flow (parent <-> child)
export interface Props {
  width?: ShallowRef<string>
  height?: ShallowRef<string>
  // the source code of the esm module
  esm?: ShallowRef<string>
  library: ShallowRef<Uint8Array>
  cameraType?: string // 'perspective' | 'orthographic'
  gateStyle?: string // 'top' | 'front'
  background?: string // CSS color for scene background (e.g. '#f0f0f0', 'transparent')
}
const props = withDefaults(defineProps<Props>(), {
  width: () => ref('100%'),
  height: () => ref('100%'),
  cameraType: 'perspective',
  gateStyle: 'top',
  background: '#f0f0f0',
})

// 'transparent' → null (Three.js renders transparent when scene.background is null)
const sceneBackground = computed(() =>
  props.background === 'transparent' ? null : props.background
)

assert(props.library != undefined, 'missing library')
const data = new VisualizerData(pb2.Library.fromBinary(props.library.value))
provide(VisualizerDataInjectionKey, data)

// models (bi-directional props)
const selectedBytes = defineModel<ShallowRef<Uint8Array>>('selected', {
  default: () => shallowRef(new Uint8Array()),
})
const hoveredBytes = defineModel<ShallowRef<Uint8Array>>('hovered', {
  default: () => shallowRef(new Uint8Array()),
})
const cameraPosition = defineModel<Reactive<Vector3>>('cameraPosition', {
  default: () => reactive(new Vector3(0, 0, 10)),
})
const orbitTarget = defineModel<Reactive<Vector3>>('orbitTarget', {
  default: () => reactive(new Vector3(0, 0, 0)),
})
const displayModePython = defineModel<Reactive<{ [gid: string]: PartialDisplayMode }>>('displayMode', {
  default: () => reactive({}),
})

provide(VisualizerInjectionKey, {
  cameraPosition,
  orbitTarget,
  dragCamera,
  setGlobalDisplayMode,
  setDisplayMode,
})

const containerRef = useTemplateRef('container')
const rendererRef = useTemplateRef('renderer')
const cameraRef = useTemplateRef('camera')
const statsRef = useTemplateRef('stats')

// Reference FOV in radians, matching three.js default PerspectiveCamera FOV (50°).
// Used to compute orthographic frustum size from camera-target distance.
const REFERENCE_FOV_RAD = (50 * Math.PI) / 180

const aspectRatio = ref(1.0)
let lastClientWidth = 0
let lastClientHeight = 0
function onResize() {
  const container = containerRef.value!
  const canvas = (rendererRef.value as any)?.canvas as HTMLCanvasElement
  if (canvas == null) {
    return
  }
  if (container.clientWidth != lastClientWidth || container.clientHeight != lastClientHeight) {
    if (container.clientWidth != 0 && container.clientHeight != 0) {
      canvas.width = lastClientWidth = container.clientWidth
      canvas.height = lastClientHeight = container.clientHeight
      aspectRatio.value = container.clientWidth / container.clientHeight
    }
  }
}
let resizeInterval: number | undefined = undefined

function isSaving(event: KeyboardEvent): boolean {
  return (event.ctrlKey && event.key === 's') || (event.metaKey && event.key === 's')
}

const multiSelecting = ref(false)

function onKeyDown(event: KeyboardEvent) {
  multiSelecting.value = event.ctrlKey
  if (isSaving(event)) {
    // config.value.download_html()
    event.preventDefault()
    event.stopPropagation()
    return
  }
  // Handle Escape key to close focused info pane
  if (event.key === 'Escape') {
    data.closeFocusedPane()
    event.preventDefault()
    event.stopPropagation()
    return
  }
  // console.log('key down:', event.key)
  if (!event.metaKey && !event.ctrlKey && !event.altKey) {
    if (event.key === 'a' || event.key === 'ArrowLeft') {
      dragCamera(10, 0)
    } else if (event.key === 'd' || event.key === 'ArrowRight') {
      dragCamera(-10, 0)
    } else if (event.key === 'w' || event.key === 'ArrowUp') {
      dragCamera(0, 10)
    } else if (event.key === 's' || event.key === 'ArrowDown') {
      dragCamera(0, -10)
    } else if (event.key === 'q') {
      dragCamera(0, 0, -10)
    } else if (event.key === 'e') {
      dragCamera(0, 0, 10)
    }
    event.preventDefault()
    event.stopPropagation()
  }
}

function onKeyUp(event: KeyboardEvent) {
  multiSelecting.value = event.ctrlKey
}

function dragCamera(deltaX: number, deltaY: number, deltaZ: number = 0) {
  const cameraLine = new Vector3()
  cameraLine.subVectors(cameraPosition.value, orbitTarget.value)
  const direction = cameraLine.clone().normalize()
  const left = new Vector3()
  left.crossVectors(new Vector3(0, 1, 0), direction).normalize()
  const up = new Vector3()
  up.crossVectors(left, direction).normalize()
  const moveScale = cameraLine.length() * 0.001
  const delta = new Vector3()
  delta.addScaledVector(up, -deltaY * moveScale)
  delta.addScaledVector(left, -deltaX * moveScale)
  delta.addScaledVector(direction, -deltaZ * moveScale)
  cameraPosition.value.add(delta)
  orbitTarget.value.add(delta)
}

onBeforeUnmount(() => {
  const container = containerRef.value!
  container.removeEventListener('resize', onResize)
  window.removeEventListener('resize', onResize)
  if (resizeInterval != undefined) {
    clearInterval(resizeInterval)
  }
})

onMounted(() => {
  // make the renderer selected to react to key events (https://stackoverflow.com/a/12887221)
  const canvas: HTMLElement = (rendererRef.value as any).canvas
  canvas.setAttribute('tabindex', '1')
  canvas.style.setProperty('outline-style', 'none') // remove select border
  canvas.addEventListener('mouseenter', () => canvas.focus())
  canvas.addEventListener('mouseleave', () => canvas.blur())

  // set up resize observer
  onResize()
  resizeInterval = setInterval(onResize, 100)
  const container = containerRef.value!
  new ResizeObserver(onResize).observe(container)
  container.addEventListener('resize', onResize)
  window.addEventListener('resize', onResize)

  // set up camera and orbit controls
  const three = (rendererRef.value as any).three
  let orbitControls: OrbitControls

  if (props.cameraType === 'orthographic') {
    // Create orthographic camera with frustum derived from camera-target distance
    const distance = cameraPosition.value.distanceTo(orbitTarget.value)
    const halfHeight = distance * Math.tan(REFERENCE_FOV_RAD / 2)
    const halfWidth = halfHeight * aspectRatio.value
    const orthoCamera = new ThreeOrthographicCamera(
      -halfWidth, halfWidth, halfHeight, -halfHeight, -100000, 100000,
    )
    orthoCamera.position.copy(cameraPosition.value)
    orthoCamera.lookAt(orbitTarget.value)

    // Replace the placeholder perspective camera with our orthographic camera
    three.camera = orthoCamera
    three.cameraCtrl?.dispose()
    orbitControls = new OrbitControls(orthoCamera, canvas)
    orbitControls.enableDamping = false
    orbitControls.enablePan = false
    orbitControls.enableZoom = false // zoom via dolly below
    orbitControls.target.copy(orbitTarget.value)
    three.cameraCtrl = orbitControls

    // Dolly zoom: move camera position instead of changing orthographic zoom
    canvas.addEventListener('wheel', (event: WheelEvent) => {
      event.preventDefault()
      const factor = event.deltaY > 0 ? 1.1 : 1 / 1.1
      const dir = new Vector3().subVectors(cameraPosition.value, orbitTarget.value)
      dir.multiplyScalar(factor)
      cameraPosition.value.copy(orbitTarget.value).add(dir)
      orthoCamera.position.copy(cameraPosition.value)
    }, { passive: false })

    // Sync reactive cameraPosition → orthographic camera object
    watchEffect(() => {
      orthoCamera.position.set(cameraPosition.value.x, cameraPosition.value.y, cameraPosition.value.z)
    })

    // Recompute frustum every frame so zoom (distance change) takes effect
    ;(rendererRef.value as any).onBeforeRender(() => {
      const dist = orthoCamera.position.distanceTo(orbitControls.target)
      const hh = dist * Math.tan(REFERENCE_FOV_RAD / 2)
      const hw = hh * aspectRatio.value
      orthoCamera.left = -hw
      orthoCamera.right = hw
      orthoCamera.top = hh
      orthoCamera.bottom = -hh
      orthoCamera.updateProjectionMatrix()
    })

    orbitControls.addEventListener('change', () => {
      cameraPosition.value.copy(orthoCamera.position)
    })
  } else {
    // Perspective camera (default)
    const camera: ThreePerspectiveCamera = (cameraRef.value as any).camera
    orbitControls = three.cameraCtrl as OrbitControls
    orbitControls.enablePan = false // pan is strange, replace with custom dragCamera
    orbitControls.addEventListener('change', () => {
      cameraPosition.value.copy(camera.position)
    })
  }

  // set up stats
  watchEffect(() => {
    if (statsRef.value != null) {
      const stats = statsRef.value!.stats
      stats.setMode(0) // 0: fps, 1: ms
      stats.domElement.style.position = 'absolute'
      stats.domElement.style.left = '0'
      stats.domElement.style.top = '0'
      const renderer = rendererRef.value!
      renderer.onBeforeRender(stats.begin)
      renderer.onAfterRender(stats.end)
      containerRef.value?.appendChild(stats.domElement)
    }
  })

  // when `orbitTarget` changes, update the orbit controls target
  watchEffect(() => {
    orbitControls.target.copy(orbitTarget.value)
  })
})

function setDisplayMode(gid: bigint, displayMode: PartialDisplayMode) {
  const mode = data.displayMode.get(gid)!
  if (displayMode.showBlock !== undefined) {
    mode.showBlock = displayMode.showBlock
  }
  if (displayMode.showRealization !== undefined) {
    mode.showRealization = displayMode.showRealization
  }
  if (displayMode.showPorts !== undefined) {
    mode.showPorts = displayMode.showPorts
  }
  if (displayMode.showCheckModel !== undefined) {
    mode.showCheckModel = displayMode.showCheckModel
  }
  if (displayMode.showErrorModel !== undefined) {
    mode.showErrorModel = displayMode.showErrorModel
  }
}

function setGlobalDisplayMode(displayMode: PartialDisplayMode) {
  for (const [gid, _gadget] of data.gadgets) {
    setDisplayMode(gid, displayMode)
  }
}
const displayMode = jsonValueRef(displayModePython.value)
watch(
  displayMode,
  () => {
    for (const [gidStr, mode] of Object.entries(displayMode.value)) {
      const gid = BigInt(gidStr)
      setDisplayMode(gid, mode)
    }
  },
  { immediate: true },
)
watch(Array.from(data.displayMode.values()), () => {
  const newDisplayMode: { [gid: string]: PartialDisplayMode } = {}
  for (const [gid, mode] of data.displayMode) {
    newDisplayMode[gid.toString()] = { ...mode }
  }
  // Mutate in-place so bindReactive's watch detects the change
  const current = displayModePython.value
  for (const key of Object.keys(current)) {
    if (!(key in newDisplayMode)) {
      delete current[key]
    }
  }
  Object.assign(current, newDisplayMode)
})

export interface VisualizerPublicInterface {
  setGlobalDisplayMode(displayMode: vis_pb2.DisplayMode): void
  setDisplayMode(gid: bigint, displayMode: PartialDisplayMode): void
}

const selected = jsonValueRef(vis_pb2.MultiSelectable.fromBinary(selectedBytes.value.value))
const hovered = jsonValueRef(vis_pb2.MultiSelectable.fromBinary(hoveredBytes.value.value))
const selectionManagerState = { selected, hovered }

// bidirectional binding with bytes
watch([selected], () => {
  selectedBytes.value.value = vis_pb2.MultiSelectable.toBinary(selected.value)
})
watch([hovered], () => {
  hoveredBytes.value.value = vis_pb2.MultiSelectable.toBinary(hovered.value)
})
watch(selectedBytes.value, () => {
  selected.value = vis_pb2.MultiSelectable.fromBinary(selectedBytes.value.value)
})
watch(hoveredBytes.value, () => {
  hovered.value = vis_pb2.MultiSelectable.fromBinary(hoveredBytes.value.value)
})

defineExpose<VisualizerPublicInterface>({
  setGlobalDisplayMode,
  setDisplayMode,
})
</script>

<template>
  <div class="container" :style="{ width: props.width.value, height: props.height.value }" ref="container"
    @keydown="onKeyDown" @keyup="onKeyUp">
    <!-- placeholder for controller pane container -->

    <SharedRenderer ref="renderer" :params="{
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
      precision: 'highp',
      stencil: true,
    }">
      <PerspectiveCamera :aspect="aspectRatio" :near="0.1" :far="100000" :position="cameraPosition" ref="camera" />
      <Stats v-if="data.showStates.value" :noSetup="true" ref="stats"></Stats>
      <Scene :background="sceneBackground">
        <AmbientLight color="#FFFFFF" :intensity="3"></AmbientLight>
        <SelectionManager :state="selectionManagerState" :multiSelecting="multiSelecting">
          <div v-show="data.showConfig.value" class="config-container">
            <ConfigPane />
          </div>
          <!-- Floating info panes - use snapshot array for stable props during unmount -->
          <FloatingInfoPane v-for="pane of data.infoPanes.value.values()" :key="pane.id" :id="pane.id"
            :initialX="pane.x" :initialY="pane.y" :selection="pane.selection" :zIndex="pane.zIndex" />
          <Blocks />
          <div v-for="gid of data.gadgets.keys()" :key="gid.toString()">
            <Realization :gid="gid" :gateStyle="props.gateStyle" />
          </div>
          <div v-for="cid of data.checkModels.keys()" :key="cid.toString()">
            <CheckModel :cid="cid" />
          </div>
          <div v-for="eid of data.errorModels.keys()" :key="eid.toString()">
            <ErrorModel :eid="eid" />
          </div>
        </SelectionManager>
      </Scene>
    </SharedRenderer>
  </div>
</template>

<style scoped>
.container {
  /* this can remove scroll bar in Jupyter notebook in browser */
  margin: 5px 0 5px 0;
}

canvas {
  margin: 0;
  overflow: hidden;
}

.config-container {
  position: absolute;
  top: 0;
  right: 0;
  width: 300px;
  padding: 0;
  margin: 0;
}
</style>
