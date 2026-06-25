<script setup lang="ts">
import { ref, computed, inject, onBeforeUnmount, watch } from 'vue'
import { VisualizerDataInjectionKey } from '@/misc/VisualizerData'
import { SelectionManagerInjectionKey } from '@/misc/SelectionManager'
import { bigintToString, bigintArrayToString } from '@/util'
import * as vis_pb from '@/proto/visualizer'
import RelationLink from '@/components/RelationLink.vue'

const props = defineProps<{
  id: string
  initialX: number
  initialY: number
  selection: vis_pb.Selectable
  zIndex: number
}>()

const data = inject(VisualizerDataInjectionKey)!
const selectionManager = inject(SelectionManagerInjectionKey)!

// Position state - use transform for GPU acceleration
const x = ref(props.initialX)
const y = ref(props.initialY)

// Local zIndex that watches the infoPanes map for updates
const localZIndex = ref(props.zIndex)

// Watch for zIndex changes from the parent (when focusInfoPane is called)
watch(
  () => data.infoPanes.value.get(props.id)?.zIndex,
  (newZ) => {
    if (newZ !== undefined) {
      localZIndex.value = newZ
    }
  }
)

// Drag state - non-reactive for performance
let isDragging = false
let dragOffsetX = 0
let dragOffsetY = 0

// Use a ref to the DOM element for direct manipulation during drag
const paneRef = ref<HTMLElement | null>(null)

function onTitleMouseDown(e: MouseEvent) {
  isDragging = true
  data.isDraggingPane.value = true  // Signal to SelectionManager to skip hover
  dragOffsetX = e.clientX - x.value
  dragOffsetY = e.clientY - y.value
  window.addEventListener('mousemove', onMouseMove)
  window.addEventListener('mouseup', onMouseUp)
  // Bring to front
  data.focusInfoPane(props.id)
  // Immediately update local zIndex
  const pane = data.infoPanes.value.get(props.id)
  if (pane) {
    localZIndex.value = pane.zIndex
  }
}

function onMouseMove(e: MouseEvent) {
  if (isDragging && paneRef.value) {
    // Direct DOM manipulation for smooth dragging - bypass Vue reactivity
    const newX = e.clientX - dragOffsetX
    const newY = e.clientY - dragOffsetY
    paneRef.value.style.transform = `translate(${newX}px, ${newY}px)`
  }
}

function onMouseUp(e: MouseEvent) {
  if (isDragging) {
    // Update reactive state only on mouse up
    x.value = e.clientX - dragOffsetX
    y.value = e.clientY - dragOffsetY
  }
  isDragging = false
  data.isDraggingPane.value = false  // Resume hover detection
  window.removeEventListener('mousemove', onMouseMove)
  window.removeEventListener('mouseup', onMouseUp)
}

function onPaneMouseDown() {
  // Bring to front on any click
  data.focusInfoPane(props.id)
  // Immediately update local zIndex
  const pane = data.infoPanes.value.get(props.id)
  if (pane) {
    localZIndex.value = pane.zIndex
  }
}

// Track which sections are expanded (collapsed by default)
const expandedSections = ref<Set<string>>(new Set())

function toggleSection(sectionId: string) {
  if (expandedSections.value.has(sectionId)) {
    expandedSections.value.delete(sectionId)
  } else {
    expandedSections.value.add(sectionId)
  }
  // Trigger reactivity
  expandedSections.value = new Set(expandedSections.value)
}

function isSectionExpanded(sectionId: string): boolean {
  return expandedSections.value.has(sectionId)
}

function onClose() {
  data.closeInfoPane(props.id)
}

onBeforeUnmount(() => {
  window.removeEventListener('mousemove', onMouseMove)
  window.removeEventListener('mouseup', onMouseUp)
})

// Compute the title based on selection type
const title = computed(() => {
  const sel = props.selection
  switch (sel.e.oneofKind) {
    case 'gadget':
      return `Gadget gid=${sel.e.gadget.gid}`
    case 'location':
      return `Location gid=${sel.e.location.gid} #${sel.e.location.locationIndex}`
    case 'port': {
      const io = sel.e.port.io
      const ioStr = io.oneofKind === 'input' ? `in[${io.input}]` : io.oneofKind === 'output' ? `out[${io.output}]` : '?'
      return `Port gid=${sel.e.port.gid} ${ioStr}`
    }
    case 'observable': {
      const io = sel.e.observable.io
      const ioStr = io.oneofKind === 'input' ? `in[${io.input}]` : io.oneofKind === 'output' ? `out[${io.output}]` : '?'
      return `Observable gid=${sel.e.observable.gid} ${ioStr} #${sel.e.observable.observableIndex}`
    }
    case 'check':
      return `Check cid=${sel.e.check.cid} #${sel.e.check.checkIndex}`
    case 'error':
      return `Error eid=${sel.e.error.eid} #${sel.e.error.errorIndex}`
    default:
      return 'Unknown'
  }
})

// Get detailed info based on selection type
const checkInfo = computed(() => {
  if (props.selection.e.oneofKind !== 'check') return null
  const { cid, checkIndex } = props.selection.e.check
  const checkModel = data.checkModel(cid)
  const checkModelType = data.checkModelType(checkModel.ctype)
  const check = checkModelType.checks[checkIndex]
  const gadget = data.gadget(checkModel.gid)
  const gadgetType = data.gadgetType(gadget.gtype)
  const attachedEids = data.checkModelAttach(cid)

  return {
    cid,
    checkIndex,
    tag: check?.tag || '(no tag)',
    naturallyFlipped: check?.naturallyFlipped ?? false,
    measurementCount: check?.measurements.length ?? 0,
    gadgetGid: checkModel.gid,
    gadgetName: gadgetType.name,
    attachedEids,
  }
})

const errorInfo = computed(() => {
  if (props.selection.e.oneofKind !== 'error') return null
  const { eid, errorIndex } = props.selection.e.error
  const errorModel = data.errorModel(eid)
  const errorModelType = data.errorModelType(errorModel.etype)
  const error = errorModelType.errors[errorIndex]
  const remoteCheckModels = data.remoteCheckModelVec(eid)

  // Collect connected checks
  const connectedChecks: { cid: bigint; checkIndex: number }[] = []
  for (const checkRef of error?.checks || []) {
    const remote = checkRef.remoteCheckModel !== undefined
      ? remoteCheckModels[Number(checkRef.remoteCheckModel)]
      : undefined
    const remoteCid = remote?.cid ?? errorModel.cid
    const bias = remote?.bias ?? 0n
    connectedChecks.push({
      cid: remoteCid,
      checkIndex: Number(checkRef.checkIndex + bias),
    })
  }

  return {
    eid,
    errorIndex,
    tag: error?.tag || '(no tag)',
    probability: error?.probability,
    residual: error?.residual ?? [],
    parentCid: errorModel.cid,
    connectedChecks,
  }
})

const gadgetInfo = computed(() => {
  if (props.selection.e.oneofKind !== 'gadget') return null
  const { gid } = props.selection.e.gadget
  const gadget = data.gadget(gid)
  const gadgetType = data.gadgetType(gadget.gtype)
  const boundCid = data.gadgetBinding(gid)
  const attachedEids = boundCid !== undefined ? data.checkModelAttach(boundCid) : []

  return {
    gid,
    name: gadgetType.name,
    position: gadget.position,
    inputCount: gadgetType.inputs.length,
    outputCount: gadgetType.outputs.length,
    measurementCount: gadgetType.measurements.length,
    boundCid,
    attachedEids,
  }
})

const locationInfo = computed(() => {
  if (props.selection.e.oneofKind !== 'location') return null
  const { gid, locationIndex } = props.selection.e.location
  const gadget = data.gadget(gid)
  const gadgetType = data.gadgetType(gadget.gtype)
  const realization = gadgetType.realization
  const location = realization?.locations[locationIndex]
  const operation = location?.operation

  // Get the attached error model for this gadget (if any)
  let attachedEid: bigint | undefined = undefined
  const cid = data.gadgetBinding(gid)
  if (cid !== undefined) {
    const eids = data.checkModelAttach(cid)
    attachedEid = eids.length > 0 ? eids[0] : undefined
  }

  // Process noise distributions for display
  const noises = location?.noises ?? []
  const noiseSummary = noises.map((dist, distIdx) => {
    const masses = dist.masses.map((mass) => {
      const faultStr = mass.faults.map(f => `${f.type}@q${f.qubit}`).join(', ')
      return {
        faults: faultStr || '(identity)',
        probability: mass.probability,
        edgeIndex: mass.edgeIndex,
        // Include eid for clickable link if edge_index is present and we have an attached error model
        eid: mass.edgeIndex !== undefined && attachedEid !== undefined ? attachedEid : undefined,
      }
    })
    return { distIdx, masses }
  })

  return {
    gid,
    locationIndex,
    t: location?.t ?? 0,
    operationType: operation?.type !== undefined ? vis_pb.OperationType[operation.type] : 'UNKNOWN',
    support: operation?.support || [],
    pauli: operation?.pauli || '',
    inverted: operation?.inverted ?? false,
    noiseCount: noises.length,
    noiseSummary,
  }
})

// Helper to create a Selectable for a check
function makeCheckSelectable(cid: bigint, checkIndex: number): vis_pb.Selectable {
  return {
    e: {
      oneofKind: 'check',
      check: { cid, checkIndex },
    },
  }
}

// Helper to create a Selectable for an error
function makeErrorSelectable(eid: bigint, errorIndex: number): vis_pb.Selectable {
  return {
    e: {
      oneofKind: 'error',
      error: { eid, errorIndex },
    },
  }
}

// Helper to create a Selectable for a gadget
function makeGadgetSelectable(gid: bigint): vis_pb.Selectable {
  return {
    e: {
      oneofKind: 'gadget',
      gadget: { gid },
    },
  }
}

// Get error labels for display
function getErrorsForCheck(attachedEids: bigint[]): { eid: bigint; errorIndex: number; label: string }[] {
  const errors: { eid: bigint; errorIndex: number; label: string }[] = []
  for (const eid of attachedEids) {
    const errorModel = data.errorModel(eid)
    const errorModelType = data.errorModelType(errorModel.etype)
    for (let i = 0; i < errorModelType.errors.length; i++) {
      const error = errorModelType.errors[i]
      errors.push({
        eid,
        errorIndex: i,
        label: `eid=${bigintToString(eid)} #${i} ${error?.tag}`,
      })
    }
  }
  return errors
}

function onPaneMouseEnter() {
  // Clear hover state when mouse enters the pane
  selectionManager.hover({ elements: [] })
}
</script>

<template>
  <div ref="paneRef" class="floating-pane" :style="{ transform: `translate(${x}px, ${y}px)`, zIndex: localZIndex }"
    @mousedown="onPaneMouseDown" @mouseenter="onPaneMouseEnter">
    <div class="pane-header" @mousedown.stop="onTitleMouseDown">
      <span class="pane-title">{{ title }}</span>
      <div class="pane-buttons">
        <button class="pane-btn close-btn" @click.stop="onClose" title="Close pane">×</button>
      </div>
    </div>

    <div class="pane-content">
      <template v-if="checkInfo">
        <div class="info-section">
          <div class="info-row">
            <span class="info-label">CID:</span>
            <span class="info-value">{{ bigintToString(checkInfo.cid) }}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Index:</span>
            <span class="info-value">{{ checkInfo.checkIndex }}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Tag:</span>
            <span class="info-value">{{ checkInfo.tag }}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Naturally Flipped:</span>
            <span class="info-value">{{ checkInfo.naturallyFlipped }}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Measurements:</span>
            <span class="info-value">{{ checkInfo.measurementCount }}</span>
          </div>
        </div>

        <div class="info-section">
          <div class="section-title">Parent Gadget</div>
          <RelationLink :target="makeGadgetSelectable(checkInfo.gadgetGid)"
            :label="`${checkInfo.gadgetName} (gid=${bigintToString(checkInfo.gadgetGid)})`" />
        </div>

        <div class="info-section" v-if="checkInfo.attachedEids.length > 0">
          <div class="section-title collapsible" @click="toggleSection('check-errors')">
            <span class="collapse-icon">{{ isSectionExpanded('check-errors') ? '▼' : '▶' }}</span>
            Connected Errors ({{ getErrorsForCheck(checkInfo.attachedEids).length }})
          </div>
          <div class="link-list" v-show="isSectionExpanded('check-errors')">
            <div v-for="err in getErrorsForCheck(checkInfo.attachedEids)"
              :key="`${bigintToString(err.eid)}-${err.errorIndex}`">
              <RelationLink :target="makeErrorSelectable(err.eid, err.errorIndex)" :label="err.label" />
            </div>
          </div>
        </div>
      </template>

      <template v-else-if="errorInfo">
        <div class="info-section">
          <div class="info-row">
            <span class="info-label">EID:</span>
            <span class="info-value">{{ bigintToString(errorInfo.eid) }}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Index:</span>
            <span class="info-value">{{ errorInfo.errorIndex }}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Tag:</span>
            <span class="info-value">{{ errorInfo.tag }}</span>
          </div>
          <div class="info-row" v-if="errorInfo.probability !== undefined">
            <span class="info-label">Probability:</span>
            <span class="info-value">{{ errorInfo.probability.toExponential(3) }}</span>
          </div>
          <div class="info-row" v-if="errorInfo.residual.length > 0">
            <span class="info-label">Residual:</span>
            <span class="info-value">[{{ bigintArrayToString(errorInfo.residual).join(', ') }}]</span>
          </div>
        </div>

        <div class="info-section" v-if="errorInfo.connectedChecks.length > 0">
          <div class="section-title collapsible" @click="toggleSection('error-checks')">
            <span class="collapse-icon">{{ isSectionExpanded('error-checks') ? '▼' : '▶' }}</span>
            Connected Checks ({{ errorInfo.connectedChecks.length }})
          </div>
          <div class="link-list" v-show="isSectionExpanded('error-checks')">
            <div v-for="(check, idx) in errorInfo.connectedChecks" :key="idx">
              <RelationLink :target="makeCheckSelectable(check.cid, check.checkIndex)"
                :label="`cid=${bigintToString(check.cid)} #${check.checkIndex}`" />
            </div>
          </div>
        </div>
      </template>

      <template v-else-if="gadgetInfo">
        <div class="info-section">
          <div class="info-row">
            <span class="info-label">GID:</span>
            <span class="info-value">{{ bigintToString(gadgetInfo.gid) }}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Name:</span>
            <span class="info-value">{{ gadgetInfo.name }}</span>
          </div>
          <div class="info-row" v-if="gadgetInfo.position">
            <span class="info-label">Position:</span>
            <span class="info-value">(t={{ gadgetInfo.position.t }}, i={{ gadgetInfo.position.i }}, j={{
              gadgetInfo.position.j }})</span>
          </div>
          <div class="info-row">
            <span class="info-label">Inputs:</span>
            <span class="info-value">{{ gadgetInfo.inputCount }}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Outputs:</span>
            <span class="info-value">{{ gadgetInfo.outputCount }}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Measurements:</span>
            <span class="info-value">{{ gadgetInfo.measurementCount }}</span>
          </div>
        </div>

        <div class="info-section" v-if="gadgetInfo.boundCid !== undefined">
          <div class="section-title">Bound Check Model</div>
          <RelationLink :target="makeCheckSelectable(gadgetInfo.boundCid, 0)"
            :label="`cid=${bigintToString(gadgetInfo.boundCid)}`" />
        </div>

        <div class="info-section" v-if="gadgetInfo.attachedEids.length > 0">
          <div class="section-title collapsible" @click="toggleSection('gadget-errors')">
            <span class="collapse-icon">{{ isSectionExpanded('gadget-errors') ? '▼' : '▶' }}</span>
            Attached Errors ({{ getErrorsForCheck(gadgetInfo.attachedEids).length }})
          </div>
          <div class="link-list" v-show="isSectionExpanded('gadget-errors')">
            <div v-for="err in getErrorsForCheck(gadgetInfo.attachedEids)"
              :key="`${bigintToString(err.eid)}-${err.errorIndex}`">
              <RelationLink :target="makeErrorSelectable(err.eid, err.errorIndex)" :label="err.label" />
            </div>
          </div>
        </div>
      </template>

      <template v-else-if="locationInfo">
        <div class="info-section">
          <div class="info-row">
            <span class="info-label">GID:</span>
            <span class="info-value">{{ bigintToString(locationInfo.gid) }}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Index:</span>
            <span class="info-value">{{ locationInfo.locationIndex }}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Time (t):</span>
            <span class="info-value">{{ locationInfo.t }}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Operation:</span>
            <span class="info-value">{{ locationInfo.operationType }}</span>
          </div>
          <div class="info-row" v-if="locationInfo.support.length > 0">
            <span class="info-label">Support:</span>
            <span class="info-value">[{{ locationInfo.support.join(', ') }}]</span>
          </div>
          <div class="info-row" v-if="locationInfo.pauli">
            <span class="info-label">Pauli:</span>
            <span class="info-value">{{ locationInfo.pauli }}</span>
          </div>
          <div class="info-row" v-if="locationInfo.inverted">
            <span class="info-label">Inverted:</span>
            <span class="info-value">{{ locationInfo.inverted }}</span>
          </div>
        </div>

        <div class="info-section" v-if="locationInfo.noiseCount > 0">
          <div class="section-title collapsible" @click="toggleSection('location-noises')">
            <span class="collapse-icon">{{ isSectionExpanded('location-noises') ? '▼' : '▶' }}</span>
            Noise Distributions ({{ locationInfo.noiseCount }})
          </div>
          <div class="noise-list" v-show="isSectionExpanded('location-noises')">
            <div v-for="dist in locationInfo.noiseSummary" :key="dist.distIdx" class="noise-dist">
              <div v-if="locationInfo.noiseCount > 1" class="noise-dist-header">Distribution #{{ dist.distIdx }}</div>
              <div v-for="(mass, massIdx) in dist.masses" :key="massIdx" class="noise-mass">
                <div class="noise-mass-row">
                  <span class="noise-faults">{{ mass.faults }}</span>
                  <span class="noise-prob">{{ mass.probability.toExponential(3) }}</span>
                </div>
                <div v-if="mass.edgeIndex !== undefined" class="noise-edge">
                  <template v-if="mass.eid !== undefined">
                    <RelationLink :target="makeErrorSelectable(mass.eid, mass.edgeIndex)"
                      :label="`edge #${mass.edgeIndex}`" />
                  </template>
                  <template v-else>
                    edge #{{ mass.edgeIndex }}
                  </template>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="info-section">
          <div class="section-title">Parent Gadget</div>
          <RelationLink :target="makeGadgetSelectable(locationInfo.gid)"
            :label="`gid=${bigintToString(locationInfo.gid)}`" />
        </div>
      </template>

      <template v-else>
        <div class="info-section">
          <span class="info-value">Unknown selection type</span>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.floating-pane {
  position: fixed;
  left: 0;
  top: 0;
  min-width: 280px;
  max-width: 400px;
  background: rgba(28, 28, 28, 0.95);
  border-radius: 6px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
  color: #e0e0e0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-size: 12px;
  overflow: hidden;
  pointer-events: auto;
  will-change: transform;
}

.pane-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: rgba(50, 50, 50, 0.95);
  padding: 8px 10px;
  cursor: grab;
  user-select: none;
  border-bottom: 1px solid rgba(80, 80, 80, 0.5);
}

.pane-header:active {
  cursor: grabbing;
}

.pane-title {
  font-weight: 600;
  color: #fff;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.pane-buttons {
  display: flex;
  gap: 4px;
  margin-left: 8px;
}

.pane-btn {
  background: transparent;
  border: none;
  color: #aaa;
  cursor: pointer;
  font-size: 14px;
  padding: 2px 6px;
  border-radius: 3px;
  transition: all 0.15s ease;
}

.pane-btn:hover {
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
}

.close-btn {
  font-size: 18px;
  line-height: 1;
}

.close-btn:hover {
  background: rgba(255, 80, 80, 0.3);
  color: #ff6b6b;
}

.pane-content {
  padding: 10px;
  max-height: 400px;
  overflow-y: auto;
}

.info-section {
  margin-bottom: 12px;
}

.info-section:last-child {
  margin-bottom: 0;
}

.section-title {
  font-weight: 600;
  color: #aaa;
  margin-bottom: 6px;
  padding-bottom: 4px;
  border-bottom: 1px solid rgba(80, 80, 80, 0.5);
  text-transform: uppercase;
  font-size: 10px;
  letter-spacing: 0.5px;
}

.section-title.collapsible {
  cursor: pointer;
  user-select: none;
  transition: color 0.15s ease;
}

.section-title.collapsible:hover {
  color: #ccc;
}

.collapse-icon {
  display: inline-block;
  width: 12px;
  font-size: 8px;
  margin-right: 4px;
}

.info-row {
  display: flex;
  justify-content: space-between;
  padding: 3px 0;
}

.info-label {
  color: #888;
}

.info-value {
  color: #e0e0e0;
  text-align: right;
  max-width: 60%;
  overflow: hidden;
  text-overflow: ellipsis;
}

.link-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-height: 150px;
  overflow-y: auto;
  padding: 4px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 4px;
}

/* Noise distribution styling */
.noise-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 200px;
  overflow-y: auto;
  padding: 4px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 4px;
}

.noise-dist {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.noise-dist-header {
  font-weight: 600;
  color: #aaa;
  font-size: 10px;
  text-transform: uppercase;
  padding-bottom: 2px;
  border-bottom: 1px solid rgba(80, 80, 80, 0.3);
}

.noise-mass {
  padding: 4px 6px;
  background: rgba(40, 40, 40, 0.6);
  border-radius: 3px;
  font-family: 'Monaco', 'Consolas', monospace;
  font-size: 11px;
}

.noise-mass-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
}

.noise-faults {
  color: #e0e0e0;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
}

.noise-prob {
  color: #8bc34a;
  font-weight: 500;
  white-space: nowrap;
}

.noise-edge {
  color: #888;
  font-size: 10px;
  margin-top: 2px;
}

/* Scrollbar styling */
.pane-content::-webkit-scrollbar,
.link-list::-webkit-scrollbar,
.noise-list::-webkit-scrollbar {
  width: 6px;
}

.pane-content::-webkit-scrollbar-track,
.link-list::-webkit-scrollbar-track,
.noise-list::-webkit-scrollbar-track {
  background: rgba(0, 0, 0, 0.2);
  border-radius: 3px;
}

.pane-content::-webkit-scrollbar-thumb,
.link-list::-webkit-scrollbar-thumb,
.noise-list::-webkit-scrollbar-thumb {
  background: rgba(100, 100, 100, 0.5);
  border-radius: 3px;
}

.pane-content::-webkit-scrollbar-thumb:hover,
.link-list::-webkit-scrollbar-thumb:hover,
.noise-list::-webkit-scrollbar-thumb:hover {
  background: rgba(120, 120, 120, 0.7);
}
</style>
