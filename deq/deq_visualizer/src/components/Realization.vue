<script setup lang="ts">
import { inject, ref, watch, computed } from 'vue'
import { VisualizerDataInjectionKey, VisualizerData } from '@/misc/VisualizerData'
import MeshVecProvider, { type MeshVecProviderPublicInterface, MeshGroup } from '@/misc/MeshVecProvider'
import { Position, MultiSelectable, Realization, OperationType, Operation, Mesh } from '@/proto/visualizer'
import { assert, originRotationOf, areSetsEqual } from '@/util'
import {
  meshConnecting,
  supportRelativeOf,
  colorOfPauli,
  meshesOfPauliX,
  meshesOfPauliY,
  meshesOfPauliZ,
  transformMeshes,
  flattenMeshesForTopView,
  eulerRotationOf,
  parseMeasurementPauli,
  parsePauliString,
} from '@/misc/meshes'

const data = inject(VisualizerDataInjectionKey)!
assert(data != undefined, 'missing visualizerData injection')

interface Props {
  gid: bigint
  gateStyle?: string // 'top' | 'front'
}
const props = withDefaults(defineProps<Props>(), {
  gateStyle: 'top',
})

const gadget = data.gadget(props.gid)
const gadgetType = data.gadgetType(gadget.gtype)

const displayMode = data.displayMode.get(props.gid)!
const show = computed(() => {
  return displayMode.showRealization && gadgetType.realization !== undefined
})

function locationSelectionOf(locationIndex: number): MultiSelectable {
  return {
    elements: [{ e: { oneofKind: 'location', location: { gid: props.gid, locationIndex } } }],
  }
}
const locationMeshVecProvider = ref<MeshVecProviderPublicInterface>(null!)
function loadLocationGroups(meshVecProvider: MeshVecProviderPublicInterface) {
  if (gadgetType.realization == undefined) {
    return
  }
  const locations = gadgetType.realization.locations
  for (const [locationIndex, location] of locations.entries()) {
    const group = new MeshGroup<number>([], locationIndex, locationSelectionOf)
    group.relative = Position.create({ t: location.t })
    // add the operation mesh
    group.meshes.push(...operationMeshes(location.operation!))
    // add the noise mesh (if present)
    if (location.noises) {
      const noisySupports: Set<number> = new Set()
      for (const noise of location.noises) {
        for (const mass of noise.masses) {
          for (const fault of mass.faults) {
            noisySupports.add(fault.qubit)
          }
        }
      }
      for (const qid of noisySupports) {
        const height = 1 * data.qubitRadius
        const width = 0.3 * data.qubitRadius
        const relative = supportRelativeOf(qid, gadgetType.realization!)
        relative.t += height / 2 // to make it clear that errors happen **after** the operation
        group.meshes.push(
          Mesh.create({
            geometry: { type: 'box', size: [height, width] },
            material: { type: 'standard', color: '#FFCC00' },
            relative,
          }),
        )
      }
    }
    meshVecProvider.addGroup(group)
  }
}
watch([locationMeshVecProvider], ([meshVecProvider]) => {
  if (meshVecProvider) {
    loadLocationGroups(meshVecProvider)
    meshVecProvider.finishLoading()
  }
})

const lifetimeMeshVecProvider = ref<MeshVecProviderPublicInterface>(null!)
function loadLifetimeGroups(meshVecProvider: MeshVecProviderPublicInterface) {
  if (gadgetType.realization == undefined) {
    return
  }
  const qubitsLifetimes = qubitsLifetimeOf(data, gadgetType.realization)
  const realization = gadgetType.realization
  const group = new MeshGroup()
  for (let qid = 0; qid < realization.positions.length; ++qid) {
    for (const range of qubitsLifetimes[qid]!.ranges) {
      const a = Position.create({
        t: range.start,
        i: realization.positions[qid]!.i,
        j: realization.positions[qid]!.j,
      })
      const b = Position.create({
        t: range.end,
        i: realization.positions[qid]!.i,
        j: realization.positions[qid]!.j,
      })
      group.meshes.push(meshConnecting(a, b, 0.05 * data.qubitRadius, 'lightgrey'))
    }
  }
  meshVecProvider.addGroup(group)
}
watch([lifetimeMeshVecProvider], ([meshVecProvider]) => {
  if (meshVecProvider) {
    loadLifetimeGroups(meshVecProvider)
    meshVecProvider.finishLoading()
  }
})

type QubitLifetime = {
  ranges: LifeRange[]
}

type LifeRange = {
  start?: number
  end?: number
}

function qubitsLifetimeOf(data: VisualizerData, realization: Realization): QubitLifetime[] {
  const qubitsLifetimes = Array.from({ length: realization.positions.length }, () => ({ ranges: [] as LifeRange[] }))
  // if a qubit is never touched, give it a lifetime that spans the whole realization
  const locations = realization.locations
  if (locations.length === 0) {
    return qubitsLifetimes
  }
  const defaultStart = locations[0]!.t
  const defaultEnd = locations[locations.length - 1]!.t
  for (const location of locations) {
    if (location.operation!.type === OperationType.PREPARE) {
      for (const qid of location.operation!.support) {
        const qubitLifetime = qubitsLifetimes[qid]!
        if (qubitLifetime.ranges.length === 0 || qubitLifetime.ranges[qubitLifetime.ranges.length - 1]!.end !== undefined) {
          qubitLifetime.ranges.push({ start: location.t })
        }
      }
    } else if (location.operation!.type === OperationType.DISCARD) {
      for (const qid of location.operation!.support) {
        const qubitLifetime = qubitsLifetimes[qid]!
        if (qubitLifetime.ranges.length === 0) {
          qubitLifetime.ranges.push({ end: location.t })
        } else if (qubitLifetime.ranges[qubitLifetime.ranges.length - 1]!.end === undefined) {
          qubitLifetime.ranges[qubitLifetime.ranges.length - 1]!.end = location.t
        }
      }
    } else if (location.operation!.type === OperationType.SHUFFLE) {
      const shuttleHeight = shuttleHeightScale * data.qubitRadius
      for (const qid of new Set(location.operation!.support)) {
        const qubitLifetime = qubitsLifetimes[qid]!
        if (qubitLifetime.ranges.length === 0) {
          qubitLifetime.ranges.push({ end: location.t - shuttleHeight })
        } else if (qubitLifetime.ranges[qubitLifetime.ranges.length - 1]!.end === undefined) {
          qubitLifetime.ranges[qubitLifetime.ranges.length - 1]!.end = location.t - shuttleHeight
        }
        qubitLifetime.ranges.push({ start: location.t })
      }
    }
  }
  for (const qubitLifetime of qubitsLifetimes) {
    if (qubitLifetime.ranges.length === 0) {
      qubitLifetime.ranges.push({ start: defaultStart, end: defaultEnd })
      continue
    }
    if (qubitLifetime.ranges[0]!.start === undefined) {
      qubitLifetime.ranges[0]!.start = defaultStart
    }
    if (qubitLifetime.ranges[qubitLifetime.ranges.length - 1]!.end === undefined) {
      qubitLifetime.ranges[qubitLifetime.ranges.length - 1]!.end = defaultEnd
    }
  }
  return qubitsLifetimes
}

function operationMeshes(operation: Operation): Mesh[] {
  if (operation.type === OperationType.PREPARE) {
    return prepareMeshes(operation)
  } else if (operation.type === OperationType.HADAMARD) {
    return hadamardMeshes(operation)
  } else if (operation.type === OperationType.SQRT_PAULI) {
    return sqrtPauliMeshes(operation)
  } else if (operation.type === OperationType.CONTROLLED_PAULI) {
    return controlledPauliMeshes(operation)
  } else if (operation.type === OperationType.MEASURE) {
    return measureMeshes(operation)
  } else if (operation.type === OperationType.SHUFFLE) {
    return shuffleMeshes(operation)
  } else if (operation.type === OperationType.DISCARD) {
    return discardMeshes(operation)
  } else if (operation.type === OperationType.PAULI) {
    return pauliMeshes(operation)
  } else if (operation.type === OperationType.CONDITIONAL_PAULI) {
    return conditionalPauliMeshes(operation)
  } else {
    console.error('Unsupported operation in realization:', OperationType[operation.type])
    return unknownOperationMeshes(operation)
  }
}

function unknownOperationMeshes(operation: Operation): Mesh[] {
  const length = 0.8 * data.qubitRadius
  return operation.support.map((support) =>
    Mesh.create({
      geometry: { type: 'box', size: [length, length, length] },
      material: { type: 'standard', color: 'red' },
      relative: supportRelativeOf(support, gadgetType.realization!),
    }),
  )
}

function prepareMeshes(operation: Operation): Mesh[] {
  if (operation.support.length !== 1) {
    console.error('Prepare operation should have exactly one support, actually: ' + operation.support.length)
    return unknownOperationMeshes(operation)
  }
  if (!is_valid_pauli_string(operation.pauli)) {
    console.error('Prepare operation should have Pauli strings like +X, -Y, etc., actually: ' + operation.pauli)
    return unknownOperationMeshes(operation)
  }
  const qubitIndex = operation.support[0]!
  const radius = 1 * data.qubitRadius
  const height = 2 * data.qubitRadius
  return [
    Mesh.create({
      geometry: { type: 'cone', size: [radius, height] },
      material: { type: 'standard', color: colorOfPauli[operation.pauli] },
      relative: supportRelativeOf(qubitIndex, gadgetType.realization!),
    }),
  ]
}

function is_valid_pauli_string(pauli: string): boolean {
  const valid_paulis = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
  return valid_paulis.includes(pauli)
}

function hadamardMeshes(operation: Operation): Mesh[] {
  if (operation.support.length !== 1) {
    console.error('Hadamard operation should have exactly one support, actually: ' + operation.support.length)
    return unknownOperationMeshes(operation)
  }
  if (operation.pauli.length !== 0) {
    console.error('Hadamard operation does not take Pauli string, ignored')
  }
  const qubitIndex = operation.support[0]!
  const radius = 0.2 * data.qubitRadius
  const width = 0.618 * data.qubitRadius
  const height = 2 * data.qubitRadius
  const r = supportRelativeOf(qubitIndex, gadgetType.realization!)
  let meshes = [
    Mesh.create({
      geometry: { type: 'box', size: [height, radius, radius] },
      material: { type: 'standard', color: 'blue' },
      relative: { i: r.i, j: r.j - width },
    }),
    Mesh.create({
      geometry: { type: 'box', size: [height, radius, radius] },
      material: { type: 'standard', color: 'blue' },
      relative: { i: r.i, j: r.j + width },
    }),
    Mesh.create({
      geometry: { type: 'box', size: [radius, radius, width * 2] },
      material: { type: 'standard', color: 'blue' },
      relative: { i: r.i, j: r.j },
    }),
  ]
  if (props.gateStyle === 'top') {
    meshes = flattenMeshesForTopView(meshes)
  }
  return meshes
}

function sqrtPauliMeshes(operation: Operation): Mesh[] {
  if (operation.support.length !== 1) {
    console.error('SqrtPauli operation should have exactly one support, actually: ' + operation.support.length)
    return unknownOperationMeshes(operation)
  }
  if (operation.pauli.length !== 1) {
    console.error('SqrtPauli operation should take exactly one Pauli, actually: ' + operation.pauli)
    return unknownOperationMeshes(operation)
  }
  const qubitIndex = operation.support[0]!
  const radius = 0.2
  let meshes: Mesh[] = [
    Mesh.create({
      geometry: { type: 'box', size: [1.5, radius, radius] },
      material: { type: 'standard', color: 'black' },
      relative: { t: 1.4, j: -0.2 },
      rotation: { i: Math.PI / 2 },
    }),
    Mesh.create({
      geometry: { type: 'box', size: [2.3, radius, radius] },
      material: { type: 'standard', color: 'black' },
      relative: { t: 0.35, j: -1 },
      rotation: { i: -0.1 },
    }),
    Mesh.create({
      geometry: { type: 'box', size: [0.6, radius, radius] },
      material: { type: 'standard', color: 'black' },
      relative: { t: -0.55, j: -1.25 },
      rotation: { i: 0.7 },
    }),
  ]
  if (operation.pauli === 'X') {
    meshes.push(...meshesOfPauliX('red', radius))
  } else if (operation.pauli === 'Y') {
    meshes.push(...meshesOfPauliY('green', radius))
  } else if (operation.pauli === 'Z') {
    meshes.push(...meshesOfPauliZ('blue', radius))
  }
  if (props.gateStyle === 'top') {
    meshes = flattenMeshesForTopView(meshes)
  }
  const scale = 0.8 * data.qubitRadius
  const move = Position.create(supportRelativeOf(qubitIndex, gadgetType.realization!))
  return transformMeshes(meshes, scale, move)
}

function controlledPauliMeshes(operation: Operation): Mesh[] {
  if (operation.support.length !== 2) {
    console.error('ControlledPauli operation should have exactly two supports, actually: ' + operation.support.length)
    return unknownOperationMeshes(operation)
  }
  if (operation.pauli.length !== 1) {
    console.error('ControlledPauli operation should take exactly one Pauli, actually: ' + operation.pauli)
    return unknownOperationMeshes(operation)
  }
  const controlIndex = operation.support[0]!
  const targetIndex = operation.support[1]!
  const radius = 0.2
  const scaledRadius = radius * data.qubitRadius
  const controlRelative = supportRelativeOf(controlIndex, gadgetType.realization!)
  const targetRelative = supportRelativeOf(targetIndex, gadgetType.realization!)
  let meshes: Mesh[] = [
    // control qubit sphere
    Mesh.create({
      geometry: { type: 'sphere', size: [3 * scaledRadius] },
      material: { type: 'standard', color: 'black' },
      relative: controlRelative,
    }),
  ]
  const scale = 1 * data.qubitRadius
  const move = Position.create(targetRelative)
  const width = 1.4 * scaledRadius
  let connectingTargetRelative = targetRelative
  if (operation.pauli === 'X') {
    // meshes.push(...transformMeshes(meshesOfPauliX('red', radius), scale, move))
    meshes.push(
      Mesh.create({
        geometry: { type: 'torus', size: [10 * scaledRadius] },
        material: { type: 'standard', color: 'black' },
        relative: targetRelative,
        rotation: props.gateStyle === 'front'
          ? {}
          : { j: Math.PI / 2 },
      }),
    )
    const dt = targetRelative.t - controlRelative.t
    const di = targetRelative.i - controlRelative.i
    const dj = targetRelative.j - controlRelative.j
    const normalize = 1 / Math.sqrt(dt * dt + di * di + dj * dj)
    const biasLength = 10 * scaledRadius
    connectingTargetRelative = {
      t: targetRelative.t + dt * biasLength * normalize,
      i: targetRelative.i + di * biasLength * normalize,
      j: targetRelative.j + dj * biasLength * normalize,
    }
    // also draw the perpendicular line
    if (props.gateStyle !== 'front') {
      const perpendicularRelative = Position.create({
        i: controlRelative.j - targetRelative.j,
        j: targetRelative.i - controlRelative.i,
      })
      meshes.push(
        Mesh.create({
          geometry: { type: 'box', size: [20 * scaledRadius, width] },
          material: { type: 'standard', color: 'black' },
          relative: targetRelative,
          rotation: originRotationOf(eulerRotationOf(Position.create(), perpendicularRelative)),
        }),
      )
    } else {
      meshes.push(
        Mesh.create({
          geometry: { type: 'box', size: [20 * scaledRadius, width] },
          material: { type: 'standard', color: 'black' },
          relative: targetRelative,
        }),
      )
    }
  } else if (operation.pauli === 'Y') {
    let yMeshes = meshesOfPauliY('green', radius)
    if (props.gateStyle === 'top') yMeshes = flattenMeshesForTopView(yMeshes)
    meshes.push(...transformMeshes(yMeshes, scale, move))
  } else if (operation.pauli === 'Z') {
    // meshes.push(...transformMeshes(meshesOfPauliZ('blue', radius), scale, move))
    meshes.push(
      Mesh.create({
        geometry: { type: 'sphere', size: [3 * scaledRadius] },
        material: { type: 'standard', color: 'black' },
        relative: targetRelative,
      }),
    )
  }
  // draw the connecting line between control and target
  meshes.push(meshConnecting(controlRelative, connectingTargetRelative, width))
  return meshes
}

function measureMeshes(operation: Operation): Mesh[] {
  if (operation.support.length < 1) {
    console.error('Measure operation should have at least one support, actually: ' + operation.support.length)
    return unknownOperationMeshes(operation)
  }
  const measurePauli = parseMeasurementPauli(operation.pauli, operation.support.length)
  if (!measurePauli) {
    // error already printed
    return unknownOperationMeshes(operation)
  }
  const radius = 1 * data.qubitRadius
  const height = 2 * data.qubitRadius
  const meshes: Mesh[] = []
  let previousRelative: Position | undefined = undefined
  for (let i = 0; i < operation.support.length; i++) {
    const qubitIndex = operation.support[i]!
    const relative = supportRelativeOf(qubitIndex, gadgetType.realization!)
    meshes.push(
      Mesh.create({
        geometry: { type: 'cone', size: [radius, height] },
        material: { type: 'standard', color: colorOfPauli[measurePauli.observable.signedCharAt(i)] },
        relative: relative,
        rotation: { j: Math.PI },
      }),
    )
    meshes.push(
      Mesh.create({
        geometry: { type: 'torus', size: [1.5 * radius] },
        material: { type: 'standard', color: 'black' },
        relative: relative,
        rotation: { t: Math.PI / 4 },
      }),
    )
    meshes.push(
      Mesh.create({
        geometry: { type: 'torus', size: [1.5 * radius] },
        material: { type: 'standard', color: 'black' },
        relative: relative,
        rotation: { t: 3 * Math.PI / 4 },
      }),
    )
    if (previousRelative) {
      meshes.push(meshConnecting(previousRelative, relative, 0.3 * radius, 'darkgrey'))
    }
    previousRelative = relative
  }
  return meshes
}

function shuffleMeshes(operation: Operation): Mesh[] {
  const mapping = shuffleMapping(operation.support)
  if (mapping === null) {
    console.error(`Shuffle operation invalid supports, actually: ${operation.support}`)
    return unknownOperationMeshes(operation)
  }
  if (operation.pauli !== '') {
    console.error('Shuffle operation does not take any Pauli, actually: ' + operation.pauli)
    return unknownOperationMeshes(operation)
  }
  const width = 0.2 * data.qubitRadius
  let meshes: Mesh[] = []
  for (const [i, [source, target]] of mapping.entries()) {
    const sourceRelative = supportRelativeOf(source, gadgetType.realization!)
    const targetRelative = supportRelativeOf(target, gadgetType.realization!)
    targetRelative.t -= shuttleHeightScale * data.qubitRadius
    meshes.push(meshConnecting(sourceRelative, targetRelative, width, colors[i % colors.length]))
  }
  return meshes
}

function shuffleMapping(support: number[]): [number, number][] | null {
  if (support.length === 0 || support.length % 2 !== 0) {
    return null // must be even
  }
  const result: [number, number][] = []
  const source = new Set<number>()
  const target = new Set<number>()
  for (let i = 0; i < support.length; i += 2) {
    const a = support[i]!
    const b = support[i + 1]!
    if (source.has(a) || target.has(b)) {
      return null // duplicate
    }
    source.add(a)
    target.add(b)
    result.push([a, b])
  }
  if (!areSetsEqual(source, target)) {
    return null // not a valid mapping
  }
  return result
}

const shuttleHeightScale = 0.5
const colors = ['#d6938e', '#92abc0', '#aac8a4', '#bca9c1', '#dab786', '#cfcfcf']

function discardMeshes(operation: Operation): Mesh[] {
  if (operation.support.length !== 1) {
    console.error('Discard operation should have exactly one support, actually: ' + operation.support.length)
    return unknownOperationMeshes(operation)
  }
  if (operation.pauli.length !== 0) {
    console.error('Discard operation does not take pauli string, ignored')
  }
  const qubitIndex = operation.support[0]!
  const radius = 1 * data.qubitRadius
  return [
    Mesh.create({
      geometry: { type: 'circle', size: [radius] },
      material: { type: 'standard', color: 'darkgrey' },
      relative: supportRelativeOf(qubitIndex, gadgetType.realization!),
      rotation: { j: Math.PI / 2 },
    }),
  ]
}

function pauliMeshes(operation: Operation): Mesh[] {
  if (operation.support.length !== 1) {
    console.error('Pauli operation should have exactly one support, actually: ' + operation.support.length)
    return unknownOperationMeshes(operation)
  }
  if (operation.pauli.length !== 1) {
    console.error('Pauli operation should take exactly one Pauli, actually: ' + operation.pauli)
    return unknownOperationMeshes(operation)
  }
  const qubitIndex = operation.support[0]!
  const radius = 0.2
  let meshes: Mesh[] = []
  if (operation.pauli === 'X') {
    meshes = meshesOfPauliX('red', radius)
  } else if (operation.pauli === 'Y') {
    meshes = meshesOfPauliY('green', radius)
  } else if (operation.pauli === 'Z') {
    meshes = meshesOfPauliZ('blue', radius)
  } // else if (operation.pauli === 'I'), show nothing
  if (props.gateStyle === 'top') {
    meshes = flattenMeshesForTopView(meshes)
  }
  const scale = 1 * data.qubitRadius
  const move = Position.create(supportRelativeOf(qubitIndex, gadgetType.realization!))
  return transformMeshes(meshes, scale, move)
}

function conditionalPauliMeshes(operation: Operation): Mesh[] {
  if (operation.support.length < 1) {
    console.error('Measure operation should have at least one support, actually: ' + operation.support.length)
    return unknownOperationMeshes(operation)
  }
  const pauliString = parsePauliString(operation.pauli, operation.support.length)
  if (!pauliString) {
    // error already printed
    return unknownOperationMeshes(operation)
  }
  const radius = 1 * data.qubitRadius
  const meshes: Mesh[] = []
  let previousRelative: Position | undefined = undefined
  for (let i = 0; i < operation.support.length; i++) {
    const qubitIndex = operation.support[i]!
    const relative = supportRelativeOf(qubitIndex, gadgetType.realization!)
    if (pauliString.pauli[i] === 'X') {
      let letterMeshes: Mesh[] = meshesOfPauliX('red', 0.2)
      if (props.gateStyle === 'top') letterMeshes = flattenMeshesForTopView(letterMeshes)
      meshes.push(...transformMeshes(letterMeshes, radius, relative))
    } else if (pauliString.pauli[i] === 'Y') {
      let letterMeshes: Mesh[] = meshesOfPauliY('green', 0.2)
      if (props.gateStyle === 'top') letterMeshes = flattenMeshesForTopView(letterMeshes)
      meshes.push(...transformMeshes(letterMeshes, radius, relative))
    } else if (pauliString.pauli[i] === 'Z') {
      let letterMeshes: Mesh[] = meshesOfPauliZ('blue', 0.2)
      if (props.gateStyle === 'top') letterMeshes = flattenMeshesForTopView(letterMeshes)
      meshes.push(...transformMeshes(letterMeshes, radius, relative))
    } // else if (operation.pauli === 'I'), show nothing
    // also show a grey ball at each pauli position to differentiate from Pauli operations
    meshes.push(
      Mesh.create({
        geometry: { type: 'sphere', size: [0.5 * radius] },
        material: { type: 'standard', color: 'black' },
        relative: relative,
      }),
    )
    if (previousRelative) {
      meshes.push(meshConnecting(previousRelative, relative, 0.3 * radius, 'darkgrey'))
    }
    previousRelative = relative
  }
  return meshes
}

function portSelectionOf(portIndex: number): MultiSelectable {
  if (portIndex < gadgetType.inputs.length) {
    return {
      elements: [{ e: { oneofKind: 'port', port: { gid: props.gid, io: { oneofKind: 'input', input: portIndex } } } }],
    }
  } else {
    const outputIndex = portIndex - gadgetType.inputs.length
    return {
      elements: [{ e: { oneofKind: 'port', port: { gid: props.gid, io: { oneofKind: 'output', output: outputIndex } } } }],
    }
  }
}
const portMeshVecProvider = ref<MeshVecProviderPublicInterface>(null!)
function loadPortGroups(meshVecProvider: MeshVecProviderPublicInterface) {
  for (const [portIndex, port] of [...gadgetType.inputs, ...gadgetType.outputs].entries()) {
    const group = new MeshGroup<number>([], portIndex, portSelectionOf)
    const portType = data.portType(port.ptype)
    group.relative = port.relative
    // add custom mesh for the port
    group.meshes.push(...portType.mesh)
    // add data qubits
    for (const qubitRelative of portType.positions) {
      group.meshes.push(
        Mesh.create({
          geometry: { type: 'sphere', size: [data.qubitRadius] },
          material: { type: 'standard', color: 'darkgrey' },
          relative: qubitRelative,
        }),
      )
    }
    meshVecProvider.addGroup(group)
    // add logical observables
  }
}
watch([portMeshVecProvider], ([meshVecProvider]) => {
  if (meshVecProvider) {
    loadPortGroups(meshVecProvider)
    meshVecProvider.finishLoading()
  }
})

function observableSelectionOf(observableKey: string): MultiSelectable {
  const parts = observableKey.split(',')
  assert(parts.length === 2, 'invalid observable key: ' + observableKey)
  const portIndex = parseInt(parts[0]!)
  const observableIndex = parseInt(parts[1]!)
  if (portIndex < gadgetType.inputs.length) {
    return {
      elements: [{ e: { oneofKind: 'observable', observable: { gid: props.gid, io: { oneofKind: 'input', input: portIndex }, observableIndex } } }],
    }
  } else {
    const outputIndex = portIndex - gadgetType.inputs.length
    return {
      elements: [{ e: { oneofKind: 'observable', observable: { gid: props.gid, io: { oneofKind: 'output', output: outputIndex }, observableIndex } } }],
    }
  }
}
const observableMeshVecProvider = ref<MeshVecProviderPublicInterface>(null!)
function loadObservableGroups(meshVecProvider: MeshVecProviderPublicInterface) {
  for (const [portIndex, port] of [...gadgetType.inputs, ...gadgetType.outputs].entries()) {
    const portType = data.portType(port.ptype)
    for (const [observableIndex, observable] of portType.observables.entries()) {
      const observableKey = `${portIndex},${observableIndex}`
      const group = new MeshGroup<string>([], observableKey, observableSelectionOf)
      group.relative = port.relative
      // add custom mesh for the observable
      group.meshes.push(
        Mesh.create({
          geometry: { type: 'box', size: [2 * data.qubitRadius] },
          material: { type: 'standard', color: 'darkgrey' },
          relative: observable.relative,
        }),
      )
      meshVecProvider.addGroup(group)
    }
  }
}
watch([observableMeshVecProvider], ([meshVecProvider]) => {
  if (meshVecProvider) {
    loadObservableGroups(meshVecProvider)
    meshVecProvider.finishLoading()
  }
})
</script>

<template>
  <MeshVecProvider v-if="show" ref="locationMeshVecProvider" :relative="gadget.position"
    :selectable="{ oneofKind: 'locations', gid: props.gid }" />
  <MeshVecProvider v-if="show" ref="lifetimeMeshVecProvider" :relative="gadget.position" />
  <MeshVecProvider v-if="displayMode.showPorts" ref="portMeshVecProvider" :relative="gadget.position"
    :selectable="{ oneofKind: 'ports', gid: props.gid }" />
  <MeshVecProvider v-if="displayMode.showPorts" ref="observableMeshVecProvider" :relative="gadget.position"
    :selectable="{ oneofKind: 'observables', gid: props.gid }" />
</template>
