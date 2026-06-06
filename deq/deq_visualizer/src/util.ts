import { Position, Euler, Mesh } from '@/proto/visualizer'
import * as THREE from 'three'
import { customRef, type Ref, type InjectionKey } from 'vue'

// Custom deep equal function that handles BigInt values
export function deepEqual(a: any, b: any): boolean {
  // Same reference or primitives (including BigInt)
  if (a === b) return true

  // Handle null/undefined
  if (a == null || b == null) return a === b

  // Handle BigInt specifically
  if (typeof a === 'bigint' || typeof b === 'bigint') {
    return a === b
  }

  // Handle different types
  if (typeof a !== typeof b) return false

  // Handle arrays
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false
    for (let i = 0; i < a.length; i++) {
      if (!deepEqual(a[i], b[i])) return false
    }
    return true
  }

  // One is array, one is not
  if (Array.isArray(a) || Array.isArray(b)) return false

  // Handle objects
  if (typeof a === 'object' && typeof b === 'object') {
    const keysA = Object.keys(a)
    const keysB = Object.keys(b)
    if (keysA.length !== keysB.length) return false
    for (const key of keysA) {
      if (!Object.prototype.hasOwnProperty.call(b, key)) return false
      if (!deepEqual(a[key], b[key])) return false
    }
    return true
  }

  return false
}

// Convert BigInt to string for Vue template rendering (Vue 3 reactivity corrupts BigInt)
export function bigintToString(value: bigint): string {
  return value.toString()
}

// Convert array of BigInt to array of strings for Vue template rendering
export function bigintArrayToString(arr: bigint[] | undefined | null): string[] {
  if (!arr) return []
  return arr.map(n => n.toString())
}

export function assert(condition: boolean, msg?: string): asserts condition {
  if (!condition) {
    throw new Error(msg)
  }
}

export function bestEffortJsonParse(str: string): any {
  try {
    return JSON.parse(str)
  } catch (_e) {
    return {}
  }
}

export function pos(position?: Position): THREE.Vector3 {
  if (position === undefined) {
    return new THREE.Vector3(0, 0, 0)
  }
  return new THREE.Vector3(position.j, position.t, position.i)
}

export function siz(size?: Position): THREE.Vector3 {
  if (size === undefined) {
    return new THREE.Vector3(0, 0, 0)
  }
  return pos(size).addScaledVector(pos(), -1)
}

// relative position
export function rpos(position?: Position): THREE.Vector3 {
  return siz(position)
}

export function rotationOf(rotation?: Euler): THREE.Euler {
  if (rotation === undefined) {
    return new THREE.Euler(0, 0, 0)
  }
  return new THREE.Euler(rotation.j, rotation.t, rotation.i)
}

export function originRotationOf(rotation: THREE.Euler): Euler {
  return { t: rotation.y, i: rotation.z, j: rotation.x }
}

export function addTo(a: Position, b: Position) {
  a.t += b.t ?? 0
  a.i += b.i ?? 0
  a.j += b.j ?? 0
}

export function addPos(a: Position, b: Position): Position {
  return {
    t: a.t + b.t,
    i: a.i + b.i,
    j: a.j + b.j,
  }
}

export function divideBy(position: Position, divisor: number) {
  position.t /= divisor
  position.i /= divisor
  position.j /= divisor
}

export function biasedBy(position: Position, bias: Position): Position {
  return {
    t: position.t + bias.t,
    i: position.i + bias.i,
    j: position.j + bias.j,
  }
}

export function meshVecBiasedBy(meshes: Mesh[], bias: Position): Mesh[] {
  return meshes.map((mesh) => ({ ...mesh, relative: biasedBy(mesh.relative || Position.create(), bias) }))
}

export function cached_dict(_target: any, propertyKey: string, descriptor: PropertyDescriptor) {
  if (!descriptor.value || typeof descriptor.value !== 'function') {
    throw new Error('The cached_dict can be used only on a function property')
  }
  const original = descriptor.value
  const resolve = (cache: any, obj: any, key: any) => {
    cache[key] = original.apply(obj, [key])
    return cache[key]
  }
  return {
    value(key: any): any {
      const symbol = `@cached_dict#${propertyKey}`
      const that = this as any
      if (!that[symbol]) {
        that[symbol] = {}
      }
      const cache = that[symbol]
      if (!(key in cache)) resolve(cache, that, key)
      return cache[key]
    },
  }
}

export function areSetsEqual(set1: Set<any>, set2: Set<any>): boolean {
  if (set1.size !== set2.size) {
    return false
  }
  return [...set1].every((item) => set2.has(item))
}

export function colorOf(mesh: Mesh): THREE.Color {
  return new THREE.Color(mesh.material?.color || '#FF0000')
}

export function materialOf(mesh: Mesh): THREE.Material {
  if (mesh.material?.type === 'standard') {
    const materialPros = bestEffortJsonParse(mesh.material.materialProps)
    const side = 'side' in materialPros ? materialPros.side : 2 // by default both side
    return new THREE.MeshStandardMaterial({ color: colorOf(mesh), side: side, transparent: true })
  } else if (mesh.material?.type === 'transmission') {
    const materialPros = bestEffortJsonParse(mesh.material.materialProps)
    const thickness = 'thickness' in materialPros ? materialPros.thickness : 0.5
    const anisotropy = 'anisotropy' in materialPros ? materialPros.anisotropy : 0.5
    return new THREE.MeshPhysicalMaterial({
      color: colorOf(mesh),
      transmission: 1,
      anisotropy: anisotropy,
      thickness: thickness,
      side: 2,
      transparent: true,
    })
  }
  // default to basic
  if (mesh.material?.type === 'basic') {
    console.error(`unsupported material type for incompatible mesh: ${JSON.stringify(mesh.material)}, default to basic`)
  }
  return new THREE.MeshBasicMaterial({ color: colorOf(mesh) })
}

export function geometryOf(mesh: Mesh): THREE.BufferGeometry {
  if (mesh.geometry?.type === 'capsule') {
    const radius = Math.abs(mesh.geometry.size[0] || 0.1)
    const height = Math.abs(mesh.geometry.size[1] || radius)
    const geometryPros = bestEffortJsonParse(mesh.geometry.geometryProps)
    const capSegments = 'capSegments' in geometryPros ? geometryPros.capSegments : 4
    const radialSegments = 'radialSegments' in geometryPros ? geometryPros.radialSegments : 8
    const heightSegments = 'heightSegments' in geometryPros ? geometryPros.heightSegments : 1
    return new THREE.CapsuleGeometry(radius, height, capSegments, radialSegments, heightSegments)
  } else if (mesh.geometry?.type === 'circle') {
    const radius = Math.abs(mesh.geometry.size[0] || 0.1)
    const thetaStart = mesh.geometry.size[1] || 0
    const thetaLength = mesh.geometry.size[2] || 2 * Math.PI
    const geometryPros = bestEffortJsonParse(mesh.geometry.geometryProps)
    const segments = 'segments' in geometryPros ? geometryPros.segments : 16
    return new THREE.CircleGeometry(radius, segments, thetaStart, thetaLength)
  } else if (mesh.geometry?.type === 'cone') {
    const radius = Math.abs(mesh.geometry.size[0] || 0.1)
    const height = Math.abs(mesh.geometry.size[1] || radius)
    const thetaStart = mesh.geometry.size[2] || 0
    const thetaLength = mesh.geometry.size[3] || 2 * Math.PI
    const geometryPros = bestEffortJsonParse(mesh.geometry.geometryProps)
    const radialSegments = 'radialSegments' in geometryPros ? geometryPros.radialSegments : 16
    const heightSegments = 'heightSegments ' in geometryPros ? geometryPros.heightSegments : 1
    const openEnded = 'openEnded' in geometryPros ? geometryPros.openEnded : false
    return new THREE.ConeGeometry(radius, height, radialSegments, heightSegments, openEnded, thetaStart, thetaLength)
  } else if (mesh.geometry?.type === 'sphere') {
    const radius = Math.abs(mesh.geometry.size[0] || 0.1)
    const geometryPros = bestEffortJsonParse(mesh.geometry.geometryProps)
    const widthSegments = 'widthSegments' in geometryPros ? geometryPros.widthSegments : 16
    const heightSegments = 'heightSegments' in geometryPros ? geometryPros.heightSegments : 8
    return new THREE.SphereGeometry(radius, widthSegments, heightSegments)
  } else if (mesh.geometry?.type === 'torus') {
    const radius = Math.abs(mesh.geometry.size[0] || 0.1)
    const tube = Math.abs(mesh.geometry.size[1] || radius * 0.1)
    const geometryPros = bestEffortJsonParse(mesh.geometry.geometryProps)
    const radialSegments = 'radialSegments ' in geometryPros ? geometryPros.radialSegments : 6
    const tubularSegments = 'tubularSegments ' in geometryPros ? geometryPros.tubularSegments : 24
    const arc = 'arc' in geometryPros ? geometryPros.arc : Math.PI * 2
    return new THREE.TorusGeometry(radius, tube, radialSegments, tubularSegments, arc)
  }
  // default to box
  if (mesh.geometry?.type !== 'box') {
    console.error(`unsupported geometry type for incompatible mesh: ${JSON.stringify(mesh.geometry)}, default to box`)
  }
  const dt = Math.abs(mesh.geometry?.size[0] || 0.1)
  const di = Math.abs(mesh.geometry?.size[1] || dt)
  const dj = Math.abs(mesh.geometry?.size[2] || di)
  const size = siz(Position.create({ t: dt, i: di, j: dj }))
  return new THREE.BoxGeometry(size.x, size.y, size.z)
}

export function jsonValueRef<T>(initial: T): Ref<T> {
  return customRef((track, trigger) => {
    return {
      get() {
        track()
        return initial
      },
      set(newValue) {
        if (!deepEqual(initial, newValue)) {
          initial = newValue
          trigger()
        }
      },
    }
  })
}

export interface PartialDisplayMode {
  showBlock?: boolean
  showRealization?: boolean
  showCheckModel?: boolean
  showErrorModel?: boolean
  showPorts?: boolean
}

export interface VisualizerPublicInterface {
  cameraPosition: Ref<THREE.Vector3>
  orbitTarget: Ref<THREE.Vector3>
  dragCamera: (deltaX: number, deltaY: number) => void
  setGlobalDisplayMode: (displayMode: PartialDisplayMode) => void
  setDisplayMode: (gid: bigint, displayMode: PartialDisplayMode) => void
}
export const VisualizerInjectionKey: InjectionKey<VisualizerPublicInterface> = Symbol('Visualizer')
