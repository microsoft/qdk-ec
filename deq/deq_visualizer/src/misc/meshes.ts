import { Mesh, Position, Realization } from '@/proto/visualizer'
import * as THREE from 'three'
import { pos, originRotationOf } from '../util'

// Quaternion that rotates 90° around the j-axis (X in THREE.js),
// mapping the t-j plane (front view) to the i-j plane (top view).
const topViewQuat = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2)

export function meshesOfPauliX(color: string, radius: number = 0.2): Mesh[] {
  return [
    Mesh.create({
      geometry: { type: 'box', size: [2.351, radius, radius] },
      material: { type: 'standard', color },
      relative: {},
      rotation: { i: 0.553 },
    }),
    Mesh.create({
      geometry: { type: 'box', size: [2.351, radius, radius] },
      material: { type: 'standard', color },
      relative: {},
      rotation: { i: -0.553 },
    }),
  ]
}

export function meshesOfPauliY(color: string, radius: number = 0.2): Mesh[] {
  return [
    Mesh.create({
      geometry: { type: 'box', size: [1.3, radius, radius] },
      material: { type: 'standard', color },
      relative: { t: 0.5, j: -0.398 },
      rotation: { i: 0.553 },
    }),
    Mesh.create({
      geometry: { type: 'box', size: [1.3, radius, radius] },
      material: { type: 'standard', color },
      relative: { t: 0.5, j: 0.398 },
      rotation: { i: -0.553 },
    }),
    Mesh.create({
      geometry: { type: 'box', size: [1.1, radius, radius] },
      material: { type: 'standard', color },
      relative: { t: -0.45 },
    }),
  ]
}

export function meshesOfPauliZ(color: string, radius: number = 0.2): Mesh[] {
  return [
    Mesh.create({
      geometry: { type: 'box', size: [1.236, radius, radius] },
      material: { type: 'standard', color },
      relative: { t: 0.9 },
      rotation: { i: Math.PI / 2 },
    }),
    Mesh.create({
      geometry: { type: 'box', size: [1.236, radius, radius] },
      material: { type: 'standard', color },
      relative: { t: -0.9 },
      rotation: { i: Math.PI / 2 },
    }),
    Mesh.create({
      geometry: { type: 'box', size: [2, radius, radius] },
      material: { type: 'standard', color },
      rotation: { i: -0.553 },
    }),
  ]
}

export function transformMeshes(meshes: Mesh[], scale: number = 1, move: Position = Position.create()): Mesh[] {
  return meshes.map((m) => {
    const newSize = m.geometry?.size.map((s) => s * scale)
    const newRelative = Position.create({
      t: (m.relative?.t || 0) * scale + (move?.t || 0),
      i: (m.relative?.i || 0) * scale + (move?.i || 0),
      j: (m.relative?.j || 0) * scale + (move?.j || 0),
    })
    return Mesh.create({
      ...m,
      geometry: { ...m.geometry, size: newSize },
      relative: newRelative,
    })
  })
}

// Remap meshes from the t-j plane (front view) to the i-j plane (top view)
// by applying a 90° rotation around the j-axis (X in THREE.js).
export function flattenMeshesForTopView(meshes: Mesh[]): Mesh[] {
  return meshes.map((m) => {
    // Rotate position
    const p = new THREE.Vector3(m.relative?.j || 0, m.relative?.t || 0, m.relative?.i || 0)
    p.applyQuaternion(topViewQuat)

    // Compose rotation: existing mesh rotation * topView rotation
    const existingEuler = new THREE.Euler(m.rotation?.j || 0, m.rotation?.t || 0, m.rotation?.i || 0)
    const existingQuat = new THREE.Quaternion().setFromEuler(existingEuler)
    const composedQuat = new THREE.Quaternion().multiplyQuaternions(topViewQuat, existingQuat)
    const composedEuler = new THREE.Euler().setFromQuaternion(composedQuat)

    return Mesh.create({
      ...m,
      relative: Position.create({ t: p.y, i: p.z, j: p.x }),
      rotation: originRotationOf(composedEuler),
    })
  })
}

export function lengthBetween(a: Position, b: Position): number {
  const dt = a.t - b.t
  const di = a.i - b.i
  const dj = a.j - b.j
  return Math.sqrt(dt * dt + di * di + dj * dj)
}

const unitUpVector = new THREE.Vector3(0, 1, 0)

export function eulerRotationOf(a: Position, b: Position): THREE.Euler {
  // create a rotation such that a vector pointing upwards would rotate to a direction that points from a to b
  const target = pos(b).sub(pos(a)).normalize()
  const quaternion = new THREE.Quaternion()
  quaternion.setFromUnitVectors(unitUpVector, target).normalize()
  return new THREE.Euler().setFromQuaternion(quaternion)
}

export const colorOfPauli: { [key: string]: string } = {
  X: 'red',
  Y: 'green',
  Z: 'blue',
  '+X': 'red',
  '+Y': 'green',
  '+Z': 'blue',
  '-X': 'darkred',
  '-Y': 'darkgreen',
  '-Z': 'darkblue',
}

export function meshConnecting(a: Position, b: Position, width: number, color: string = 'black'): Mesh {
  const length = lengthBetween(a, b)
  const connectCenter = Position.create({
    t: (a.t + b.t) / 2,
    i: (a.i + b.i) / 2,
    j: (a.j + b.j) / 2,
  })
  return Mesh.create({
    geometry: { type: 'box', size: [length, width] },
    material: { type: 'standard', color },
    relative: connectCenter,
    rotation: originRotationOf(eulerRotationOf(a, b)),
  })
}

export type MeasurePauli = {
  observable: PauliString
  correction?: PauliString
  labels?: string[]
}

export class PauliString {
  sign: '+' | '-'
  pauli: string

  constructor(sign: '+' | '-', pauli: string) {
    this.sign = sign
    this.pauli = pauli
  }

  signedCharAt(index: number): string {
    return this.sign + this.pauli[index]
  }
}

export function parseMeasurementPauli(pauli: string, n: number): MeasurePauli | undefined {
  const observable = parsePauliString(pauli, n)
  if (!observable) {
    // error already printed
    return
  }
  pauli = pauli.slice(1 + n)
  let correction: PauliString | undefined = undefined
  if (pauli[0] === '%') {
    // this is the optional correction field
    correction = parsePauliString(pauli.slice(1), n)
    if (!correction) {
      // error already printed
      return
    }
    pauli = pauli.slice(2 + n)
  }
  let labels: string[] | undefined = undefined
  if (pauli.length > 0) {
    // this is the optional labels field
    labels = JSON.parse(pauli)
    if (!Array.isArray(labels) || labels.length !== n) {
      console.error('Labels should be a JSON array of strings with length ' + n + ', actually: ' + pauli)
      return
    }
  }
  return { observable, correction, labels }
}

export function parsePauliString(pauli: string, n: number): PauliString | undefined {
  if (pauli.length < 1 + n) {
    console.error('Pauli string should have length at least ' + (1 + n) + ', actually: ' + pauli)
    return
  }
  const sign = pauli[0]
  if (sign !== '+' && sign !== '-') {
    console.error('Pauli string should start with + or -, actually: ' + pauli)
    return
  }
  const pauliString = pauli.slice(1, 1 + n)
  for (const c of pauliString) {
    if (c !== 'I' && c !== 'X' && c !== 'Y' && c !== 'Z') {
      console.error('Pauli string should only contain I, X, Y, Z after the sign, actually: ' + pauli)
      return
    }
  }
  return new PauliString(sign, pauliString)
}

export function supportRelativeOf(support: number | bigint, realization: Realization): Position {
  const index = support as unknown as number
  if (realization.positions[index]) {
    const { i, j } = realization.positions[index]
    return Position.create({ i, j })
  }
  return Position.create({ i: index, j: 0 })
}
