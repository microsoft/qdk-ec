import { defineComponent, type ComponentPublicInstance, inject, type InjectionKey, type PropType } from 'vue'
import { assert, pos, rpos, siz, rotationOf, colorOf, materialOf, geometryOf } from '@/util'
import * as THREE from 'three'
import { Mesh, Position, Material, type MultiSelectable } from '@/proto/visualizer'
import { RendererInjectionKey, type RendererPublicInterface } from 'troisjs'
import { SelectionManagerInjectionKey, type SelectableProvider, type SelectionManagerPublicInterface } from '@/misc/SelectionManager'
import { InstancedMesh } from 'three'

// group meshes so that they are selected as a whole
export class MeshGroup<T extends string | bigint | number = any> {
  meshes: Mesh[]
  // one should keep userData minimal because they are used to identify the group when selected
  // it should be a number, bigint or serializable object with a consistent order
  key?: T
  selectionOf?: (key: T) => MultiSelectable
  relative?: Position

  constructor(meshes?: Mesh[], key?: T, selectionOf?: (key: T) => MultiSelectable) {
    this.meshes = meshes || []
    this.key = key
    this.selectionOf = selectionOf
  }
}

const standardMaterial = new THREE.MeshStandardMaterial({ color: 0xffffff, side: 2 })

const boxGeometry = new THREE.BoxGeometry(1, 1, 1)
const coneGeometry = new THREE.ConeGeometry(1, 1, 16, 1)
const sphereGeometry = new THREE.SphereGeometry(1, 16, 8)
const torusGeometry = new THREE.TorusGeometry(1, 0.1, 6, 24)
const circleGeometry = new THREE.CircleGeometry(1, 16)

// internally how to access the three object
type MeshPointer =
  | {
      oneofKind: 'instance'
      original: Mesh
      instanced: MyInstancedMesh2
      index: number
    }
  | {
      oneofKind: 'mesh'
      original: Mesh
      mesh: THREE.Mesh
    }

interface MeshGroupInternal {
  // not necessarily in the same order as MeshGroup.meshes
  meshPointers: MeshPointer[]
}

const colorScale = 1.01 // to ensure that the color change is visible despite z-fighting

export interface MeshVecProviderInterface {
  selectable?: SelectableProvider
  addGroup(group: MeshGroup): void
  removeGroup(group?: MeshGroup): void
  setColor(key: number | string | bigint, color: THREE.Color | undefined): void
  extendBoundingBox(key: number | string | bigint, bbox: THREE.Box3): void
  invalidate(): void
  // call this function to tell the selection manager that the objects has been added
  finishLoading(): void

  renderer: RendererPublicInterface
  selectionManager: SelectionManagerPublicInterface
}

export interface MeshVecProviderPublicInterface extends ComponentPublicInstance, MeshVecProviderInterface {}

export const MeshVecProviderInjectionKey: InjectionKey<MeshVecProviderPublicInterface> = Symbol('MeshVecProvider')

export default defineComponent({
  name: 'MeshVecProvider',
  props: {
    selectable: {
      type: Object as PropType<SelectableProvider>,
    },
    relative: {
      type: Object as PropType<Position>,
    },
  },
  inheritAttrs: false,
  setup() {
    const renderer = inject(RendererInjectionKey)!
    const selectionManager = inject(SelectionManagerInjectionKey)!

    return {
      renderer,
      selectionManager,
      groups: new Map<MeshGroup, MeshGroupInternal>(),
      keyToGroup: new Map<any, MeshGroup>(),
      selected: new Set<MeshGroup>(),
      hovered: new Set<MeshGroup>(),
      invalidated: true,
      meshes2: [] as MyInstancedMesh2[],
      otherMeshes: [] as THREE.Mesh[],
    }
  },
  beforeMount() {
    this.selectionManager.registerLoading(this.$.uid)
    this.meshes2.push(new BoxMesh2(this))
    this.meshes2.push(new ConeMesh2(this))
    this.meshes2.push(new SphereMesh2(this))
    this.meshes2.push(new TorusMesh2(this))
    this.meshes2.push(new CircleMesh2(this))
    this.renderer.onBeforeRender(this.beforeRender)
    this.selectionManager.addMeshVecProvider(this!)
  },
  beforeUnmount() {
    this.selectionManager.finishLoading(this.$.uid)
    this.renderer.offBeforeRender(this.beforeRender)
    this.selectionManager.removeMeshVecProvider(this!)
    for (const mesh2 of this.meshes2) {
      if (mesh2.instancedMesh !== undefined) {
        this.renderer.three.scene!.remove(mesh2.instancedMesh)
        mesh2.instancedMesh.dispose()
      }
    }
    this.disposeOtherMeshes()
  },
  provide() {
    return {
      [MeshVecProviderInjectionKey as symbol]: this,
    }
  },
  methods: {
    beforeRender() {
      if (this.invalidated) {
        this.reconstruct()
        this.invalidated = false
      }
    },
    addGroup(group: MeshGroup) {
      assert(group instanceof MeshGroup)
      assert(!this.groups.has(group))
      this.groups.set(group, { meshPointers: [] })
      if (group.key !== undefined) {
        assert(!this.keyToGroup.has(group.key), `duplicate group key: ${group.key}`)
        this.keyToGroup.set(group.key, group)
      }
      this.invalidated = true
    },
    removeGroup(group?: MeshGroup) {
      if (group === undefined) {
        return
      }
      assert(group instanceof MeshGroup)
      assert(this.groups.has(group))
      this.groups.delete(group)
      if (group.key !== undefined) {
        assert(this.keyToGroup.has(group.key), `missing group key: ${group.key}`)
        this.keyToGroup.delete(group.key)
      }
      this.invalidated = true
    },
    filter(sizeOf: (mesh: Mesh) => THREE.Vector3 | undefined): [Mesh, THREE.Vector3, MeshGroup][] {
      return Array.from(this.groups.keys()).flatMap((group) =>
        group.meshes.map((mesh) => [mesh, sizeOf(mesh), group] as [Mesh, THREE.Vector3, MeshGroup]).filter(([_, passed]) => passed),
      )
    },
    invalidate() {
      this.invalidated = true
    },
    reconstruct() {
      for (const groupInternal of this.groups.values()) {
        groupInternal.meshPointers.length = 0
      }
      for (const mesh2 of this.meshes2) {
        this.refreshMeshes(mesh2)
      }
      this.disposeOtherMeshes()
      for (const [mesh, group] of this.incompatibleMeshes()) {
        if (mesh.geometry?.type === 'html') {
          // TODO
        } else {
          // manually construct the geometry and materials
          const geometry = geometryOf(mesh)
          const material = materialOf(mesh)
          console.warn('incompatible mesh might result in poor performance')
          const threeMesh = new THREE.Mesh(geometry, material)
          const position = this.positionOf(mesh, group)
          const quaternion = new THREE.Quaternion().setFromEuler(rotationOf(mesh.rotation))
          threeMesh.position.copy(position)
          threeMesh.quaternion.copy(quaternion)
          threeMesh.userData = group
          this.renderer.three.scene!.add(threeMesh)
          this.otherMeshes.push(threeMesh)
          const groupInternal = this.groups.get(group)!
          groupInternal.meshPointers.push({
            oneofKind: 'mesh',
            original: mesh,
            mesh: threeMesh,
          })
        }
      }
      this.selectionManager.finishLoading(this.$.uid)
    },
    positionOf(mesh: Mesh, group: MeshGroup): THREE.Vector3 {
      const position = pos(mesh.relative).add(rpos(this.relative))
      if (group.relative !== undefined) {
        position.add(rpos(group.relative))
      }
      return position
    },
    refreshMeshes(mesh2: MyInstancedMesh2) {
      const meshes = this.filter(mesh2.compatibleSizeOf.bind(mesh2))
      mesh2.groups.length = 0
      mesh2.reserveInstances(meshes.length)
      if (mesh2.instancedMesh !== undefined) {
        mesh2.instancedMesh.count = meshes.length
      }
      for (const [i, [mesh, size, group]] of meshes.entries()) {
        const position = this.positionOf(mesh, group)
        const quaternion = new THREE.Quaternion().setFromEuler(rotationOf(mesh.rotation))
        const matrix = new THREE.Matrix4().compose(position, quaternion, size)
        mesh2.instancedMesh!.setMatrixAt(i, matrix)
        mesh2.instancedMesh!.setColorAt(i, colorOf(mesh))
        mesh2.groups.push(group)
        const groupInternal = this.groups.get(group)!
        groupInternal.meshPointers.push({
          oneofKind: 'instance',
          original: mesh,
          instanced: mesh2,
          index: i,
        })
      }
      if (mesh2.instancedMesh?.instanceColor != null) {
        mesh2.instancedMesh.instanceColor.needsUpdate = true
      }
      if (mesh2.instancedMesh?.instanceMatrix != null) {
        mesh2.instancedMesh.instanceMatrix.needsUpdate = true
      }
    },
    setColor(key: number | string | bigint, color: THREE.Color | undefined) {
      const group = this.keyToGroup.get(key)!
      if (!group) {
        // console.warn(`missing group for key=${key}`)
        return
      }
      const groupInternal = this.groups.get(group)!
      for (const meshPointer of groupInternal.meshPointers) {
        const originColor = colorOf(meshPointer.original)
        const newColor = color === undefined ? originColor : new THREE.Color().lerpColors(color, originColor, 0.4)
        if (meshPointer.oneofKind === 'instance') {
          meshPointer.instanced.instancedMesh!.setColorAt(meshPointer.index, newColor)
          const position = this.positionOf(meshPointer.original, group)
          const quaternion = new THREE.Quaternion().setFromEuler(rotationOf(meshPointer.original.rotation))
          const size = meshPointer.instanced.compatibleSizeOf(meshPointer.original)!
          if (color !== undefined) {
            size.multiplyScalar(colorScale)
          }
          const newMatrix = new THREE.Matrix4().compose(position, quaternion, size)
          meshPointer.instanced.instancedMesh!.setMatrixAt(meshPointer.index, newMatrix)
          meshPointer.instanced.instancedMesh!.instanceColor!.needsUpdate = true
          meshPointer.instanced.instancedMesh!.instanceMatrix!.needsUpdate = true
        } else if (meshPointer.oneofKind === 'mesh') {
          const material = meshPointer.mesh.material as THREE.Material
          ;(material as any).color! = newColor
          const scale = color === undefined ? 1 : colorScale
          meshPointer.mesh.scale.set(scale, scale, scale)
          material.needsUpdate = true
        }
      }
    },
    extendBoundingBox(key: number | string | bigint, bbox: THREE.Box3) {
      const group = this.keyToGroup.get(key)!
      if (!group) {
        // console.warn(`missing group for key=${key}`)
        return
      }
      const groupInternal = this.groups.get(group)!
      for (const meshPointer of groupInternal.meshPointers) {
        if (meshPointer.oneofKind === 'instance') {
          const instancedMesh = meshPointer.instanced.instancedMesh!
          if (!instancedMesh.geometry.boundingBox) {
            instancedMesh.geometry.computeBoundingBox()
          }
          const instanceBox = instancedMesh.geometry.boundingBox!.clone()
          const instanceMatrix = new THREE.Matrix4()
          instancedMesh.getMatrixAt(meshPointer.index, instanceMatrix)
          instanceBox.applyMatrix4(instanceMatrix)
          bbox.union(instanceBox)
        } else if (meshPointer.oneofKind === 'mesh') {
          bbox.expandByObject(meshPointer.mesh)
        }
      }
    },
    incompatibleMeshes(): [Mesh, MeshGroup][] {
      return Array.from(this.groups.keys()).flatMap((group) =>
        group.meshes
          .map((mesh) => [mesh, group] as [Mesh, MeshGroup])
          .filter(([mesh, _]) => {
            for (const mesh2 of this.meshes2) {
              if (mesh2.isCompatible(mesh)) {
                return false
              }
            }
            return true
          }),
      )
    },
    disposeOtherMeshes() {
      for (const mesh of this.otherMeshes) {
        this.renderer.three.scene!.remove(mesh)
        mesh.geometry.dispose()
        ;(mesh.material as THREE.Material).dispose()
      }
      this.otherMeshes.length = 0
    },
    finishLoading() {
      this.selectionManager.finishLoading(this.$.uid)
    },
  },
  render() {
    return this.$slots.default ? this.$slots.default() : []
  },
  __hmrId: 'MeshVecProvider',
})

function isStandardMaterial(material?: Material): boolean {
  if (material === undefined) {
    return true
  }
  return material.type === 'standard' || material.type === ''
}

export class MyInstancedMesh2 {
  meshVecProvider: MeshVecProviderPublicInterface
  groups: MeshGroup[] // to get group object from index
  geometry: THREE.BufferGeometry
  instancedMesh?: InstancedMesh
  capacity: number

  constructor(geometry: THREE.BufferGeometry, meshVecProvider: MeshVecProviderPublicInterface) {
    this.capacity = 0
    this.geometry = geometry
    this.instancedMesh = undefined
    this.meshVecProvider = meshVecProvider
    this.groups = []
  }
  isCompatible(_mesh: Mesh): boolean {
    assert(false, 'not implemented')
  }
  compatibleSizeOf(_mesh: Mesh): THREE.Vector3 | undefined {
    assert(false, 'not implemented')
  }
  selectionOf(index: number): MultiSelectable {
    const group = this.groups[index]
    if (group == undefined || group.selectionOf == undefined) {
      return { elements: [] }
    }
    return group.selectionOf(group.key)
  }
  reserveInstances(count: number) {
    if (count > this.capacity) {
      if (this.instancedMesh !== undefined) {
        this.meshVecProvider.renderer.three.scene!.remove(this.instancedMesh)
        this.instancedMesh.dispose()
      }
      this.instancedMesh = new InstancedMesh(this.geometry, standardMaterial, count)
      this.instancedMesh.userData = this
      this.meshVecProvider.renderer.three.scene!.add(this.instancedMesh)
      this.capacity = count
      this.meshVecProvider.selectionManager.registerLoading(this.meshVecProvider.$.uid)
    }
  }
}

export class BoxMesh2 extends MyInstancedMesh2 {
  constructor(meshVecProvider: MeshVecProviderPublicInterface) {
    super(boxGeometry, meshVecProvider)
  }
  override isCompatible(mesh: Mesh): boolean {
    if (mesh.geometry?.type !== 'box' || !isStandardMaterial(mesh.material)) {
      return false
    }
    return isWithoutArgs(mesh) && mesh.geometry.size.length <= 3 && mesh.geometry.size.length >= 1
  }
  override compatibleSizeOf(mesh: Mesh): THREE.Vector3 | undefined {
    if (this.isCompatible(mesh)) {
      const dt = Math.abs(mesh.geometry!.size[0]!)
      const di = Math.abs(mesh.geometry!.size[1] || dt)
      const dj = Math.abs(mesh.geometry!.size[2] || di)
      return siz(Position.create({ t: dt, i: di, j: dj }))
    }
  }
}

export class ConeMesh2 extends MyInstancedMesh2 {
  constructor(meshVecProvider: MeshVecProviderPublicInterface) {
    super(coneGeometry, meshVecProvider)
  }
  override isCompatible(mesh: Mesh): boolean {
    if (mesh.geometry?.type !== 'cone' || !isStandardMaterial(mesh.material)) {
      return false
    }
    return isWithoutArgs(mesh) && mesh.geometry.size.length <= 2 && mesh.geometry.size.length >= 1
  }
  override compatibleSizeOf(mesh: Mesh): THREE.Vector3 | undefined {
    if (this.isCompatible(mesh)) {
      const radius = Math.abs(mesh.geometry!.size[0]!)
      const height = Math.abs(mesh.geometry!.size[1] || radius)
      return siz(Position.create({ t: height, i: radius, j: radius }))
    }
  }
}

export class SphereMesh2 extends MyInstancedMesh2 {
  constructor(meshVecProvider: MeshVecProviderPublicInterface) {
    super(sphereGeometry, meshVecProvider)
  }
  override isCompatible(mesh: Mesh): boolean {
    if (mesh.geometry?.type !== 'sphere' || !isStandardMaterial(mesh.material)) {
      return false
    }
    return isWithoutArgs(mesh) && mesh.geometry.size.length === 1
  }
  override compatibleSizeOf(mesh: Mesh): THREE.Vector3 | undefined {
    if (this.isCompatible(mesh)) {
      const radius = Math.abs(mesh.geometry!.size[0]!)
      return siz(Position.create({ t: radius, i: radius, j: radius }))
    }
  }
}

export class TorusMesh2 extends MyInstancedMesh2 {
  constructor(meshVecProvider: MeshVecProviderPublicInterface) {
    super(torusGeometry, meshVecProvider)
  }
  override isCompatible(mesh: Mesh): boolean {
    if (mesh.geometry?.type !== 'torus' || !isStandardMaterial(mesh.material)) {
      return false
    }
    return isWithoutArgs(mesh) && mesh.geometry.size.length === 1
  }
  override compatibleSizeOf(mesh: Mesh): THREE.Vector3 | undefined {
    if (this.isCompatible(mesh)) {
      const radius = Math.abs(mesh.geometry!.size[0]!)
      return siz(Position.create({ t: radius, i: radius, j: radius }))
    }
  }
}

export class CircleMesh2 extends MyInstancedMesh2 {
  constructor(meshVecProvider: MeshVecProviderPublicInterface) {
    super(circleGeometry, meshVecProvider)
  }
  override isCompatible(mesh: Mesh): boolean {
    if (mesh.geometry?.type !== 'circle' || !isStandardMaterial(mesh.material)) {
      return false
    }
    return isWithoutArgs(mesh) && mesh.geometry.size.length === 1
  }
  override compatibleSizeOf(mesh: Mesh): THREE.Vector3 | undefined {
    if (this.isCompatible(mesh)) {
      const radius = Math.abs(mesh.geometry!.size[0]!)
      return siz(Position.create({ t: radius, i: 1, j: radius }))
    }
  }
}

export function isWithoutArgs(mesh: Mesh): boolean {
  return mesh.geometry?.geometryProps === '' && mesh.material?.materialProps === ''
}
