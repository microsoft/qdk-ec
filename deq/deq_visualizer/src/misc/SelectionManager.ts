import { defineComponent, type ComponentPublicInstance, inject, type InjectionKey, type PropType, type Ref, h, ref } from 'vue'
import * as THREE from 'three'
import { assert, VisualizerInjectionKey, deepEqual, pos, rpos, geometryOf, rotationOf } from '@/util'
import { type MeshVecProviderPublicInterface } from '@/misc/MeshVecProvider'
import { RendererInjectionKey } from 'troisjs'
import { MyInstancedMesh2, MeshGroup } from '@/misc/MeshVecProvider'
import { MultiSelectable, type Selectable, Mesh, Position } from '@/proto/visualizer'
import Tooltip from '@/misc/Tooltip.vue'
import * as JSON from '@ungap/raw-json'
import { VisualizerDataInjectionKey } from './VisualizerData'

/*
Capability: select/hover one or multiple elements in the scene both interactively
and programmatically.

Programmability is required because sometimes we want to show user a subset of
elements, e.g., the region in a window decoder, from the Python side instead of
manually type in all the ids. Given this requirement, we need to have a global
interface to set the selection/hover state instead of letting each component
managing its own state.

Scalability is also required because we sometimes need to display very large graphs
consisting of millions of elements. In such case, we cannot afford to iterate over
all elements to check if they identify themselves as selected/hovered. Therefore, we
need to scope down and build maps to directly access the element that are affected.

SelectionManager is a component that provides such capability at scale. It injects
the interface to be used by its descendants, and any MeshVecProvider under its tree
can register to the SelectionManager
*/

export type SelectableProvider =
  | {
      oneofKind: 'gadgets'
    }
  | {
      // the locations are grouped by gadgets
      oneofKind: 'locations'
      gid: bigint
    }
  | {
      oneofKind: 'ports'
      gid: bigint
    }
  | {
      oneofKind: 'observables'
      gid: bigint
    }
  | {
      oneofKind: 'checks'
      cid: bigint
    }
  | {
      oneofKind: 'errors'
      eid: bigint
    }

export interface SelectionManagerInterface {
  addMeshVecProvider(provider: MeshVecProviderPublicInterface): void
  removeMeshVecProvider(provider: MeshVecProviderPublicInterface): void
  select(element: MultiSelectable): void
  hover(element: MultiSelectable, fromRelationLink?: boolean): void
  // tell the selection manager when the objects are all loaded so that it can
  // apply the initial selection/hover state
  state: { selected: Ref<MultiSelectable>; hovered: Ref<MultiSelectable> }
  registerLoading(uid: number): void
  finishLoading(uid: number): void
  objectsOf(selectable: MultiSelectable): [MeshVecProviderPublicInterface, number | bigint | string][]
  getSelected(): MultiSelectable
  getHovered(): MultiSelectable
  gidsOf(selectable: MultiSelectable): Set<bigint>
}

export interface SelectionManagerPublicInterface extends ComponentPublicInstance, SelectionManagerInterface {}

export const SelectionManagerInjectionKey: InjectionKey<SelectionManagerPublicInterface> = Symbol('SelectionManager')

export default defineComponent({
  name: 'SelectionManager',
  props: {
    hoverColor: {
      type: Object as PropType<THREE.Color>,
      default: () => new THREE.Color('#6FDFDF'),
    },
    selectColor: {
      type: Object as PropType<THREE.Color>,
      default: () => new THREE.Color('#4B7BE5'),
    },
    state: {
      type: Object as PropType<{ selected: Ref<MultiSelectable>; hovered: Ref<MultiSelectable> }>,
      required: true,
    },
    multiSelecting: {
      type: Boolean,
      default: false,
    },
  },
  inheritAttrs: false,
  setup(props) {
    const data = inject(VisualizerDataInjectionKey)!
    const renderer = inject(RendererInjectionKey)!
    const visualizer = inject(VisualizerInjectionKey)!
    assert(visualizer != undefined, 'missing visualizer injection')
    const raycaster = new THREE.Raycaster()

    return {
      data,
      visualizer,
      gadgets: null as MeshVecProviderPublicInterface | null,
      locations: new Map<bigint, MeshVecProviderPublicInterface>(),
      ports: new Map<bigint, MeshVecProviderPublicInterface>(),
      observables: new Map<bigint, MeshVecProviderPublicInterface>(),
      checks: new Map<bigint, MeshVecProviderPublicInterface>(),
      errors: new Map<bigint, MeshVecProviderPublicInterface>(),
      renderer,
      raycaster,
      selected: props.state.selected,
      hovered: props.state.hovered,
      text: ref(''),
      loading: new Set<number>(),
      // Tooltip position override (undefined = use mouse position)
      tooltipX: ref<number | undefined>(undefined),
      tooltipY: ref<number | undefined>(undefined),
      // Wireframe rendering state (managed directly, not via Vue component)
      wireframeObjects: [] as THREE.Object3D[],
      wireframeBoundingBox: null as THREE.Box3 | null,
      wireframeMaterial: new THREE.LineBasicMaterial({ 
        color: 0x000000, 
        linewidth: 2,
        transparent: true,
        opacity: 0.8,
      }),
    }
  },
  mounted() {
    // hover and click handlers
    let mousedown_clientX: number | undefined = undefined
    let mousedown_clientY: number | undefined = undefined
    let is_mouse_currently_down = false
    let is_right_mouse_currently_down = false
    this.renderer.canvas.addEventListener('mousedown', (event) => {
      mousedown_clientX = event.clientX
      mousedown_clientY = event.clientY
      is_mouse_currently_down = true
      if (event.button === 2) {
        is_right_mouse_currently_down = true
      }
    })
    this.renderer.canvas.addEventListener('mouseup', (event) => {
      if (mousedown_clientX == event.clientX && mousedown_clientY == event.clientY) {
        this.onMouseChange(event, true)
      }
      is_mouse_currently_down = false
      is_right_mouse_currently_down = false
    })
    this.renderer.canvas.addEventListener('mousemove', (event) => {
      // to prevent triggering hover while moving camera
      if (is_right_mouse_currently_down) {
        this.visualizer.dragCamera(event.movementX, event.movementY)
      }
      // Skip hover detection while dragging a pane
      if (!is_mouse_currently_down && !this.data.isDraggingPane.value) {
        this.onMouseChange(event, false)
      }
    })
  },
  provide() {
    return {
      [SelectionManagerInjectionKey as symbol]: this,
    }
  },
  methods: {
    addMeshVecProvider(provider: MeshVecProviderPublicInterface) {
      if (!provider.selectable) {
        return
      }
      switch (provider.selectable.oneofKind) {
        case 'gadgets': {
          assert(this.gadgets === null, `duplicate gadgets provider`)
          this.gadgets = provider
          break
        }
        case 'locations': {
          const gid = provider.selectable.gid
          assert(this.locations.has(gid) === false, `duplicate locations provider for gid=${gid}`)
          this.locations.set(gid, provider)
          break
        }
        case 'ports': {
          const gid = provider.selectable.gid
          assert(this.ports.has(gid) === false, `duplicate ports provider for gid=${gid}`)
          this.ports.set(gid, provider)
          break
        }
        case 'observables': {
          const gid = provider.selectable.gid
          assert(this.observables.has(gid) === false, `duplicate observables provider for gid=${gid}`)
          this.observables.set(gid, provider)
          break
        }
        case 'checks': {
          const cid = provider.selectable.cid
          assert(this.checks.has(cid) === false, `duplicate checks provider for cid=${cid}`)
          this.checks.set(cid, provider)
          break
        }
        case 'errors': {
          const eid = provider.selectable.eid
          assert(this.errors.has(eid) === false, `duplicate errors provider for eid=${eid}`)
          this.errors.set(eid, provider)
          break
        }
      }
    },
    removeMeshVecProvider(provider: MeshVecProviderPublicInterface) {
      if (!provider.selectable) {
        return
      }
      switch (provider.selectable.oneofKind) {
        case 'gadgets':
          assert(this.gadgets === provider)
          this.gadgets = null
          break
        case 'locations':
          assert(this.locations.has(provider.selectable.gid))
          this.locations.delete(provider.selectable.gid)
          break
        case 'ports':
          assert(this.ports.has(provider.selectable.gid))
          this.ports.delete(provider.selectable.gid)
          break
        case 'observables':
          assert(this.observables.has(provider.selectable.gid))
          this.observables.delete(provider.selectable.gid)
          break
        case 'checks':
          assert(this.checks.has(provider.selectable.cid))
          this.checks.delete(provider.selectable.cid)
          break
        case 'errors':
          assert(this.errors.has(provider.selectable.eid))
          this.errors.delete(provider.selectable.eid)
          break
      }
    },
    select(element: MultiSelectable) {
      if (this.multiSelecting) {
        const elements = this.selected.elements.slice()
        for (const newElement of element.elements) {
          let found = false
          for (let i = 0; i < elements.length; i++) {
            const oldElement = elements[i]
            if (deepEqual(oldElement, newElement)) {
              // already selected, unselect it
              elements.splice(i, 1)
              found = true
              break
            }
          }
          if (!found) {
            elements.push(newElement)
          }
        }
        this.selected = { elements }
      } else {
        this.selected = element
      }
    },
    hover(element: MultiSelectable, fromRelationLink: boolean = false) {
      this.hovered = element
      if (fromRelationLink && element.elements.length > 0) {
        const firstElement = element.elements[0]!
        
        // Create wireframe directly and get bounding box
        this.createWireframe(firstElement)
        
        // Try to get position from existing mesh first, then from wireframe
        const screenPos = this.getScreenPositionOf(firstElement)
        if (screenPos) {
          this.tooltipX = screenPos.x
          this.tooltipY = screenPos.y
        } else if (this.wireframeBoundingBox) {
          // Use wireframe bounding box for tooltip position
          this.updateTooltipFromWireframe()
        } else {
          this.tooltipX = undefined
          this.tooltipY = undefined
        }
      } else {
        // Clear wireframe and use mouse position (default behavior)
        this.clearWireframe()
        this.tooltipX = undefined
        this.tooltipY = undefined
      }
    },
    clearWireframe() {
      for (const obj of this.wireframeObjects) {
        this.renderer.scene?.remove(obj)
        if (obj instanceof THREE.LineSegments) {
          obj.geometry.dispose()
        }
      }
      this.wireframeObjects = []
      this.wireframeBoundingBox = null
    },
    getMeshesForSelectable(selectable: Selectable): { meshes: Mesh[], basePosition: Position } {
      const e = selectable.e
      switch (e.oneofKind) {
        case 'gadget': {
          const gadget = this.data.gadget(e.gadget.gid)
          return { meshes: this.data.gadgetMeshes(e.gadget.gid), basePosition: gadget.position ?? Position.create() }
        }
        case 'check': {
          const checkModel = this.data.checkModel(e.check.cid)
          const gadget = this.data.gadget(checkModel.gid)
          return { meshes: [this.data.checkMesh(e.check.cid, e.check.checkIndex)], basePosition: gadget.position ?? Position.create() }
        }
        case 'error': {
          const errorModel = this.data.errorModel(e.error.eid)
          const checkModel = this.data.checkModel(errorModel.cid)
          const gadget = this.data.gadget(checkModel.gid)
          return { meshes: [this.data.errorMesh(e.error.eid, e.error.errorIndex)], basePosition: gadget.position ?? Position.create() }
        }
        case 'location': {
          const gadget = this.data.gadget(e.location.gid)
          const mesh = this.data.locationMesh(e.location.gid, e.location.locationIndex)
          if (!mesh) return { meshes: [], basePosition: Position.create() }
          return { meshes: [mesh], basePosition: gadget.position ?? Position.create() }
        }
        default:
          return { meshes: [], basePosition: Position.create() }
      }
    },
    createWireframe(selectable: Selectable) {
      this.clearWireframe()
      
      const { meshes, basePosition } = this.getMeshesForSelectable(selectable)
      if (meshes.length === 0) return
      
      const bbox = new THREE.Box3()
      const objects: THREE.Object3D[] = []
      
      for (const mesh of meshes) {
        const geometry = geometryOf(mesh)
        const wireframeGeometry = new THREE.WireframeGeometry(geometry)
        const wireframe = new THREE.LineSegments(wireframeGeometry, this.wireframeMaterial)
        
        const basePos = pos(basePosition)
        const relativePos = rpos(mesh.relative)
        wireframe.position.copy(basePos).add(relativePos)
        
        if (mesh.rotation) {
          wireframe.rotation.copy(rotationOf(mesh.rotation))
        }
        
        this.renderer.scene?.add(wireframe)
        objects.push(wireframe)
        
        wireframe.updateMatrixWorld(true)
        const meshBbox = new THREE.Box3().setFromObject(wireframe)
        bbox.union(meshBbox)
        
        geometry.dispose()
      }
      
      this.wireframeObjects = objects
      this.wireframeBoundingBox = bbox.isEmpty() ? null : bbox
    },
    updateTooltipFromWireframe() {
      if (this.wireframeBoundingBox) {
        const center = new THREE.Vector3()
        this.wireframeBoundingBox.getCenter(center)
        
        const camera = this.renderer.camera
        if (camera) {
          center.project(camera)
          const canvas = this.renderer.canvas
          const rect = canvas.getBoundingClientRect()
          this.tooltipX = ((center.x + 1) / 2) * rect.width
          this.tooltipY = ((-center.y + 1) / 2) * rect.height
        }
      }
    },
    getScreenPositionOf(selectable: { e: any }): { x: number; y: number } | null {
      // Use objectsOf to get the actual mesh provider and key
      const objects = this.objectsOf({ elements: [selectable as any] })
      if (objects.length === 0) return null
      
      const [provider, key] = objects[0]!
      if (!provider) return null
      
      // Calculate bounding box of the mesh
      const bbox = new THREE.Box3()
      provider.extendBoundingBox(key, bbox)
      
      if (bbox.isEmpty()) return null
      
      // Get the center of the bounding box
      const center = new THREE.Vector3()
      bbox.getCenter(center)
      
      // Project to screen coordinates
      const camera = this.renderer.camera
      if (!camera) return null
      
      center.project(camera)
      
      // Convert from normalized device coordinates to canvas coordinates
      const canvas = this.renderer.canvas
      const rect = canvas.getBoundingClientRect()
      const x = ((center.x + 1) / 2) * rect.width
      const y = ((-center.y + 1) / 2) * rect.height
      
      return { x, y }
    },
    onMouseChange(event: MouseEvent, is_click: boolean = true) {
      const rect = this.renderer.canvas.getBoundingClientRect()
      const position = new THREE.Vector2(event.clientX - rect.left, event.clientY - rect.top)
      const positionN = new THREE.Vector2((position.x / rect.width) * 2 - 1, -(position.y / rect.height) * 2 + 1)
      this.raycaster.setFromCamera(positionN, this.renderer.camera!)
      const intersects = this.raycaster.intersectObjects(this.renderer.scene!.children, false)
      for (const intersect of intersects) {
        if (!intersect.object.visible) continue // don't select invisible object
        if (intersect.object.userData instanceof MyInstancedMesh2) {
          const mesh2 = intersect.object.userData
          const selection = mesh2.selectionOf(intersect.instanceId!)
          if (is_click && event.shiftKey && selection.elements.length > 0) {
            // Shift+click: spawn info pane at cursor position
            this.data.spawnInfoPane(event.clientX, event.clientY, selection.elements[0]!)
          } else {
            ;(is_click ? this.select : this.hover)(selection)
          }
          return
        } else if (intersect.object instanceof THREE.Mesh) {
          if (intersect.object.userData instanceof MeshGroup) {
            const group = intersect.object.userData as MeshGroup
            const selection = group.selectionOf ? group.selectionOf(group.key!) : { elements: [] }
            if (is_click && event.shiftKey && selection.elements.length > 0) {
              // Shift+click: spawn info pane at cursor position
              this.data.spawnInfoPane(event.clientX, event.clientY, selection.elements[0]!)
            } else {
              ;(is_click ? this.select : this.hover)(selection)
            }
            return
          }
        }
      }
      if (is_click) {
        this.select({ elements: [] })
      } else {
        this.hover({ elements: [] })
      }
    },
    objectsOf(selectable: MultiSelectable): [MeshVecProviderPublicInterface, number | bigint | string][] {
      if (!selectable || selectable.elements === undefined) {
        return []
      }
      const objects: [MeshVecProviderPublicInterface, number | bigint | string][] = []
      for (const element of selectable.elements) {
        switch (element.e.oneofKind) {
          case 'gadget': {
            objects.push([this.gadgets!, element.e.gadget.gid])
            break
          }
          case 'location': {
            objects.push([this.locations.get(element.e.location.gid)!, element.e.location.locationIndex])
            break
          }
          case 'port': {
            let portIndex = 0
            if (element.e.port.io.oneofKind === 'input') {
              portIndex = element.e.port.io.input
            } else if (element.e.port.io.oneofKind === 'output') {
              const gadget = this.data.gadget(element.e.port.gid)
              const gadgetType = this.data.gadgetType(gadget.gtype)
              portIndex = element.e.port.io.output + gadgetType.inputs.length
            }
            objects.push([this.ports.get(element.e.port.gid)!, portIndex])
            break
          }
          case 'observable': {
            let portIndex = 0
            if (element.e.observable.io.oneofKind === 'input') {
              portIndex = element.e.observable.io.input
            } else if (element.e.observable.io.oneofKind === 'output') {
              const gadget = this.data.gadget(element.e.observable.gid)
              const gadgetType = this.data.gadgetType(gadget.gtype)
              portIndex = element.e.observable.io.output + gadgetType.inputs.length
            }
            const observableKey = `${portIndex},${element.e.observable.observableIndex}`
            objects.push([this.observables.get(element.e.observable.gid)!, observableKey])
            break
          }
          case 'check': {
            objects.push([this.checks.get(element.e.check.cid)!, element.e.check.checkIndex])
            break
          }
          case 'error': {
            objects.push([this.errors.get(element.e.error.eid)!, element.e.error.errorIndex])
            break
          }
        }
      }
      return objects.filter(([provider, _index]) => provider !== undefined)
    },
    gidsOf(selectable: MultiSelectable): Set<bigint> {
      const gids: Set<bigint> = new Set()
      for (const element of selectable.elements) {
        switch (element.e.oneofKind) {
          case 'gadget':
            gids.add(element.e.gadget.gid)
            break
          case 'location':
            gids.add(element.e.location.gid)
            break
          case 'port':
            gids.add(element.e.port.gid)
            break
          case 'observable':
            gids.add(element.e.observable.gid)
            break
          case 'check':
            gids.add(this.data.checkModel(element.e.check.cid).gid)
            break
          case 'error': {
            const errorModel = this.data.errorModel(element.e.error.eid)
            gids.add(this.data.checkModel(errorModel.cid).gid)
            break
          }
        }
      }
      return gids
    },
    getSelected(): MultiSelectable {
      return this.selected
    },
    getHovered(): MultiSelectable {
      return this.hovered
    },
    updateColor(selectable: MultiSelectable, color: THREE.Color | undefined = undefined) {
      // if color is undefined, recover the color of the elements
      for (const [provider, index] of this.objectsOf(selectable)) {
        provider.setColor(index, color)
      }
    },
    registerLoading(uid: number): void {
      this.loading.add(uid)
    },
    finishLoading(uid: number): void {
      this.loading.delete(uid)
      if (this.loading.size === 0) {
        this.applyHover(this.hovered, this.hovered)
        this.applySelect(this.selected, this.selected)
      }
    },
    applySelect(newValue: MultiSelectable, oldValue: MultiSelectable) {
      this.updateColor(oldValue)
      this.updateColor(this.hovered, this.hoverColor)
      this.updateColor(newValue, this.selectColor)
    },
    applyHover(newValue: MultiSelectable, oldValue: MultiSelectable) {
      this.updateColor(oldValue)
      this.updateColor(this.selected, this.selectColor)
      this.updateColor(newValue, this.hoverColor)
    },
  },
  watch: {
    selected(newValue: MultiSelectable, oldValue: MultiSelectable) {
      this.applySelect(newValue, oldValue)
    },
    hovered(newValue: MultiSelectable, oldValue: MultiSelectable) {
      if (newValue?.elements?.length > 0) {
        const element = newValue.elements[0]
        const oneofKind = element?.e.oneofKind
        if (oneofKind && (element.e as any)[oneofKind] !== undefined) {
          const data = (element.e as any)[oneofKind]
          this.text =
            oneofKind +
            ': ' +
            JSON.stringify(data, (key, value) => {
              // skip oneofKind in nested objects because it's redundant
              if (key == 'oneofKind') {
                return undefined
              }
              let result
              if (typeof value === 'bigint' && value !== null) {
                result = JSON.rawJSON(value.toString())
              } else {
                result = value
              }
              return result
            })
        }
      } else {
        this.text = ''
      }
      this.applyHover(newValue, oldValue)
    },
  },
  unmounted() {
    // Clean up wireframe resources
    this.clearWireframe()
    this.wireframeMaterial.dispose()
  },
  render() {
    return this.$slots.default
      ? [
          this.$slots.default(),
          h(Tooltip, {
            element: this.renderer.canvas,
            text: this.text,
            overrideX: this.tooltipX,
            overrideY: this.tooltipY,
          }),
        ]
      : []
  },
  __hmrId: 'SelectionManager',
})
