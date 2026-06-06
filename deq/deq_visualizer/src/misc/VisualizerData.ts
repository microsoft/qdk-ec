import * as pb2 from '@/proto/deq_bin'
import * as vis_pb from '@/proto/visualizer'
import { assert, addTo, divideBy, cached_dict, meshVecBiasedBy } from '@/util'
import { cached_property } from 'cached_property'
import { type InjectionKey, type Reactive, reactive, type Ref, ref, type ShallowRef, shallowRef } from 'vue'
import { supportRelativeOf } from '@/misc/meshes'

export const VisualizerDataInjectionKey: InjectionKey<VisualizerData> = Symbol('visualizerData')

// Info pane state for floating, draggable info panels
export interface InfoPaneState {
  id: string
  x: number
  y: number
  selection: vis_pb.Selectable
  zIndex: number
}

let nextPaneId = 0

export class VisualizerData {
  readonly library: pb2.Library
  displayMode: Map<bigint, Reactive<vis_pb.DisplayMode>>
  showStates: Ref<boolean>
  showConfig: Ref<boolean>
  showInfo: Ref<boolean>
  
  // Info pane management
  infoPanes: ShallowRef<Map<string, InfoPaneState>>
  nextZIndex: Ref<number>
  focusedPaneId: Ref<string | null>
  isDraggingPane: Ref<boolean>

  constructor(library: pb2.Library) {
    assign_instance_ids(library)
    this.library = library
    this.displayMode = new Map()
    for (const gid of this.gadgets.keys()) {
      this.displayMode.set(gid, reactive(vis_pb.DisplayMode.create({ showBlock: true })))
    }
    this.showStates = ref(false)
    this.showConfig = ref(true)
    this.showInfo = ref(true)
    
    // Initialize info pane state
    this.infoPanes = shallowRef(new Map())
    this.nextZIndex = ref(1000)
    this.focusedPaneId = ref(null)
    this.isDraggingPane = ref(false)
  }

  // Spawn a new info pane at the specified position
  spawnInfoPane(x: number, y: number, selection: vis_pb.Selectable): string {
    const id = `pane-${nextPaneId++}`
    const zIndex = this.nextZIndex.value++
    const newMap = new Map(this.infoPanes.value)
    newMap.set(id, { id, x, y, selection, zIndex })
    this.infoPanes.value = newMap
    this.focusedPaneId.value = id
    return id
  }

  // Close an info pane by id
  closeInfoPane(id: string): void {
    const newMap = new Map(this.infoPanes.value)
    newMap.delete(id)
    this.infoPanes.value = newMap
    if (this.focusedPaneId.value === id) {
      // Find the pane with highest zIndex to become focused
      let maxZIndex = -1
      let newFocusedId: string | null = null
      for (const [paneId, pane] of this.infoPanes.value) {
        if (pane.zIndex > maxZIndex) {
          maxZIndex = pane.zIndex
          newFocusedId = paneId
        }
      }
      this.focusedPaneId.value = newFocusedId
    }
  }

  // Bring a pane to front and set it as focused
  focusInfoPane(id: string): void {
    const pane = this.infoPanes.value.get(id)
    if (pane) {
      const newMap = new Map(this.infoPanes.value)
      newMap.set(id, { ...pane, zIndex: this.nextZIndex.value++ })
      this.infoPanes.value = newMap
      this.focusedPaneId.value = id
    }
  }

  // Close the currently focused pane
  closeFocusedPane(): void {
    if (this.focusedPaneId.value) {
      this.closeInfoPane(this.focusedPaneId.value)
    }
  }

  @cached_property
  get gadgets(): Map<bigint, pb2.Gadget> {
    const gadgets: Map<bigint, pb2.Gadget> = new Map()
    for (const instruction of this.library.program) {
      if (instruction.create.oneofKind === 'gadget') {
        const gadget = instruction.create.gadget
        gadgets.set(gadget.gid, gadget)
      }
    }
    return gadgets
  }

  gadget(gid: bigint): pb2.Gadget {
    return this.gadgets.get(gid)!
  }

  // Get all meshes for a gadget including its ports
  gadgetMeshes(gid: bigint): vis_pb.Mesh[] {
    const gadget = this.gadget(gid)
    const gadgetType = this.gadgetType(gadget.gtype)
    const meshes: vis_pb.Mesh[] = [...gadgetType.mesh]
    for (const portGroup of [gadgetType.inputs, gadgetType.outputs]) {
      for (const port of portGroup) {
        const portType = this.portType(port.ptype)
        meshes.push(...meshVecBiasedBy(portType.mesh, vis_pb.Position.create(port.relative)))
      }
    }
    // Default box representation if no meshes defined
    if (meshes.length === 0) {
      meshes.push(vis_pb.Mesh.create({
        geometry: { type: 'box', size: [1, 1, 1] },
        material: { type: 'standard', color: 'grey' },
        relative: vis_pb.Position.create(),
      }))
    }
    return meshes
  }

  // Get mesh for a single check (sphere at check position)
  checkMesh(cid: bigint, checkIndex: number): vis_pb.Mesh {
    const checkPositions = this.checkPositions(cid)
    const checkModel = this.checkModel(cid)
    const checkModelType = this.checkModelType(checkModel.ctype)
    return vis_pb.Mesh.create({
      geometry: { type: 'sphere', size: [this.qubitRadius] },
      material: { type: 'standard', color: checkModelType.checks[checkIndex]!.color || 'black' },
      relative: checkPositions[checkIndex],
    })
  }

  // Get mesh for a single error (box at error position)
  errorMesh(eid: bigint, errorIndex: number): vis_pb.Mesh {
    const errorPositions = this.errorPositions(eid)
    const errorModel = this.errorModel(eid)
    const errorModelType = this.errorModelType(errorModel.etype)
    return vis_pb.Mesh.create({
      geometry: { type: 'box', size: [this.qubitRadius * 0.5] },
      material: { type: 'standard', color: errorModelType.errors[errorIndex]!.color || 'red' },
      relative: errorPositions[errorIndex],
    })
  }

  // Get mesh for a location (sphere at location time position)
  locationMesh(gid: bigint, locationIndex: number): vis_pb.Mesh | null {
    const gadget = this.gadget(gid)
    const gadgetType = this.gadgetType(gadget.gtype)
    const location = gadgetType.realization?.locations[locationIndex]
    if (!location) return null
    return vis_pb.Mesh.create({
      geometry: { type: 'sphere', size: [this.qubitRadius * 0.5] },
      relative: vis_pb.Position.create({ t: location.t, i: 0, j: 0 }),
    })
  }

  @cached_property
  get gadgetOutputPeers(): Map<bigint, pb2.Gadget_Connector[]> {
    const gadgetOutputPeers: Map<bigint, pb2.Gadget_Connector[]> = new Map()
    // create the lists
    for (const gadget of this.gadgets.values()) {
      const gadgetType = this.gadgetType(gadget.gtype)
      const outputPeers: pb2.Gadget_Connector[] = []
      outputPeers.length = gadgetType.outputs.length
      gadgetOutputPeers.set(gadget.gid, outputPeers)
    }
    // assign the lists
    for (const gadget of this.gadgets.values()) {
      gadget.connectors.forEach((connector, index) => {
        const outputPeers = gadgetOutputPeers.get(connector.gid)!
        outputPeers[Number(connector.port)] = pb2.Gadget_Connector.create({
          gid: gadget.gid,
          port: BigInt(index),
        })
      })
    }
    return gadgetOutputPeers
  }

  gadgetOutputPeer(gid: bigint, port: bigint | number): pb2.Gadget_Connector {
    return this.gadgetOutputPeers.get(gid)![Number(port)]!
  }

  @cached_dict
  measurementPositions(gid: bigint): vis_pb.Position[] {
    const gadget = this.gadget(gid)
    const gadgetType = this.gadgetType(gadget.gtype)
    const measurementPositions: vis_pb.Position[] = []
    const gadgetPos = gadget.position ?? vis_pb.Position.create()
    for (const location of gadgetType.realization?.locations || []) {
      if (location.operation?.type === vis_pb.OperationType.MEASURE) {
        // the relative position of the measurement is the center of all its supports
        const center = vis_pb.Position.create()
        for (const qubit of location.operation.support) {
          const relative = supportRelativeOf(qubit, gadgetType.realization!)
          addTo(center, relative)
        }
        divideBy(center, location.operation.support.length)
        center.t += location.t
        addTo(center, gadgetPos)
        measurementPositions.push(center)
      }
    }
    if (measurementPositions.length < gadgetType.measurements.length) {
      // there are unspecified measurement positions; fill them with the gadget position plus its index
      for (let mi = measurementPositions.length; mi < gadgetType.measurements.length; mi++) {
        measurementPositions.push(
          vis_pb.Position.create({
            t: gadgetPos.t,
            i: gadgetPos.i + mi,
            j: gadgetPos.j,
          }),
        )
      }
    }
    return measurementPositions
  }

  @cached_property
  get checkModels(): Map<bigint, pb2.CheckModel> {
    const checkModels: Map<bigint, pb2.CheckModel> = new Map()
    for (const instruction of this.library.program) {
      if (instruction.create.oneofKind === 'checkModel') {
        const checkModel = instruction.create.checkModel
        checkModels.set(checkModel.cid, checkModel)
      }
    }
    return checkModels
  }

  checkModel(cid: bigint): pb2.CheckModel {
    return this.checkModels.get(cid)!
  }

  @cached_property
  get gadgetBindings(): Map<bigint, bigint> {
    const gadgetBindings: Map<bigint, bigint> = new Map()
    for (const checkModel of this.checkModels.values()) {
      assert(!gadgetBindings.has(checkModel.gid), `duplicate gadget binding`)
      gadgetBindings.set(checkModel.gid, checkModel.cid)
    }
    return gadgetBindings
  }

  gadgetBinding(gid: bigint): bigint | undefined {
    return this.gadgetBindings.get(gid)
  }

  @cached_dict
  remoteGadgetVec(cid: bigint): { gid: bigint; bias: bigint }[] {
    const checkModel = this.checkModel(cid)
    const checkModelType = this.checkModelType(checkModel.ctype)
    // apply the modifier
    const remoteGadgets = [...checkModelType.remoteGadgets]
    checkModel.modifier?.rerouteRemoteGadgets.forEach((reroute) => {
      while (reroute.remoteGadgetIndex >= remoteGadgets.length) {
        remoteGadgets.push(pb2.CheckModelType_RemoteGadget.create({ tag: 'placeholder' }))
      }
      remoteGadgets[Number(reroute.remoteGadgetIndex)] = reroute.value!
    })
    const remoteGadgetGidVec: bigint[] = []
    remoteGadgetGidVec.length = remoteGadgets.length
    for (let ri = 0; ri < remoteGadgets.length; ri++) {
      this.expandGid(checkModel.cid, checkModel.gid, remoteGadgets, ri, remoteGadgetGidVec)
    }
    return remoteGadgetGidVec.map((gid, index) => ({ gid, bias: remoteGadgets[index]!.measurementBias ?? 0n }))
  }

  private expandGid(cid: bigint, gid: bigint, remoteGadgets: pb2.CheckModelType_RemoteGadget[], ri: number, remoteGadgetGidVec: bigint[]) {
    if (remoteGadgetGidVec[ri] !== undefined) {
      return // already expanded
    }
    const remoteGadget = remoteGadgets[ri]!
    if (this.isPlaceholderRemoteGadget(remoteGadget)) {
      return // no need to expand
    }

    // if absolute_gid is provided, use it directly
    if (remoteGadget.absoluteGid !== undefined) {
      remoteGadgetGidVec[ri] = remoteGadget.absoluteGid
      return
    }

    assert(remoteGadget.port.oneofKind !== undefined)

    // find the previous gadget instance
    let previousGid = gid
    if (remoteGadget.previousRemoteGadget !== undefined) {
      const previous = Number(remoteGadget.previousRemoteGadget)
      if (remoteGadgetGidVec[previous] === undefined) {
        this.expandGid(cid, gid, remoteGadgets, previous, remoteGadgetGidVec)
      }
      previousGid = remoteGadgetGidVec[previous]!
      assert(previousGid !== undefined)
    }

    const previousGadget = this.gadget(previousGid)

    // write the current gadget
    if (remoteGadget.port.oneofKind === 'input') {
      remoteGadgetGidVec[ri] = previousGadget.connectors[Number(remoteGadget.port.input!)]!.gid
    } else {
      remoteGadgetGidVec[ri] = this.gadgetOutputPeer(previousGadget.gid, remoteGadget.port.output!).gid
    }
  }

  private isPlaceholderRemoteGadget(remoterGadget: pb2.CheckModelType_RemoteGadget): boolean {
    return remoterGadget.tag === 'placeholder' && remoterGadget.port.oneofKind === undefined
  }

  @cached_dict
  checkPositions(cid: bigint): vis_pb.Position[] {
    const checkModel = this.checkModel(cid)
    const checkModelType = this.checkModelType(checkModel.ctype)
    const remoteGadgetVec = this.remoteGadgetVec(checkModel.cid)
    const gadgetPos = this.gadget(checkModel.gid).position ?? vis_pb.Position.create()
    const checkPositions: vis_pb.Position[] = []
    for (const check of checkModelType.checks) {
      let center = vis_pb.Position.create(check.relative)
      if (check.relative === undefined && check.measurements.length > 0) {
        // position check at the measurement with the highest t value
        let bestPos: vis_pb.Position | undefined
        for (const m of check.measurements) {
          const rg = m.remoteGadget === undefined ? { gid: checkModel.gid, bias: 0n } : remoteGadgetVec[Number(m.remoteGadget)]!
          const mPositions = this.measurementPositions(rg.gid)
          const mPos = mPositions[Number(m.measurementIndex + rg.bias)]!
          if (bestPos === undefined || mPos.t > bestPos.t) {
            bestPos = mPos
          }
        }
        // convert absolute position to gadget-relative
        center = vis_pb.Position.create({
          t: bestPos!.t - gadgetPos.t,
          i: bestPos!.i - gadgetPos.i,
          j: bestPos!.j - gadgetPos.j,
        })
      }
      checkPositions.push(center)
    }
    return checkPositions
  }

  @cached_property
  get errorModels(): Map<bigint, pb2.ErrorModel> {
    const errorModels: Map<bigint, pb2.ErrorModel> = new Map()
    for (const instruction of this.library.program) {
      if (instruction.create.oneofKind === 'errorModel') {
        const errorModel = instruction.create.errorModel
        errorModels.set(errorModel.eid, errorModel)
      }
    }
    return errorModels
  }

  errorModel(eid: bigint): pb2.ErrorModel {
    return this.errorModels.get(eid)!
  }

  @cached_dict
  remoteCheckModelVec(eid: bigint): { cid: bigint; bias: bigint }[] {
    const errorModel = this.errorModel(eid)
    const errorModelType = this.errorModelType(errorModel.etype)
    // apply the modifier
    const remoteCheckModels = [...errorModelType.remoteCheckModels]
    errorModel.modifier?.rerouteRemoteCheckModels.forEach((reroute) => {
      while (reroute.remoteCheckModelIndex >= remoteCheckModels.length) {
        remoteCheckModels.push(pb2.ErrorModelType_RemoteCheckModel.create({ tag: 'placeholder' }))
      }
      remoteCheckModels[Number(reroute.remoteCheckModelIndex)] = reroute.value!
    })
    const remoteCheckModelCidVec: bigint[] = []
    remoteCheckModelCidVec.length = remoteCheckModels.length
    for (let ri = 0; ri < remoteCheckModels.length; ri++) {
      this.expandCid(errorModel.eid, errorModel.cid, remoteCheckModels, ri, remoteCheckModelCidVec)
    }
    return remoteCheckModelCidVec.map((cid, index) => ({ cid, bias: remoteCheckModels[index]!.checkBias ?? 0n }))
  }

  private expandCid(eid: bigint, cid: bigint, remoteCheckModels: pb2.ErrorModelType_RemoteCheckModel[], ri: number, remoteCheckModelCidVec: bigint[]) {
    if (remoteCheckModelCidVec[ri] !== undefined) {
      return // already expanded
    }
    const remoteCheckModel = remoteCheckModels[ri]!
    if (this.isPlaceholderRemoteCheckModel(remoteCheckModel)) {
      return // no need to expand
    }

    // if absolute_cid is provided, use it directly
    if (remoteCheckModel.absoluteCid !== undefined) {
      remoteCheckModelCidVec[ri] = remoteCheckModel.absoluteCid
      return
    }

    assert(remoteCheckModel.port.oneofKind !== undefined)

    // find the previous gadget instance
    let previousCid = cid
    if (remoteCheckModel.previousRemoteCheckModel !== undefined) {
      const previous = Number(remoteCheckModel.previousRemoteCheckModel)
      if (remoteCheckModelCidVec[previous] === undefined) {
        this.expandCid(eid, cid, remoteCheckModels, previous, remoteCheckModelCidVec)
      }
      previousCid = remoteCheckModelCidVec[previous]!
      assert(previousCid !== undefined)
    }

    const previousCheckModel = this.checkModel(previousCid)
    const previousGid = previousCheckModel.gid
    const previousGadget = this.gadget(previousGid)

    // write the current check model
    let gid = -1n
    if (remoteCheckModel.port.oneofKind === 'input') {
      gid = previousGadget.connectors[Number(remoteCheckModel.port.input!)]!.gid
    } else {
      gid = this.gadgetOutputPeer(previousGadget.gid, remoteCheckModel.port.output!).gid
    }
    assert(gid !== -1n)

    const remoteCid = this.gadgetBinding(gid)
    assert(remoteCid !== undefined)
    remoteCheckModelCidVec[ri] = remoteCid
  }

  private isPlaceholderRemoteCheckModel(remoterCheckModel: pb2.ErrorModelType_RemoteCheckModel): boolean {
    return remoterCheckModel.tag === 'placeholder' && remoterCheckModel.port.oneofKind === undefined
  }

  @cached_dict
  errorPositions(eid: bigint): vis_pb.Position[] {
    const errorModel = this.errorModel(eid)
    const errorModelType = this.errorModelType(errorModel.etype)
    const gid = this.checkModel(errorModel.cid).gid
    const gadgetPos = this.gadget(gid).position ?? vis_pb.Position.create()
    const remoteCheckModels = this.remoteCheckModelVec(errorModel.cid)
    const errorPositions: vis_pb.Position[] = []
    for (const error of errorModelType.errors) {
      const center = vis_pb.Position.create(error.relative)
      if (error.relative === undefined && error.checks.length > 0) {
        // the center of the error is the average of all its checks
        for (const check of error.checks) {
          const remoteCheckModel = check.remoteCheckModel === undefined ? { cid: errorModel.cid, bias: 0n } : remoteCheckModels[Number(check.remoteCheckModel)]!
          const checkPositions = this.checkPositions(remoteCheckModel.cid)
          const checkPos = checkPositions[Number(check.checkIndex + remoteCheckModel.bias)]!
          // checkPositions are relative to their owning gadget; convert to
          // relative to this error's gadget for consistent averaging
          const remoteCid = remoteCheckModel.cid
          const remoteGid = this.checkModel(remoteCid).gid
          const remoteGadgetPos = this.gadget(remoteGid).position ?? vis_pb.Position.create()
          addTo(center, {
            t: checkPos.t + remoteGadgetPos.t - gadgetPos.t,
            i: checkPos.i + remoteGadgetPos.i - gadgetPos.i,
            j: checkPos.j + remoteGadgetPos.j - gadgetPos.j,
          })
        }
        divideBy(center, error.checks.length)
      }
      errorPositions.push(center)
    }
    return errorPositions
  }

  @cached_property
  get checkModelAttaches(): Map<bigint, bigint[]> {
    const checkModelAttaches: Map<bigint, bigint[]> = new Map()
    for (const errorModel of this.errorModels.values()) {
      if (!checkModelAttaches.has(errorModel.cid)) {
        checkModelAttaches.set(errorModel.cid, [])
      }
      checkModelAttaches.get(errorModel.cid)!.push(errorModel.eid)
    }
    return checkModelAttaches
  }

  checkModelAttach(cid: bigint): bigint[] {
    return this.checkModelAttaches.get(cid) || []
  }

  @cached_property
  get gadgetTypes(): Map<bigint, pb2.GadgetType> {
    const gadgetTypes: Map<bigint, pb2.GadgetType> = new Map()
    for (const gadgetType of this.library.gadgetTypes) {
      gadgetTypes.set(gadgetType.gtype, gadgetType)
    }
    return gadgetTypes
  }

  gadgetType(gtype: bigint): pb2.GadgetType {
    return this.gadgetTypes.get(gtype)!
  }

  @cached_property
  get portTypes(): Map<bigint, pb2.PortType> {
    const portTypes: Map<bigint, pb2.PortType> = new Map()
    for (const portType of this.library.portTypes) {
      portTypes.set(portType.ptype, portType)
    }
    return portTypes
  }

  portType(ptype: bigint): pb2.PortType {
    return this.portTypes.get(ptype)!
  }

  @cached_property
  get checkModelTypes(): Map<bigint, pb2.CheckModelType> {
    const checkModelTypes: Map<bigint, pb2.CheckModelType> = new Map()
    for (const checkModelType of this.library.checkModelTypes) {
      checkModelTypes.set(checkModelType.ctype, checkModelType)
    }
    return checkModelTypes
  }

  checkModelType(ctype: bigint): pb2.CheckModelType {
    return this.checkModelTypes.get(ctype)!
  }

  @cached_property
  get errorModelTypes(): Map<bigint, pb2.ErrorModelType> {
    const errorModelTypes: Map<bigint, pb2.ErrorModelType> = new Map()
    for (const errorModelType of this.library.errorModelTypes) {
      errorModelTypes.set(errorModelType.etype, errorModelType)
    }
    return errorModelTypes
  }

  errorModelType(etype: bigint): pb2.ErrorModelType {
    return this.errorModelTypes.get(etype)!
  }

  @cached_property
  get qubitRadius(): number {
    return this.library.visualConfig?.qubitRadius || 0.1
  }
}

function assign_instance_ids(library: pb2.Library) {
  let gid = 1n
  let cid = 1n
  let eid = 1n
  const gids = new Set<bigint>()
  const cids = new Set<bigint>()
  const eids = new Set<bigint>()
  for (const instruction of library.program) {
    if (instruction.create.oneofKind === 'gadget') {
      const gadget = instruction.create.gadget
      if (gadget.gid === 0n) {
        gadget.gid = gid
        gid += 1n
      }
      assert(!gids.has(gadget.gid), `duplicate gid=${gadget.gid}`)
      gids.add(gadget.gid)
    } else if (instruction.create.oneofKind === 'checkModel') {
      const checkModel = instruction.create.checkModel
      if (checkModel.cid === 0n) {
        checkModel.cid = cid
        cid += 1n
      }
      assert(!cids.has(checkModel.cid), `duplicate cid=${checkModel.cid}`)
      cids.add(checkModel.cid)
    } else if (instruction.create.oneofKind === 'errorModel') {
      const checkModel = instruction.create.errorModel
      if (checkModel.eid === 0n) {
        checkModel.eid = eid
        eid += 1n
      }
      assert(!eids.has(checkModel.eid), `duplicate eid=${checkModel.eid}`)
      eids.add(checkModel.eid)
    }
  }
}
