<script setup lang="ts">
import { inject, ref, watch } from 'vue'
import { VisualizerDataInjectionKey } from '@/misc/VisualizerData'
import MeshVecProvider, { type MeshVecProviderPublicInterface, MeshGroup } from '@/misc/MeshVecProvider'
import { MultiSelectable, Mesh, Position } from '@/proto/visualizer'
import { assert } from '@/util'
import { meshConnecting } from '@/misc/meshes'

const data = inject(VisualizerDataInjectionKey)!
assert(data != undefined, 'missing visualizerData injection')

interface Props {
  eid: bigint
}
const props = withDefaults(defineProps<Props>(), {})

const errorModel = data.errorModel(props.eid)
const errorModelType = data.errorModelType(errorModel.etype)
const cid = errorModel.cid
const checkModel = data.checkModel(cid)
const gid = checkModel.gid
const gadget = data.gadget(gid)

const displayMode = data.displayMode.get(gid)!

function errorSelectionOf(errorIndex: number): MultiSelectable {
  return {
    elements: [{ e: { oneofKind: 'error', error: { eid: props.eid, errorIndex } } }],
  }
}
const errorMeshVecProvider = ref<MeshVecProviderPublicInterface>(null!)
function loadErrorGroups(meshVecProvider: MeshVecProviderPublicInterface) {
  const errorPositions = data.errorPositions(errorModel.eid)
  const remoteCheckModels = data.remoteCheckModelVec(errorModel.cid)
  for (const [errorIndex, error] of errorModelType.errors.entries()) {
    const group = new MeshGroup<number>([], errorIndex, errorSelectionOf)
    const errorCenter = errorPositions[errorIndex]!
    const errorMesh = data.errorMesh(errorModel.eid, errorIndex)
    group.meshes.push(errorMesh)
    // draw the links from the checks to the error center
    for (let check of error.checks) {
      const remoteCheckModel = check.remoteCheckModel === undefined ? { cid: errorModel.cid, bias: 0n } : remoteCheckModels[Number(check.remoteCheckModel)]!
      const checkPositions = data.checkPositions(remoteCheckModel.cid)
      const rawCheckPos = checkPositions[Number(check.checkIndex + remoteCheckModel.bias)]!
      // Convert check position from its owning gadget's coordinate space
      // to this error's gadget coordinate space
      const remoteGid = data.checkModel(remoteCheckModel.cid).gid
      const remoteGadgetPos = data.gadget(remoteGid).position ?? Position.create()
      const thisGadgetPos = gadget.position ?? Position.create()
      const checkPos = Position.create({
        t: (rawCheckPos.t ?? 0) + (remoteGadgetPos.t ?? 0) - (thisGadgetPos.t ?? 0),
        i: (rawCheckPos.i ?? 0) + (remoteGadgetPos.i ?? 0) - (thisGadgetPos.i ?? 0),
        j: (rawCheckPos.j ?? 0) + (remoteGadgetPos.j ?? 0) - (thisGadgetPos.j ?? 0),
      })
      const samePos = errorCenter.t === checkPos.t && errorCenter.i === checkPos.i && errorCenter.j === checkPos.j
      if (samePos) {
        // error and check coincide: draw a circle to make it visible and clickable
        group.meshes.push(
          Mesh.create({
            geometry: { type: 'circle', size: [data.qubitRadius * 1.5] },
            material: { type: 'standard', color: error.color ||  'lightgrey' },
            relative: checkPos,
            rotation: { j: Math.PI / 2 },
          }),
        )
      } else {
        group.meshes.push(meshConnecting(checkPos, errorCenter, 0.1 * data.qubitRadius, error.color || 'grey'))
      }
    }
    meshVecProvider.addGroup(group)
  }
}
watch([errorMeshVecProvider], ([meshVecProvider]) => {
  if (meshVecProvider) {
    loadErrorGroups(meshVecProvider)
    meshVecProvider.finishLoading()
  }
})
</script>

<template>
  <MeshVecProvider v-if="displayMode.showErrorModel" ref="errorMeshVecProvider" :relative="gadget.position" :selectable="{ oneofKind: 'errors', eid: props.eid }" />
</template>
