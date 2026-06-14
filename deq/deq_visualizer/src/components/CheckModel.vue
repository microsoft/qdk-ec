<script setup lang="ts">
import { inject, ref, watch } from 'vue'
import { VisualizerDataInjectionKey } from '@/misc/VisualizerData'
import MeshVecProvider, { type MeshVecProviderPublicInterface, MeshGroup } from '@/misc/MeshVecProvider'
import { MultiSelectable } from '@/proto/visualizer'
import { assert } from '@/util'

const data = inject(VisualizerDataInjectionKey)!
assert(data != undefined, 'missing visualizerData injection')

interface Props {
  cid: bigint
}
const props = withDefaults(defineProps<Props>(), {})

const checkModel = data.checkModel(props.cid)
const gid = checkModel.gid
const gadget = data.gadget(gid)
const checkModelType = data.checkModelType(checkModel.ctype)

const displayMode = data.displayMode.get(gid)!

function checkSelectionOf(checkIndex: number): MultiSelectable {
  return {
    elements: [{ e: { oneofKind: 'check', check: { cid: props.cid, checkIndex } } }],
  }
}
const checkMeshVecProvider = ref<MeshVecProviderPublicInterface>(null!)
function loadCheckGroups(meshVecProvider: MeshVecProviderPublicInterface) {
  for (const [checkIndex, _check] of checkModelType.checks.entries()) {
    const group = new MeshGroup<number>([], checkIndex, checkSelectionOf)
    const mesh = data.checkMesh(checkModel.cid, checkIndex)
    group.meshes.push(mesh)
    meshVecProvider.addGroup(group)
  }
}
watch([checkMeshVecProvider], ([meshVecProvider]) => {
  if (meshVecProvider) {
    loadCheckGroups(meshVecProvider)
    meshVecProvider.finishLoading()
  }
})
</script>

<template>
  <MeshVecProvider v-if="displayMode.showCheckModel" ref="checkMeshVecProvider" :relative="gadget.position" :selectable="{ oneofKind: 'checks', cid: props.cid }" />
</template>
