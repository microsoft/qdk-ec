<script setup lang="ts">
import { inject, onMounted, watch, ref } from 'vue'
import { VisualizerDataInjectionKey } from '@/misc/VisualizerData'
import { MultiSelectable } from '@/proto/visualizer'
import { MeshGroup, type MeshVecProviderPublicInterface } from '@/misc/MeshVecProvider'
import { assert } from '@/util'
import MeshVecProvider from '@/misc/MeshVecProvider'

const data = inject(VisualizerDataInjectionKey)!
assert(data != undefined, 'missing visualizerData injection')

const blocksMeshVecProvider = ref<MeshVecProviderPublicInterface>(null!)

function selectionOf(gid: bigint): MultiSelectable {
  return {
    elements: [{ e: { oneofKind: 'gadget', gadget: { gid } } }],
  }
}

const blockGroups: Map<bigint, MeshGroup> = new Map()
function loadBlockGroups(meshVecProvider: MeshVecProviderPublicInterface) {
  for (const [gid, gadget] of data.gadgets) {
    const displayMode = data.displayMode.get(gid)!
    watch(
      displayMode,
      (displayMode) => {
        // always refresh block visualization for consistency
        // we will include the ports as part of the block mesh when the realization
        // is not shown; otherwise the realization should display the ports and let
        // them be selectable independently
        const oldShow = blockGroups.has(gid)
        const newShow = displayMode.showBlock
        if (oldShow !== newShow) {
          meshVecProvider.removeGroup(blockGroups.get(gid))
          blockGroups.delete(gid)
          if (displayMode.showBlock) {
            const group = new MeshGroup<bigint>([], gid, selectionOf)
            group.relative = gadget.position
            group.meshes.push(...data.gadgetMeshes(gid))
            meshVecProvider.addGroup(group)
            blockGroups.set(gid, group)
          }
        }
      },
      { flush: 'sync', immediate: true },
    )
  }
}

onMounted(() => {
  const meshVecProvider = blocksMeshVecProvider.value
  assert(meshVecProvider != undefined, 'unreachable')
  loadBlockGroups(meshVecProvider)
  meshVecProvider.finishLoading()
})
</script>

<template><MeshVecProvider ref="blocksMeshVecProvider" :selectable="{ oneofKind: 'gadgets' }" /></template>
