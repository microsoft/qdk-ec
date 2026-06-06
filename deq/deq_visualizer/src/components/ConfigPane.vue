<script lang="ts" setup>
import { inject, onMounted, ref, onBeforeUnmount, watch } from 'vue'
import * as EssentialsPlugin from '@tweakpane/plugin-essentials'
import { VisualizerDataInjectionKey } from '@/misc/VisualizerData'
import { assert } from '@/util'
import * as THREE from 'three'
import { RendererInjectionKey } from 'troisjs'
import { SelectionManagerInjectionKey } from '@/misc/SelectionManager'
import { type ButtonGridApi } from '@tweakpane/plugin-essentials'
import { Pane, type FolderApi } from 'tweakpane'
import { VisualizerInjectionKey, type PartialDisplayMode } from '@/util'
import * as vis_pb2 from '@/proto/visualizer'

const data = inject(VisualizerDataInjectionKey)!
assert(data != undefined, 'missing visualizerData injection')
const renderer = inject(RendererInjectionKey)!
assert(renderer != undefined, 'missing renderer injection')
const selectionManager = inject(SelectionManagerInjectionKey)!
assert(selectionManager != undefined, 'missing selectionManager injection')
const visualizer = inject(VisualizerInjectionKey)!
assert(visualizer != undefined, 'missing visualizer injection')

// interface Props {}
// const props = withDefaults(defineProps<Props>(), {})

const el = ref<HTMLElement>(null!)

let pane: FolderApi = null!

class CameraConfig {
  folder: FolderApi

  constructor(public pane: FolderApi) {
    this.folder = pane.addFolder({ title: 'Camera', expanded: true })
    const camera_position_buttons: ButtonGridApi = this.folder.addBlade({
      view: 'buttongrid',
      size: [3, 1],
      cells: (x: number) => ({
        title: ['Top', 'Left', 'Front'][x],
      }),
      label: 'reset view',
    }) as any
    camera_position_buttons.on('click', (event: any) => {
      const i: number = event.index[0]
      ;[this.setTopView, this.setLeftView, this.setFrontView][i]!.bind(this)()
    })

    this.folder.addButton({ title: 'focus to all' }).on('click', () => {
      this.focusToAll()
    })
    const focusSelectedButton = this.folder.addButton({ title: 'focus to selected' }).on('click', () => {
      this.focusToSelected()
    })
    watch(
      [selectionManager.state.selected],
      () => {
        const canFocusSelected = selectionManager.state.selected.value.elements?.length > 0
        focusSelectedButton.disabled = !canFocusSelected
      },
      { immediate: true },
    )
  }

  getBoundingSphere(objects: THREE.Object3D[]): THREE.Sphere {
    const bbox = new THREE.Box3()
    for (const obj of objects) {
      bbox.expandByObject(obj)
    }
    const center = new THREE.Vector3()
    bbox.getCenter(center)
    const bsphere = bbox.getBoundingSphere(new THREE.Sphere(center))
    return bsphere
  }

  globalBoundingSphere(): THREE.Sphere {
    const scene = renderer.scene!
    return this.getBoundingSphere([scene])
  }

  setTopView() {
    const bsphere = this.globalBoundingSphere()
    const position = new THREE.Vector3(bsphere.center.x, bsphere.center.y + bsphere.radius * 2, bsphere.center.z)
    visualizer.cameraPosition.value.copy(position)
    visualizer.orbitTarget.value.copy(bsphere.center)
    renderer.canvas!.focus()
  }

  setLeftView() {
    const bsphere = this.globalBoundingSphere()
    const position = new THREE.Vector3(bsphere.center.x - bsphere.radius * 2, bsphere.center.y, bsphere.center.z)
    visualizer.cameraPosition.value.copy(position)
    visualizer.orbitTarget.value.copy(bsphere.center)
    renderer.canvas!.focus()
  }

  setFrontView() {
    const bsphere = this.globalBoundingSphere()
    const position = new THREE.Vector3(bsphere.center.x, bsphere.center.y, bsphere.center.z + bsphere.radius * 2)
    visualizer.cameraPosition.value.copy(position)
    visualizer.orbitTarget.value.copy(bsphere.center)
    renderer.canvas!.focus()
  }

  focusToBoundingSphereWithSameAngle(bsphere: THREE.Sphere) {
    visualizer.orbitTarget.value.copy(bsphere.center)
    if (!visualizer.cameraPosition.value.equals(bsphere.center)) {
      const direction = new THREE.Vector3()
      direction.subVectors(visualizer.cameraPosition.value, bsphere.center).normalize()
      const position = new THREE.Vector3()
      position
        .copy(direction)
        .multiplyScalar(bsphere.radius * 2)
        .add(bsphere.center)
      visualizer.cameraPosition.value.copy(position)
    }
    renderer.canvas!.focus()
  }

  focusToAll() {
    const bsphere = this.globalBoundingSphere()
    this.focusToBoundingSphereWithSameAngle(bsphere)
  }

  focusToSelected() {
    const selects = selectionManager.objectsOf(selectionManager.getSelected())
    if (selects.length == 0) {
      return
    }
    const bbox = new THREE.Box3()
    for (const [provider, key] of selects) {
      provider.extendBoundingBox(key, bbox)
    }
    const center = new THREE.Vector3()
    bbox.getCenter(center)
    const bsphere = bbox.getBoundingSphere(new THREE.Sphere(center))
    this.focusToBoundingSphereWithSameAngle(bsphere)
  }
}

class DisplayModeConfig {
  folder: FolderApi

  constructor(pane: FolderApi) {
    this.folder = pane.addFolder({ title: 'Display Mode', expanded: true })
    this.folder
      .addBlade({
        view: 'buttongrid',
        size: [3, 1],
        cells: (x: number) => ({
          title: [`Block`, `Realization`, `Hypergraph`][x],
        }),
      })
      .on('click', (event: any) => {
        if (event.index[0] == 0) {
          this.setDisplayMode(vis_pb2.DisplayMode.create({ showBlock: true }))
        } else if (event.index[0] == 1) {
          this.setDisplayMode(vis_pb2.DisplayMode.create({ showRealization: true, showPorts: true }))
        } else if (event.index[0] == 2) {
          this.setDisplayMode(vis_pb2.DisplayMode.create({ showCheckModel: true, showErrorModel: true }))
        }
      })
    this.folder.addBlade({ view: 'separator' })
    for (const name of ['Block', 'Realization', 'Check Model', 'Error Model', 'Ports']) {
      const key = ('show' + name.replace(' ', '')) as 'showBlock' | 'showRealization' | 'showCheckModel' | 'showErrorModel' | 'showPorts'
      this.folder
        .addBlade({
          view: 'buttongrid',
          size: [2, 1],
          cells: (x: number) => ({
            title: [`show ${name.toLowerCase()}`, `hide ${name.toLowerCase()}`][x],
          }),
        })
        .on('click', (event: any) => {
          const partialDisplayMode: PartialDisplayMode = {}
          partialDisplayMode[key] = event.index[0] == 0
          this.setDisplayMode(partialDisplayMode)
        })
    }
    watch(
      [selectionManager.state.selected],
      () => {
        const selected = selectionManager.getSelected()
        const hasSelected = selectionManager.gidsOf(selected).size > 0
        this.folder.title = hasSelected ? 'Display Mode (selected)' : 'Display Mode (global)'
      },
      { immediate: true },
    )
  }

  setDisplayMode(displayMode: PartialDisplayMode) {
    const selected = selectionManager.getSelected()
    const gids = selectionManager.gidsOf(selected)
    if (gids.size > 0) {
      for (const gid of gids) {
        visualizer.setDisplayMode(gid, displayMode)
      }
    } else {
      visualizer.setGlobalDisplayMode(displayMode)
    }
    renderer.canvas!.focus()
  }
}

onMounted(() => {
  const container = el.value.parentNode
  pane = new Pane({
    title: 'VisualQEC Control',
    container: container as HTMLElement,
    expanded: true,
  })
  onBeforeUnmount(() => {
    pane.dispose()
  })
  pane.registerPlugin(EssentialsPlugin)

  new CameraConfig(pane)
  new DisplayModeConfig(pane)
})
</script>

<template><div ref="el"></div></template>
