import type { RenderProps, AnyModel } from '@anywidget/types'
import { createApp, type App, type Reactive, reactive, watch, type ShallowRef, shallowRef } from 'vue'
import Visualizer, { type VisualizerPublicInterface } from '@/Visualizer.vue'
import { assert, type PartialDisplayMode } from '@/util'
import { Vector3 } from 'three'

interface WidgetModel {
  // python -> JS props (readonly means that the frontend may not respect changes from Python)
  readonly _esm: string
  readonly _library: DataView // proto3 type: pb2.Library
  // bidirectional props
  width: number
  height: number
  _selected: DataView // proto3 type: vis_pb2.Selectable
  _hovered: DataView // proto3 type: vis_pb2.Selectable
  cameraPosition: { x: number; y: number; z: number }
  orbitTarget: { x: number; y: number; z: number }
  displayMode: { [gid: string]: PartialDisplayMode }
  cameraType: string // 'perspective' | 'orthographic'
  gateStyle: string // 'top' | 'front'
  background: string // CSS color for scene background
}

function render({ model, el }: RenderProps<WidgetModel>) {
  const div = document.createElement('div')
  el.appendChild(div)
  model.on('msg:custom', onCustomMessage)
  const app = bind(div, {
    // python -> JS props
    esm: bindShallowRef('_esm', model, false),
    library: bindShallowRef('_library', model, false, pyToJsBytes),
    // bidirectional props
    width: bindShallowRef('width', model),
    height: bindShallowRef('height', model),
    selected: bindShallowRef('_selected', model, true, pyToJsBytes, jsToPyBytes),
    hovered: bindShallowRef('_hovered', model, true, pyToJsBytes, jsToPyBytes),
    cameraPosition: bindReactive('cameraPosition', model, pyToJsVector3),
    orbitTarget: bindReactive('orbitTarget', model, pyToJsVector3),
    displayMode: bindReactive('displayMode', model),
    cameraType: model.get('cameraType') ?? 'perspective',
    gateStyle: model.get('gateStyle') ?? 'top',
    background: model.get('background') ?? '#f0f0f0',
  })
  console.log('creating app uid', app._uid)
  Object.assign(model, { app })
}

// only the dev mode calls this function; see deq.visual.widget `ESM` for production function
function initialize({ model }: any) {
  return () => {
    console.log('unmounting app uid', model.app._uid)
    model.app?.unmount()
  }
}

export default { render, initialize }

export function bindStatic(rootContainer: string | Element, props: WidgetModel): VisualizerPublicInterface {
  assert(props != undefined)
  const app = createApp(Visualizer, {
    // python -> JS props
    esm: nullableShallowRef(props._esm),
    library: nullableShallowRef(props._library),
    // bidirectional props
    width: nullableShallowRef(props.width),
    height: nullableShallowRef(props.height),
    selected: nullableShallowRef(props._selected),
    hovered: nullableShallowRef(props._hovered),
    cameraPosition: props.cameraPosition != undefined ? reactive(pyToJsVector3(props.cameraPosition)) : undefined,
    orbitTarget: props.orbitTarget != undefined ? reactive(pyToJsVector3(props.orbitTarget)) : undefined,
    displayMode: nullableReactive(props.displayMode),
    cameraType: props.cameraType ?? 'perspective',
    gateStyle: props.gateStyle ?? 'top',
    background: props.background ?? '#f0f0f0',
  })
  const visualizer: VisualizerPublicInterface = app.mount(rootContainer) as any
  return visualizer
}

function nullableShallowRef(nullable: any): ShallowRef<any> | undefined {
  if (nullable != undefined) return shallowRef(nullable)
}

function nullableReactive(nullable: any): Reactive<any> | undefined {
  if (nullable != undefined) return reactive(nullable)
}

function bind(rootContainer: string | Element, props: any): App<Element> {
  const app = createApp(Visualizer, props)
  app.mount(rootContainer)
  return app
}

function pyToJsVector3(value: { x: number; y: number; z: number }): Vector3 {
  return new Vector3(value.x, value.y, value.z)
}

function pyToJsBytes(value: DataView): Uint8Array {
  if (value === undefined) {
    return new Uint8Array()
  }
  return new Uint8Array(value.buffer)
}

function jsToPyBytes(value: Uint8Array): DataView {
  return new DataView(value.buffer, 0, value.byteLength)
}

// create reactive object
function bindReactive<T extends Record<string, any>, K extends keyof T>(key: K, model: AnyModel<T>, pyToJs: (value: T[K]) => any = (v) => v): Reactive<any> {
  const refValue = reactive(pyToJs(model.get(key)))
  let lastLocalChangeTime = 0
  model.on(`change:${String(key)}`, () => {
    // Ignore echoed values from Python that arrive after a local JS change;
    // these are stale round-trips that cause camera shaking during interaction.
    if (Date.now() - lastLocalChangeTime < 200) return
    if (JSON.stringify(refValue) === JSON.stringify(model.get(key))) {
      return
    }
    Object.assign(refValue, pyToJs(model.get(key)))
  })
  watch(refValue, (newValue: any) => {
    lastLocalChangeTime = Date.now()
    // need to create a copy of the value because anywidget simply compares by reference
    model.set(key, JSON.parse(JSON.stringify(newValue)))
    model.save_changes()
  })
  return refValue
}

// create shallow ref
function bindShallowRef<T extends Record<string, any>, K extends keyof T>(
  key: K,
  model: AnyModel<T>,
  listen: boolean = true,
  pyToJs: (value: T[K]) => any = (v) => v,
  jsToPy: (value: any) => T[K] = (v) => v,
): ShallowRef<any> {
  const refValue = shallowRef(pyToJs(model.get(key)))
  if (listen) {
    model.on(`change:${String(key)}`, () => {
      refValue.value = pyToJs(model.get(key))
    })
  }
  watch(refValue, (newValue: any) => {
    // need to create a copy of the value because anywidget simply compares by reference
    model.set(key, jsToPy(newValue))
    model.save_changes()
  })
  return refValue
}

type CustomMessage = {
  type: 'console'
  level: 'log' | 'warn' | 'error'
  text: string
}

function onCustomMessage(msg: CustomMessage) {
  if (msg.type === 'console') {
    assert(msg.level === 'log' || msg.level === 'warn' || msg.level === 'error')
    console[msg.level](msg.text)
  }
}
