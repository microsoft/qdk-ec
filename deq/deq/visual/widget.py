import os
import json
import pathlib
import hashlib
import anywidget
import traitlets
from typing import Any, TypeAlias, Union
import deq.proto.deq_bin_pb2 as pb2
import deq.proto.visualizer_pb2 as vis_pb2
from IPython.display import display, HTML

# there is known memory leak in jupyter notebook where anywidget didn't call the dispose function
#     refreshing the page solves the problem
# only use DEV mode when developing the frontend code
# _DEV = True
_DEV = False


if _DEV:
    ESM = "http://localhost:5173/src/lib.ts?anywidget"
    js_hash = ""
else:
    compressed_js_path = pathlib.Path(__file__).parent / "static" / "lib.js.b64"
    filesize = 0
    js_hash = ""
    if os.path.exists(compressed_js_path):
        with open(compressed_js_path, "r", encoding="utf8") as f:
            compressed_js_code = f.read()
        filesize = len(compressed_js_code) // 1024
        js_hash = hashlib.sha256(compressed_js_code.encode()).hexdigest()[:16]
    ESM = """
    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms))
    }
    async function async_render(model, el) {
        const js_hash = model.get("_js_hash")
        const library_name = `visualqec_${js_hash}`
        const loading = document.createElement("span")
        loading.innerHTML = 'waiting for visual-qec embedded (you should see a green tag on the web page saying "VisualQEC embedded")'
        loading.style.cssText = "color: white; font-size: 10px; height: 20px; padding: 4px; background-color: darkorange;"
        el.appendChild(loading)
         // try it every 100ms, until 100s
        for (let i = 0; i < 1000; ++i) {
            if (window[library_name] != null) {
                break
            }
            if (i != 0 && i % 10 == 0) console.log(`window.hyperion_visual not ready, tried ${i} times`)
            await sleep(100)
        }
        el.removeChild(loading)
        if (window[library_name] != null) {
            const render = window[library_name].default.render
            render({ model, el })
        } else {
            const error = document.createElement("span")
            error.innerHTML = 'failed to find visual-qec library, make sure you call `embed_display()` at the beginning'
            error.style.cssText = "color: white; font-size: 10px; height: 20px; padding: 4px; background-color: darkred;"
            el.appendChild(error)
        }
    }
    function render({ model, el }) {
        async_render(model, el)
    }
    function initialize({ model }) {
        return () => {
            console.log('unmounting app uid', model.app._uid)
            model.app?.unmount()
        }
    }
    export default { render, initialize };
"""

library_embedded = False


def embed_display(force: bool = True) -> None:
    global library_embedded
    if library_embedded and not force:
        return
    if _DEV:
        display(
            HTML(
                """
<span style="color: white; font-size: 10px; height: 20px; padding: 4px; background-color: darkred; border-radius: 6px;" title="change the _DEV flag in deq.visual.widget to persist widget in jupyter notebook">
    VisualQEC Dev Mode (library not embedded)
</span>
"""
            )
        )
    else:
        assert os.path.exists(compressed_js_path)
        display(
            HTML(
                f"""
<span style="color: white; font-size: 10px; height: 20px; padding: 4px; background-color: rgba(36, 110, 36); border-radius: 6px;" title="make sure that you see this block when saving jupyter notebook, otherwise the renderer won't work when reopening">
    VisualQEC embedded ({filesize}kB)
</span>
<script type="module" id='visualqec-{js_hash}'>
    const base64_str = {json.dumps(compressed_js_code)}
    const base64_binary = Uint8Array.from(atob(base64_str), c => c.charCodeAt(0)).buffer
    const blob = new Blob([base64_binary])
    const decompressed_stream = blob.stream()
        .pipeThrough(new DecompressionStream('gzip'))
    const decompressed = await  new Response(decompressed_stream).arrayBuffer()
    const text_decoder = new TextDecoder("utf-8")
    const js_code = text_decoder.decode(decompressed)
    const url = URL.createObjectURL(new Blob([js_code], {{ type: "text/javascript" }}))
    const mod = await import(url);
    URL.revokeObjectURL(url);
    window.visualqec_{js_hash} = mod
    window.visualqec_{js_hash}_b64 = base64_str // for export functionality
</script>
"""
            )
        )
    library_embedded = True


class _BytesOrMemoryview(traitlets.Bytes):
    """Bytes traitlet that also accepts memoryview (from anywidget binary sync)."""

    def validate(self, obj: Any, value: Any) -> bytes:
        if isinstance(value, memoryview):
            value = bytes(value)
        result: bytes = super().validate(obj, value)  # type: ignore[assignment]
        return result


class Widget(anywidget.AnyWidget):
    _esm = ESM
    _selected = _BytesOrMemoryview(b"").tag(sync=True)
    _hovered = _BytesOrMemoryview(b"").tag(sync=True)
    width = traitlets.Unicode("800px").tag(sync=True)
    height = traitlets.Unicode("600px").tag(sync=True)
    _library = traitlets.Bytes().tag(sync=True, readonly=True)
    cameraPosition = traitlets.Dict({"x": 0, "y": 0, "z": 10}).tag(sync=True)
    displayMode = traitlets.Dict({}).tag(sync=True)
    orbitTarget = traitlets.Dict({"x": 0, "y": 0, "z": 0}).tag(sync=True)
    cameraType = traitlets.Unicode("perspective").tag(sync=True)
    gateStyle = traitlets.Unicode("top").tag(sync=True)
    background = traitlets.Unicode("#f0f0f0").tag(sync=True)
    _js_hash = traitlets.Unicode(js_hash).tag(sync=True)

    def __init__(
        self,
        library: bytes | pb2.Library,
        select: "list[SelectableElement] | SelectableElement | None" = None,
        hover: "list[SelectableElement] | SelectableElement | None" = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(library, bytes):
            kwargs["_library"] = library
        elif isinstance(library, pb2.Library):
            kwargs["_library"] = bytes(library.SerializeToString())
        else:
            raise ValueError("library must be pb2.Library or bytes")
        if select is not None:
            select_list = select if isinstance(select, list) else [select]
            kwargs["_selected"] = bytes(vis_pb2.MultiSelectable(
                elements=[elementToSelectable(element) for element in select_list]
            ).SerializeToString())
        if hover is not None:
            hover_list = hover if isinstance(hover, list) else [hover]
            kwargs["_hovered"] = bytes(vis_pb2.MultiSelectable(
                elements=[elementToSelectable(element) for element in hover_list]
            ).SerializeToString())
        super().__init__(**kwargs)

    def console_log(self, msg: str) -> None:
        self.send({"type": "console_log", "msg": msg})

    def select(self, *elements: "SelectableElement") -> None:
        self._selected = bytes(vis_pb2.MultiSelectable(
            elements=[elementToSelectable(element) for element in elements]
        ).SerializeToString())

    def hover(self, *elements: "SelectableElement") -> None:
        self._hovered = bytes(vis_pb2.MultiSelectable(
            elements=[elementToSelectable(element) for element in elements]
        ).SerializeToString())

    @property
    def selected(self) -> vis_pb2.MultiSelectable:
        return vis_pb2.MultiSelectable.FromString(self._selected)

    @property
    def hovered(self) -> vis_pb2.MultiSelectable:
        return vis_pb2.MultiSelectable.FromString(self._hovered)

    def display(self) -> None:
        display(self)


def deq_visualizer(
    library: bytes | pb2.Library,
    select: "list[SelectableElement] | SelectableElement | None" = None,
    hover: "list[SelectableElement] | SelectableElement | None" = None,
    **kwargs: Any,
) -> Widget:
    return Widget(library=library, select=select, hover=hover, **kwargs)


SelectableElement: TypeAlias = Union[vis_pb2.Selectable.Gadget, vis_pb2.Selectable.Location]


def elementToSelectable(element: SelectableElement) -> vis_pb2.Selectable:
    if isinstance(element, vis_pb2.Selectable.Gadget):
        return vis_pb2.Selectable(gadget=element)
    elif isinstance(element, vis_pb2.Selectable.Location):
        return vis_pb2.Selectable(location=element)
    else:
        raise ValueError("element must be Gadget or Location")
