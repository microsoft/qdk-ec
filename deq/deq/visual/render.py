"""Offline headless renderer for the deq visualizer.

Produces PNG screenshots of 3D visualizer scenes using Playwright + headless Chromium.
Requires: pip install deq[render] && playwright install chromium
"""

import asyncio
import base64
import json
import pathlib
from collections.abc import Mapping
from typing import Any

import deq.proto.deq_bin_pb2 as pb2
import deq.proto.visualizer_pb2 as vis_pb2

from deq.visual.widget import SelectableElement, elementToSelectable

_STATIC_DIR = pathlib.Path(__file__).parent / "static"

_RENDER_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<style>
  html, body { margin: 0; padding: 0; width: %(width)dpx; height: %(height)dpx; overflow: hidden; }
</style>
</head>
<body>
<div id="app" style="width: %(width)dpx; height: %(height)dpx;"></div>
<script type="module">
import { bindStatic } from '/lib.js';

const config = %(config_json)s;

const response = await fetch('/library.bin');
const buffer = await response.arrayBuffer();
const library = new Uint8Array(buffer);

const props = {
  _library: library,
  width: config.width + 'px',
  height: config.height + 'px',
};
if (config.cameraPosition) props.cameraPosition = config.cameraPosition;
if (config.orbitTarget) props.orbitTarget = config.orbitTarget;
if (config.displayMode) props.displayMode = config.displayMode;
if (config.cameraType) props.cameraType = config.cameraType;
if (config.gateStyle) props.gateStyle = config.gateStyle;
if (config.background) props.background = config.background;
if (config.selected) props._selected = new Uint8Array(config.selected);
if (config.hovered) props._hovered = new Uint8Array(config.hovered);

bindStatic('#app', props);

// Focus the canvas so SharedRenderer renders every frame (unfocused
// canvases are throttled to 1/30 RAF frames).
requestAnimationFrame(() => {
  const c = document.querySelector('canvas');
  if (c) { c.setAttribute('tabindex', '0'); c.focus(); }
});

// Poll until the canvas render stabilizes (2 consecutive identical frames).
// We skip the very first poll to avoid locking onto a blank canvas before
// the renderer has started.  We also require at least one non-transparent
// frame before declaring ready — a fully transparent canvas means the
// renderer hasn't drawn anything yet.
let lastDataUrl = '';
let stableCount = 0;
let firstPoll = true;
const STABLE_THRESHOLD = 2;
const POLL_MS = 100;

function isTransparent(canvas) {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return false;
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] !== 0) return false;
  }
  return true;
}

const pollInterval = setInterval(() => {
  const canvas = document.querySelector('canvas');
  if (!canvas || canvas.width === 0) return;
  if (isTransparent(canvas)) return;
  const currentUrl = canvas.toDataURL('image/png');
  if (firstPoll) {
    firstPoll = false;
    lastDataUrl = currentUrl;
    return;
  }
  if (currentUrl === lastDataUrl) {
    stableCount++;
    if (stableCount >= STABLE_THRESHOLD) {
      clearInterval(pollInterval);
      window.__RENDER_READY__ = true;
    }
  } else {
    stableCount = 0;
    lastDataUrl = currentUrl;
  }
}, POLL_MS);
</script>
</body>
</html>"""  # pylint: disable=invalid-name


async def _render_async(
    html: str,
    lib_js_bytes: bytes,
    library_bytes: bytes,
    output_path: "pathlib.Path | None",
    width: int,
    height: int,
    device_pixel_ratio: float,
    timeout_ms: int,
) -> bytes:
    """Core async rendering logic using Playwright async API."""
    from playwright.async_api import (
        async_playwright,
    )  # pylint: disable=import-outside-toplevel

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--use-gl=angle",
                "--use-angle=vulkan",
                "--enable-webgl",
                "--ignore-gpu-blocklist",
            ],
        )
        context = await browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=device_pixel_ratio,
        )
        page = await context.new_page()

        async def handle_route(route: Any) -> None:
            url = route.request.url
            if url.endswith("/lib.js"):
                await route.fulfill(
                    status=200,
                    content_type="application/javascript",
                    body=lib_js_bytes,
                )
            elif url.endswith("/library.bin"):
                await route.fulfill(
                    status=200,
                    content_type="application/octet-stream",
                    body=library_bytes,
                )
            else:
                await route.fulfill(
                    status=200,
                    content_type="text/html",
                    body=html,
                )

        await page.route("**/*", handle_route)
        await page.goto("http://deq/_render")
        await page.wait_for_function(
            "window.__RENDER_READY__ === true", timeout=timeout_ms
        )

        # Export the canvas content directly via toDataURL to avoid
        # WebGL context loss issues that occur with page.screenshot()
        data_url: str = await page.evaluate(
            """() => {
            const canvas = document.querySelector('canvas');
            return canvas.toDataURL('image/png');
        }"""
        )

        png_bytes = base64.b64decode(data_url.split(",")[1])

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(png_bytes)

        await browser.close()

    return png_bytes


def render_to_png(
    library: "bytes | pb2.Library",
    output: "str | pathlib.Path | None" = None,
    *,
    width: int = 800,
    height: int = 600,
    camera_position: "dict[str, float] | None" = None,
    orbit_target: "dict[str, float] | None" = None,
    display_mode: "dict[str, Any] | None" = None,
    select: "list[SelectableElement] | SelectableElement | None" = None,
    hover: "list[SelectableElement] | SelectableElement | None" = None,
    camera_type: str = "perspective",
    gate_style: str = "top",
    background: str = "transparent",
    device_pixel_ratio: float = 1.0,
    timeout_ms: int = 30000,
    preview: bool = False,
) -> bytes:
    """Render a deq library to a PNG image using headless Chromium.

    Uses Playwright to launch a headless browser, load the visualizer bundle,
    render the 3D scene, and capture the canvas output as a PNG image.
    Works both in regular Python scripts and inside Jupyter notebooks.

    Set ``preview=True`` to open an interactive 3D widget instead of rendering.
    This lets you orbit the camera and tweak display modes with real-time
    feedback.  A copyable code snippet is shown below the widget that updates
    as you adjust the view — only non-default parameters are included.  When
    you are happy with the view, set ``preview=False`` (or remove it) to
    produce the PNG.

    Args:
        library: Protobuf Library object or serialized bytes.
        output: Optional path to save the PNG file. If None, only returns bytes.
        width: Image width in pixels.
        height: Image height in pixels.
        camera_position: Camera position as {"x": float, "y": float, "z": float}.
        orbit_target: Orbit target as {"x": float, "y": float, "z": float}.
        display_mode: Per-gadget display mode settings.
        select: Selectable element(s) to highlight.
        hover: Selectable element(s) to show as hovered.
        camera_type: Camera projection type ("perspective" or "orthographic").
        gate_style: Gate rendering style ("top" or "front").
        background: CSS color for the scene background. Defaults to "transparent"
            for paper/presentation use. Use "#f0f0f0" for grey.
        device_pixel_ratio: Device pixel ratio for high-DPI rendering.
        timeout_ms: Timeout in milliseconds for rendering.
        preview: If True, open an interactive widget instead of rendering.

    Returns:
        PNG image bytes (when preview=False), or empty bytes (when preview=True).
    """
    if preview:
        _interactive_preview(
            library,
            width=width,
            height=height,
            camera_position=camera_position,
            orbit_target=orbit_target,
            display_mode=display_mode,
            select=select,
            hover=hover,
            camera_type=camera_type,
            gate_style=gate_style,
            background=background,
        )
        return b""
    try:
        import playwright  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import
    except ImportError as e:
        raise ImportError(
            "playwright is required for offline rendering. "
            "Install with: pip install deq[render] && playwright install chromium"
        ) from e

    # Serialize library
    if isinstance(library, pb2.Library):
        library_bytes: bytes = library.SerializeToString()
    elif isinstance(library, bytes):
        library_bytes = library
    else:
        raise ValueError("library must be pb2.Library or bytes")

    # Build config
    config: dict[str, Any] = {"width": width, "height": height}
    if camera_position is not None:
        config["cameraPosition"] = camera_position
    if orbit_target is not None:
        config["orbitTarget"] = orbit_target
    if display_mode is not None:
        config["displayMode"] = display_mode
    if camera_type != "perspective":
        config["cameraType"] = camera_type
    if gate_style != "top":
        config["gateStyle"] = gate_style
    config["background"] = background
    if select is not None:
        select_list = select if isinstance(select, list) else [select]
        config["selected"] = list(
            vis_pb2.MultiSelectable(
                elements=[elementToSelectable(e) for e in select_list]
            ).SerializeToString()
        )
    if hover is not None:
        hover_list = hover if isinstance(hover, list) else [hover]
        config["hovered"] = list(
            vis_pb2.MultiSelectable(
                elements=[elementToSelectable(e) for e in hover_list]
            ).SerializeToString()
        )

    config_json = json.dumps(config)
    html = _RENDER_HTML_TEMPLATE % {
        "width": width,
        "height": height,
        "config_json": config_json,
    }

    # Read the built lib.js bundle
    lib_js_path = _STATIC_DIR / "lib.js"
    if not lib_js_path.exists():
        raise FileNotFoundError(
            f"Visualizer bundle not found at {lib_js_path}. "
            "Run 'npm run build' in deq_visualizer/ first."
        )
    lib_js_bytes = lib_js_path.read_bytes()

    output_path = pathlib.Path(output) if output is not None else None

    coro = _render_async(
        html,
        lib_js_bytes,
        library_bytes,
        output_path,
        width,
        height,
        device_pixel_ratio,
        timeout_ms,
    )

    # If we're already inside an asyncio event loop (e.g. Jupyter notebook),
    # run in a separate thread to avoid blocking. Otherwise, use asyncio.run().
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures  # pylint: disable=import-outside-toplevel
        import sys  # pylint: disable=import-outside-toplevel

        def _run_coro(c: Any) -> bytes:
            # On Windows, the worker thread may inherit a SelectorEventLoop
            # policy (e.g. from Jupyter/tornado) which doesn't support
            # subprocesses.  Playwright needs ProactorEventLoop.
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(
                    asyncio.WindowsProactorEventLoopPolicy()  # type: ignore[attr-defined]
                )
            return asyncio.run(c)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_run_coro, coro).result()

    return asyncio.run(coro)


def _interactive_preview(
    library: "bytes | pb2.Library",
    *,
    width: int,
    height: int,
    camera_position: "dict[str, float] | None" = None,
    orbit_target: "dict[str, float] | None" = None,
    display_mode: "dict[str, Any] | None" = None,
    select: "list[SelectableElement] | SelectableElement | None" = None,
    hover: "list[SelectableElement] | SelectableElement | None" = None,
    camera_type: str = "perspective",
    gate_style: str = "top",
    background: str = "#f0f0f0",
) -> Any:
    """Open an interactive 3D widget with a live code-generation pane.

    Called internally by ``render_to_png(..., preview=True)``.
    """
    from IPython.display import (
        display as ipy_display,
        HTML,
    )  # pylint: disable=import-outside-toplevel
    from deq.visual.widget import (
        Widget,
        embed_display,
    )  # pylint: disable=import-outside-toplevel

    embed_display(force=False)

    kwargs: dict[str, Any] = {
        "width": f"{width}px",
        "height": f"{height}px",
    }
    if camera_position is not None:
        kwargs["cameraPosition"] = camera_position
    if orbit_target is not None:
        kwargs["orbitTarget"] = orbit_target
    if display_mode is not None:
        kwargs["displayMode"] = display_mode
    if camera_type != "perspective":
        kwargs["cameraType"] = camera_type
    if gate_style != "top":
        kwargs["gateStyle"] = gate_style
    if background != "#f0f0f0":
        kwargs["background"] = background

    widget = Widget(library=library, select=select, hover=hover, **kwargs)

    # Unique element id so multiple previews on one page don't collide
    import uuid  # pylint: disable=import-outside-toplevel

    code_id = f"deq_code_{uuid.uuid4().hex[:8]}"
    code_display = ipy_display(
        HTML(
            f'<pre id="{code_id}" style="background:#f5f5f5; padding:8px; '
            f"border:1px solid #ddd; border-radius:4px; font-size:12px; "
            f'white-space:pre-wrap;">'
            f"# Adjust the view above, then copy this snippet\n"
            f"render_to_png(library, output, width=..., height=...)</pre>"
        ),
        display_id=True,
    )

    default_display_mode: dict[str, bool] = {
        "showBlock": True,
        "showRealization": False,
        "showCheckModel": False,
        "showErrorModel": False,
        "showPorts": False,
    }
    default_camera = {"x": 0, "y": 0, "z": 10}
    default_orbit = {"x": 0, "y": 0, "z": 0}

    def _fmt_float(v: float) -> str:
        if v == int(v):
            return str(int(v))
        return f"{v:.4g}"

    def _fmt_dict(d: "Mapping[str, float | int]") -> str:
        items = ", ".join(f'"{k}": {_fmt_float(v)}' for k, v in d.items())
        return "{" + items + "}"

    def _fmt_selectable(elem: vis_pb2.Selectable) -> str:
        field = elem.WhichOneof("e")
        if field == "gadget":
            return f"vis_pb.Selectable.Gadget(gid={elem.gadget.gid})"
        if field == "location":
            return (
                f"vis_pb.Selectable.Location("
                f"gid={elem.location.gid}, "
                f"location_index={elem.location.location_index})"
            )
        if field == "port":
            p = elem.port
            io = f"input={p.input}" if p.HasField("input") else f"output={p.output}"
            return f"vis_pb.Selectable.Port(gid={p.gid}, {io})"
        if field == "observable":
            o = elem.observable
            io = f"input={o.input}" if o.HasField("input") else f"output={o.output}"
            return (
                f"vis_pb.Selectable.Observable("
                f"gid={o.gid}, {io}, "
                f"observableIndex={o.observableIndex})"
            )
        if field == "check":
            return (
                f"vis_pb.Selectable.Check("
                f"cid={elem.check.cid}, "
                f"checkIndex={elem.check.checkIndex})"
            )
        if field == "error":
            return (
                f"vis_pb.Selectable.Error("
                f"eid={elem.error.eid}, "
                f"errorIndex={elem.error.errorIndex})"
            )
        return "vis_pb.Selectable()"

    def _generate_code(*_: Any) -> None:
        parts: list[str] = []

        # Camera position — emit only if non-default
        cam = widget.cameraPosition
        cam_rounded = {k: round(cam.get(k, 0), 4) for k in ("x", "y", "z")}
        if cam_rounded != default_camera:
            parts.append(f"camera_position={_fmt_dict(cam_rounded)}")

        # Orbit target — emit only if non-default
        orb = widget.orbitTarget
        orb_rounded = {k: round(orb.get(k, 0), 4) for k in ("x", "y", "z")}
        if orb_rounded != default_orbit:
            parts.append(f"orbit_target={_fmt_dict(orb_rounded)}")

        # Display mode — emit only gadgets with non-default settings
        dm = widget.displayMode
        non_default_dm: dict[str, dict[str, bool]] = {}
        for gid_str, mode in dm.items():
            diff: dict[str, bool] = {}
            for key, default_val in default_display_mode.items():
                if key in mode and mode[key] != default_val:
                    diff[key] = mode[key]
            if diff:
                non_default_dm[gid_str] = diff
        if non_default_dm:
            dm_parts = []
            for gid_str, diff in sorted(non_default_dm.items()):
                inner = ", ".join(f'"{k}": {str(v)}' for k, v in diff.items())
                dm_parts.append(f'    "{gid_str}": {{{inner}}}')
            parts.append("display_mode={\n" + ",\n".join(dm_parts) + ",\n}")

        # Selected elements — emit if any
        if widget._selected:  # pylint: disable=protected-access
            multi = vis_pb2.MultiSelectable.FromString(
                widget._selected  # pylint: disable=protected-access
            )
            if multi.elements:
                parts.append(
                    f"select=[{', '.join(_fmt_selectable(e) for e in multi.elements)}]"
                )

        if parts:
            code_text = ",\n".join(parts)
        else:
            code_text = "# all parameters are default"

        if code_display is not None:
            code_display.update(
                HTML(
                    f'<pre id="{code_id}" style="background:#f5f5f5; '
                    f"padding:8px; border:1px solid #ddd; border-radius:4px; "
                    f'font-size:12px; white-space:pre-wrap;">'
                    f"{code_text}</pre>"
                )
            )

    widget.observe(
        _generate_code,
        names=["cameraPosition", "orbitTarget", "displayMode", "_selected"],
    )
    ipy_display(widget)
    _generate_code()  # initial render

    return widget
