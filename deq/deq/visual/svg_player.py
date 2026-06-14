"""Generic SVG sequence player widget for Jupyter notebooks.

Provides :class:`SVGPlayer`, an anywidget-based interactive player that
displays a sequence of pre-rendered SVG frames with continuous-time
slider scrubbing, play/pause, adjustable speed, keyboard navigation,
and SVG download.

Example usage::

    from deq.visual.svg_player import SVGPlayer, SVGFrame

    frames = [
        SVGFrame(svg="<svg>...</svg>", timestamp_ns=0, label="t=0"),
        SVGFrame(svg="<svg>...</svg>", timestamp_ns=1000, label="t=1µs"),
    ]
    player = SVGPlayer(frames)  # auto-computes speed for 10s playback
    display(player)
"""


import json
from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import anywidget
import traitlets


@dataclass
class SVGFrame:
    """A single frame in an SVG animation sequence.

    Attributes:
        svg: Complete SVG markup string for this frame.
        timestamp_ns: Timestamp in nanoseconds (used for playback timing).
        label: Human-readable label shown in the player UI.
    """

    svg: str
    timestamp_ns: int
    label: str


# pylint: disable-next=invalid-name
_SVG_PLAYER_ESM = """
export function render({ model, el }) {
  var frames = JSON.parse(model.get("_frames"));
  if (!frames.length) { el.textContent = "No frames"; return; }

  /* continuous time range (in ns) */
  var tMin = frames[0].timestamp_ns;
  var tMax = frames[frames.length - 1].timestamp_ns;
  var tSpan = Math.max(1, tMax - tMin);
  var SLD_MAX = 10000;

  /* controls */
  var root = document.createElement("div");
  root.style.cssText = "font-family:Arial,sans-serif;outline:none;";
  root.tabIndex = 0;

  var bar = document.createElement("div");
  bar.style.cssText = "display:flex;align-items:center;gap:8px;padding:4px 0;flex-wrap:wrap;";

  var pbtn = document.createElement("button");
  pbtn.textContent = "\\u25b6 Play";
  pbtn.style.cssText = "padding:3px 12px;cursor:pointer;font-size:13px;border-radius:4px;border:1px solid #999;background:#f4f4f4;";

  var prevBtn = document.createElement("button");
  prevBtn.textContent = "\\u25c0";
  prevBtn.title = "Previous frame";
  prevBtn.style.cssText = "padding:3px 8px;cursor:pointer;font-size:13px;border-radius:4px;border:1px solid #999;background:#f4f4f4;";

  var nextBtn = document.createElement("button");
  nextBtn.textContent = "\\u25b6";
  nextBtn.title = "Next frame";
  nextBtn.style.cssText = "padding:3px 8px;cursor:pointer;font-size:13px;border-radius:4px;border:1px solid #999;background:#f4f4f4;";

  var sld = document.createElement("input");
  sld.type = "range"; sld.min = "0"; sld.max = String(SLD_MAX); sld.value = "0";
  sld.style.cssText = "flex:1;min-width:180px;cursor:pointer;";

  var info = document.createElement("span");
  info.style.cssText = "font-size:12px;color:#333;min-width:200px;white-space:nowrap;";

  var spWrap = document.createElement("label");
  spWrap.style.fontSize = "12px";
  spWrap.textContent = "Speed: ";
  var spIn = document.createElement("input");
  spIn.type = "number"; spIn.value = String(model.get("speed_ratio"));
  spIn.min = "1"; spIn.step = "10";
  spIn.style.cssText = "width:55px;font-size:12px;text-align:center;";
  spWrap.appendChild(spIn);
  spWrap.appendChild(document.createTextNode(" trace-ms/s"));

  var dlBtn = document.createElement("button");
  dlBtn.textContent = "\\u2b07 SVG";
  dlBtn.title = "Download current frame as SVG";
  dlBtn.style.cssText = "padding:3px 10px;cursor:pointer;font-size:12px;border-radius:4px;border:1px solid #999;background:#f4f4f4;";

  bar.append(pbtn, prevBtn, sld, nextBtn, info, spWrap, dlBtn);
  root.appendChild(bar);
  var svgBox = document.createElement("div");
  root.appendChild(svgBox);
  el.appendChild(root);

  /* map continuous trace-time (ns) to the most recent frame index */
  function frameAtTime(t_ns) {
    var idx = 0;
    for (var i = frames.length - 1; i >= 0; i--) {
      if (t_ns >= frames[i].timestamp_ns) { idx = i; break; }
    }
    return idx;
  }

  /* state */
  var curFrame = -1;
  var curTime = tMin;

  function setTime(t_ns) {
    t_ns = Math.max(tMin, Math.min(tMax, t_ns));
    curTime = t_ns;
    sld.value = String(Math.round((t_ns - tMin) / tSpan * SLD_MAX));
    var fi = frameAtTime(t_ns);
    var relUs = Math.max(0, (t_ns - tMin) / 1e3);
    info.textContent = "Frame " + (fi + 1) + "/" + frames.length + "  \\u2022  slider: +" + Math.round(relUs) + " \\u00b5s  \\u2022  " + frames[fi].label;
    if (fi !== curFrame) {
      curFrame = fi;
      svgBox.innerHTML = frames[fi].svg;
      model.set("frame_index", fi); model.save_changes();
    }
  }

  setTime(tMin);

  sld.addEventListener("input", function() {
    var frac = parseInt(sld.value) / SLD_MAX;
    setTime(tMin + frac * tSpan);
  });

  prevBtn.addEventListener("click", function() {
    var fi = Math.max(0, (curFrame >= 0 ? curFrame : 0) - 1);
    setTime(frames[fi].timestamp_ns);
  });

  nextBtn.addEventListener("click", function() {
    var fi = Math.min(frames.length - 1, (curFrame >= 0 ? curFrame : 0) + 1);
    setTime(frames[fi].timestamp_ns);
  });

  /* play/pause with requestAnimationFrame */
  var animId = null, playStart = null, playTimeOrigin = 0;

  function stopPlay() {
    if (animId) { cancelAnimationFrame(animId); animId = null; }
    playStart = null;
    pbtn.textContent = "\\u25b6 Play";
    model.set("playing", false); model.save_changes();
  }

  function tick(wallNow) {
    if (playStart === null) { playStart = wallNow; }
    var sp = parseFloat(spIn.value) || 100;
    var elapsed_wall_ms = wallNow - playStart;
    var trace_ns = playTimeOrigin + elapsed_wall_ms * sp * 1000;
    if (trace_ns >= tMax) {
      setTime(tMax);
      stopPlay();
      return;
    }
    setTime(trace_ns);
    animId = requestAnimationFrame(tick);
  }

  pbtn.addEventListener("click", function() {
    if (animId) { stopPlay(); return; }
    pbtn.textContent = "\\u23f8 Pause";
    model.set("playing", true); model.save_changes();
    playTimeOrigin = (curTime >= tMax) ? tMin : curTime;
    playStart = null;
    animId = requestAnimationFrame(tick);
  });

  spIn.addEventListener("change", function() {
    var v = parseFloat(spIn.value) || 100;
    model.set("speed_ratio", v); model.save_changes();
    if (animId) { playTimeOrigin = curTime; playStart = null; }
  });

  dlBtn.addEventListener("click", function() {
    var fi = curFrame >= 0 ? curFrame : 0;
    var s = frames[fi].svg;
    var b = new Blob([s], { type: "image/svg+xml" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(b);
    a.download = "frame_" + String(fi).padStart(3, "0") + ".svg";
    a.click(); URL.revokeObjectURL(a.href);
  });

  root.addEventListener("keydown", function(e) {
    if (e.key === "ArrowLeft") {
      var fi = Math.max(0, (curFrame >= 0 ? curFrame : 0) - 1);
      setTime(frames[fi].timestamp_ns); e.preventDefault();
    } else if (e.key === "ArrowRight") {
      var fi = Math.min(frames.length - 1, (curFrame >= 0 ? curFrame : 0) + 1);
      setTime(frames[fi].timestamp_ns); e.preventDefault();
    } else if (e.key === " ") { pbtn.click(); e.preventDefault(); }
  });

  model.on("change:frame_index", function() {
    var v = model.get("frame_index");
    if (v !== curFrame && v >= 0 && v < frames.length) setTime(frames[v].timestamp_ns);
  });
  return function() { stopPlay(); };
}
"""


class SVGPlayer(anywidget.AnyWidget):  # pylint: disable=abstract-method
    """Interactive player widget for a sequence of SVG frames.

    Controls:
        Slider: scrub continuously through the time range.
        Play / Pause: animate at the given speed ratio.
        Speed: trace-ms per real-second (auto-computed for 10s playback
            when *speed_ratio* is ``None``).
        SVG button: download the current frame as an SVG file.
        Arrow keys: step to previous / next frame.
        Space: toggle play / pause.
    """

    _esm = _SVG_PLAYER_ESM
    _frames = traitlets.Unicode("[]").tag(sync=True)
    frame_index = traitlets.Int(0).tag(sync=True)
    playing = traitlets.Bool(False).tag(sync=True)
    speed_ratio = traitlets.Float(100.0).tag(sync=True)

    def __init__(
        self,
        frames: Sequence[SVGFrame],
        *,
        speed_ratio: Optional[float] = None,
        **kwargs: object,
    ) -> None:
        if speed_ratio is None:
            if len(frames) >= 2:
                span_ms = (frames[-1].timestamp_ns - frames[0].timestamp_ns) / 1e6
                speed_ratio = max(1.0, span_ms / 10.0)
            else:
                speed_ratio = 1.0
        kwargs["_frames"] = json.dumps([asdict(f) for f in frames])
        kwargs["speed_ratio"] = speed_ratio
        super().__init__(**kwargs)
