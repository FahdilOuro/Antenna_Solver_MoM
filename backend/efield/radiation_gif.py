"""
Animated GIF export of 3D Plotly figures (radiation patterns, surface currents).

The camera orbits 360° around the vertical axis of the scene, each view is rendered
with Kaleido (headless Chrome) and the frames are assembled into a looping GIF with Pillow.
The figure data (mesh, colors, colorbar, aspect ratio) is never modified: only the
camera and the decorations (axes, background) of a copy of the figure change.

Rendering speed: the bottleneck is the plotly.js -> PNG pipeline inside Chrome, not
the WebGL rasterization, so Chrome's GPU mode brings no measurable gain (NVIDIA T600:
34 s vs 36 s for 12 views at 3200x2800). Rendering several views in parallel tabs does
(24 s with 4 tabs), hence the `n_workers` parameter.
"""
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image, ImageChops

# Plotly's default 3D camera position, used when the figure does not define one
DEFAULT_CAMERA_EYE = dict(x=1.25, y=1.25, z=1.25)

def _orbit_cameras(fig, n_frames, camera_radius, camera_elevation, clockwise):
    """
    Build the list of camera dictionaries describing a full horizontal orbit.

    The orbit starts from the figure's current view so the first frame matches what
    is displayed in the notebook. The horizontal radius and the height of the eye are
    taken from that view unless overridden.
    """
    eye = fig.layout.scene.camera.eye
    x0 = eye.x if eye.x is not None else DEFAULT_CAMERA_EYE['x']
    y0 = eye.y if eye.y is not None else DEFAULT_CAMERA_EYE['y']
    z0 = eye.z if eye.z is not None else DEFAULT_CAMERA_EYE['z']

    radius = camera_radius if camera_radius is not None else math.hypot(x0, y0)
    height = camera_elevation if camera_elevation is not None else z0
    start = math.atan2(y0, x0)

    # endpoint=False: the last frame is not a copy of the first, so the loop is seamless
    angles = np.linspace(0.0, 2 * np.pi, n_frames, endpoint=False)
    direction = -1.0 if clockwise else 1.0

    return [
        dict(
            eye=dict(x=radius * math.cos(start + direction * a),
                     y=radius * math.sin(start + direction * a),
                     z=height),
            up=dict(x=0, y=0, z=1),
        )
        for a in angles
    ]

def _render_frames(figures, frame_paths, width, height, scale, n_workers):
    """
    Render all figures to PNG files in a single Chrome session.

    Kaleido >= 1.0 is driven directly (rather than through plotly.io.write_images)
    so that several tabs can render views in parallel. Older versions fall back to
    one write_image() call per view.
    """
    try:
        import kaleido
        kaleido_v1 = hasattr(kaleido, 'write_fig_from_object_sync')
    except ImportError:
        kaleido_v1 = False

    if not kaleido_v1:
        for figure, frame_path in zip(figures, frame_paths):
            figure.write_image(frame_path, format='png', width=width, height=height, scale=scale)
        return

    specs = [
        dict(fig=figure.to_dict(), path=Path(frame_path),
             opts=dict(format='png', width=width, height=height, scale=scale))
        for figure, frame_path in zip(figures, frame_paths)
    ]
    kopts = dict(n=max(1, min(n_workers, len(specs))))
    if pio.defaults.plotlyjs:
        kopts['plotlyjs'] = pio.defaults.plotlyjs
    kaleido.write_fig_from_object_sync(specs, kopts=kopts)

def _content_bbox(frames, background, padding):
    """
    Smallest box containing every non-background pixel over all frames, so that the
    crop is identical for every view and the animation does not jitter.
    """
    bg = Image.new('RGB', frames[0].size, background)
    boxes = [ImageChops.difference(frame, bg).convert('L').point(lambda v: 255 if v > 8 else 0).getbbox()
             for frame in frames]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
    right, bottom = max(b[2] for b in boxes), max(b[3] for b in boxes)
    pad = int(padding * max(right - left, bottom - top))
    w, h = frames[0].size
    return max(0, left - pad), max(0, top - pad), min(w, right + pad), min(h, bottom + pad)

def _shared_palette(frames, background, n_grays=24):
    """
    256-color palette shared by every view (no flicker between frames).

    The exact background color and a ramp of grays are reserved, so the background
    stays uniform and the anti-aliased colorbar labels are not dithered into colored
    fringes. The remaining entries are adapted to a mosaic of all frames, so colors
    visible only from some angles are not lost.
    """
    thumb_w = max(1, frames[0].width // 4)
    thumb_h = max(1, frames[0].height // 4)
    mosaic = Image.new('RGB', (thumb_w * len(frames), thumb_h))
    for i, frame in enumerate(frames):
        mosaic.paste(frame.resize((thumb_w, thumb_h), Image.Resampling.BOX), (i * thumb_w, 0))

    reserved = [Image.new('RGB', (1, 1), background).getpixel((0, 0))]
    reserved += [(v, v, v) for v in np.linspace(0, 255, n_grays).astype(int)]
    adaptive = mosaic.quantize(colors=256 - len(reserved), method=Image.Quantize.MEDIANCUT)
    adaptive_rgb = adaptive.getpalette()[:3 * (256 - len(reserved))]

    palette = Image.new('P', (1, 1))
    palette.putpalette([c for rgb in reserved for c in rgb] + adaptive_rgb)
    return palette

def create_rotating_gif(fig, output_path, n_frames=12, fps=2, width=800, height=700, scale=2.0,
                        supersample=2, show_axes=False, crop=True, padding=0.03,
                        camera_radius=None, camera_elevation=None, clockwise=False,
                        background='white', dither=True, loop=0, n_workers=4):
    """
    Create an animated GIF of a 3D Plotly figure with the camera orbiting 360° around it.

    Parameters:
        * fig : plotly.graph_objects.Figure, 3D figure to animate (left unchanged).
        * output_path : str, path of the GIF file to write (parent folders are created).
        * n_frames : int, number of views over one full turn (12 -> one view every 30°).
        * fps : float, views per second; one full turn lasts n_frames / fps seconds
          (12 views at 2 fps -> 6 s). For a smoother motion at the same speed, increase
          n_frames and fps together. GIF timing has a 10 ms resolution.
        * width, height : int, size of the plotly layout in pixels (fonts and colorbar
          are laid out relative to it).
        * scale : float, resolution multiplier of the final GIF (2 -> 1600x1400 before crop).
        * supersample : int, extra rendering factor; each view is rendered supersample
          times larger then downscaled with a Lanczos filter (smooth, anti-aliased edges).
        * show_axes : bool, keep the 3D axes, grid and background planes (False -> only
          the radiation pattern and its colorbar are drawn).
        * crop : bool, remove the empty margins around the pattern (same crop for all views).
        * padding : float, margin kept around the content when cropping (fraction of its size).
        * camera_radius : float, horizontal distance of the camera from the scene center,
          in Plotly normalized units. Default: taken from the figure's current camera.
        * camera_elevation : float, height of the camera (z of the eye). Default: taken
          from the figure's current camera.
        * clockwise : bool, rotation direction seen from above.
        * background : str or tuple, background color of the GIF (a GIF has no partial
          transparency, so use the background color of the slide it will be placed on).
        * dither : bool, Floyd-Steinberg dithering to hide color banding of the 256-color
          GIF palette on smooth shading.
        * loop : int, number of GIF repetitions (0 = infinite).
        * n_workers : int, number of Chrome tabs rendering views in parallel.

    Returns:
        str: path of the written GIF file.
    """
    if n_frames < 2:
        raise ValueError("n_frames must be at least 2.")
    if fps <= 0:
        raise ValueError("fps must be strictly positive.")
    supersample = max(1, int(supersample))

    cameras = _orbit_cameras(fig, n_frames, camera_radius, camera_elevation, clockwise)

    # One independent copy per view: only the camera and the decorations differ
    figures = []
    for camera in cameras:
        frame_fig = go.Figure(fig)
        frame_fig.update_layout(scene_camera=camera,
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        if not show_axes:
            frame_fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
        figures.append(frame_fig)

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    final_size = (int(round(width * scale)), int(round(height * scale)))
    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_paths = [os.path.join(tmp_dir, f"frame_{i:04d}.png") for i in range(n_frames)]
        print(f"Rendering {n_frames} views at {final_size[0] * supersample}x{final_size[1] * supersample} px...")
        _render_frames(figures, frame_paths, width, height, scale * supersample, n_workers)

        rgb_frames = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                rgba = image.convert('RGBA')
            canvas = Image.new('RGBA', rgba.size, background)
            rgb = Image.alpha_composite(canvas, rgba).convert('RGB')
            if supersample > 1:
                rgb = rgb.resize(final_size, Image.Resampling.LANCZOS)
            rgb_frames.append(rgb)

    if crop:
        bbox = _content_bbox(rgb_frames, background, padding)
        if bbox is not None:
            rgb_frames = [frame.crop(bbox) for frame in rgb_frames]

    palette = _shared_palette(rgb_frames, background)
    dither_mode = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    background_rgb = np.array(Image.new('RGB', (1, 1), background).getpixel((0, 0)), dtype=np.uint8)
    gif_frames = []
    for frame in rgb_frames:
        indexed = np.array(frame.quantize(palette=palette, dither=dither_mode))
        # Pillow's palette lookup is approximate and may pick a near-white adaptive entry:
        # map the background explicitly to its reserved, exact entry (index 0)
        indexed[np.all(np.asarray(frame) == background_rgb, axis=-1)] = 0
        gif_frame = Image.fromarray(indexed, mode='P')
        gif_frame.putpalette(palette.getpalette())
        gif_frames.append(gif_frame)

    duration_ms = max(10, int(round(1000 / fps / 10)) * 10)
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=False,
    )
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"GIF saved: {output_path} ({n_frames} frames, {gif_frames[0].width}x{gif_frames[0].height} px, "
          f"{duration_ms} ms/frame, {size_mb:.1f} MB)")
    return output_path
