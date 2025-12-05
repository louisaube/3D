"""
Toolpath Simulation Engine for PyCAM3D.

Provides:
- Real-time toolpath animation
- Material removal simulation
- Collision detection
- Machining time estimation
- Export for web visualization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any, Iterator, Callable
import numpy as np
import math
import time


class SimulationState(Enum):
    """Simulation playback state."""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    FINISHED = "finished"


class MoveType(Enum):
    """Type of tool movement."""
    RAPID = "rapid"       # G0 - rapid positioning
    LINEAR = "linear"     # G1 - linear interpolation
    CW_ARC = "cw_arc"     # G2 - clockwise arc
    CCW_ARC = "ccw_arc"   # G3 - counter-clockwise arc


@dataclass
class ToolPosition:
    """Current tool position and state."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    spindle_on: bool = False
    spindle_rpm: int = 0
    feed_rate: float = 0.0
    coolant_on: bool = False

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "spindle_on": self.spindle_on,
            "spindle_rpm": self.spindle_rpm,
            "feed_rate": self.feed_rate,
            "coolant_on": self.coolant_on,
        }

    def copy(self) -> ToolPosition:
        return ToolPosition(
            x=self.x, y=self.y, z=self.z,
            spindle_on=self.spindle_on,
            spindle_rpm=self.spindle_rpm,
            feed_rate=self.feed_rate,
            coolant_on=self.coolant_on,
        )


@dataclass
class SimulationMove:
    """A single move in the simulation."""
    move_type: MoveType
    start: np.ndarray
    end: np.ndarray
    feed_rate: float = 0.0      # 0 = rapid
    duration: float = 0.0       # seconds
    distance: float = 0.0       # mm
    is_cutting: bool = False

    # For arcs
    center: Optional[np.ndarray] = None
    radius: float = 0.0

    @property
    def is_rapid(self) -> bool:
        return self.move_type == MoveType.RAPID

    def interpolate(self, t: float) -> np.ndarray:
        """Interpolate position at time t (0-1).

        Args:
            t: Progress from 0 (start) to 1 (end)

        Returns:
            Interpolated position as [x, y, z]
        """
        t = max(0.0, min(1.0, t))

        if self.move_type in [MoveType.RAPID, MoveType.LINEAR]:
            return self.start + t * (self.end - self.start)

        elif self.move_type in [MoveType.CW_ARC, MoveType.CCW_ARC]:
            if self.center is None:
                return self.start + t * (self.end - self.start)

            # Arc interpolation in XY plane
            start_angle = math.atan2(
                self.start[1] - self.center[1],
                self.start[0] - self.center[0]
            )
            end_angle = math.atan2(
                self.end[1] - self.center[1],
                self.end[0] - self.center[0]
            )

            # Handle angle wrapping
            if self.move_type == MoveType.CW_ARC:
                if end_angle >= start_angle:
                    end_angle -= 2 * math.pi
            else:
                if end_angle <= start_angle:
                    end_angle += 2 * math.pi

            angle = start_angle + t * (end_angle - start_angle)
            z = self.start[2] + t * (self.end[2] - self.start[2])

            return np.array([
                self.center[0] + self.radius * math.cos(angle),
                self.center[1] + self.radius * math.sin(angle),
                z
            ])

        return self.end


@dataclass
class SimulationFrame:
    """A single frame of simulation state for export."""
    time: float                  # Elapsed time in seconds
    position: Tuple[float, float, float]
    move_index: int
    progress: float             # 0-1 progress through current move
    is_cutting: bool
    spindle_on: bool
    total_progress: float       # 0-1 overall progress

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "position": list(self.position),
            "move_index": self.move_index,
            "progress": self.progress,
            "is_cutting": self.is_cutting,
            "spindle_on": self.spindle_on,
            "total_progress": self.total_progress,
        }


class SimulationEngine:
    """Engine for simulating toolpath execution."""

    def __init__(self, rapid_feed: float = 5000.0):
        """
        Args:
            rapid_feed: Feed rate for rapid moves (mm/min)
        """
        self.rapid_feed = rapid_feed
        self.moves: List[SimulationMove] = []
        self.total_time: float = 0.0
        self.total_distance: float = 0.0
        self.cutting_distance: float = 0.0
        self.rapid_distance: float = 0.0

        # Playback state
        self.state = SimulationState.STOPPED
        self.current_time: float = 0.0
        self.playback_speed: float = 1.0

        # Callbacks
        self._on_position_change: Optional[Callable[[ToolPosition], None]] = None
        self._on_state_change: Optional[Callable[[SimulationState], None]] = None

    def load_from_toolpath(self, toolpath, safe_z: float = 10.0) -> None:
        """Load simulation from a Toolpath object.

        Args:
            toolpath: Toolpath object with points
            safe_z: Safe Z height for rapids
        """
        self.moves = []
        self.total_time = 0.0
        self.total_distance = 0.0
        self.cutting_distance = 0.0
        self.rapid_distance = 0.0

        points = list(toolpath.points)
        if len(points) < 2:
            return

        feed_rate = getattr(toolpath, 'feed_rate', 1000.0)
        current_pos = np.array([0.0, 0.0, safe_z])

        for i, pt in enumerate(points):
            target = np.array(pt[:3])

            # Determine if this is a cutting move or rapid
            z_change = target[2] - current_pos[2]
            is_cutting = target[2] < safe_z and current_pos[2] < safe_z

            if is_cutting:
                move_type = MoveType.LINEAR
                rate = feed_rate
            else:
                move_type = MoveType.RAPID
                rate = self.rapid_feed

            distance = np.linalg.norm(target - current_pos)
            duration = distance / rate * 60  # Convert to seconds

            move = SimulationMove(
                move_type=move_type,
                start=current_pos.copy(),
                end=target.copy(),
                feed_rate=rate,
                duration=duration,
                distance=distance,
                is_cutting=is_cutting,
            )

            self.moves.append(move)
            self.total_time += duration
            self.total_distance += distance

            if is_cutting:
                self.cutting_distance += distance
            else:
                self.rapid_distance += distance

            current_pos = target

    def load_from_gcode(self, gcode_lines: List[str]) -> None:
        """Load simulation from G-code lines.

        Args:
            gcode_lines: List of G-code lines
        """
        self.moves = []
        self.total_time = 0.0
        self.total_distance = 0.0
        self.cutting_distance = 0.0
        self.rapid_distance = 0.0

        current_pos = np.array([0.0, 0.0, 0.0])
        feed_rate = 1000.0
        current_mode = MoveType.RAPID

        for line in gcode_lines:
            line = line.strip().upper()
            if not line or line.startswith('(') or line.startswith(';'):
                continue

            # Parse G-code
            parts = line.split()
            x, y, z = current_pos.copy()
            f = feed_rate

            for part in parts:
                if part.startswith('G0'):
                    current_mode = MoveType.RAPID
                elif part.startswith('G1'):
                    current_mode = MoveType.LINEAR
                elif part.startswith('G2'):
                    current_mode = MoveType.CW_ARC
                elif part.startswith('G3'):
                    current_mode = MoveType.CCW_ARC
                elif part.startswith('X'):
                    try:
                        x = float(part[1:])
                    except ValueError:
                        pass
                elif part.startswith('Y'):
                    try:
                        y = float(part[1:])
                    except ValueError:
                        pass
                elif part.startswith('Z'):
                    try:
                        z = float(part[1:])
                    except ValueError:
                        pass
                elif part.startswith('F'):
                    try:
                        f = float(part[1:])
                        feed_rate = f
                    except ValueError:
                        pass

            target = np.array([x, y, z])
            distance = np.linalg.norm(target - current_pos)

            if distance > 0.001:  # Ignore tiny moves
                rate = self.rapid_feed if current_mode == MoveType.RAPID else feed_rate
                duration = distance / rate * 60
                is_cutting = current_mode != MoveType.RAPID

                move = SimulationMove(
                    move_type=current_mode,
                    start=current_pos.copy(),
                    end=target.copy(),
                    feed_rate=rate,
                    duration=duration,
                    distance=distance,
                    is_cutting=is_cutting,
                )

                self.moves.append(move)
                self.total_time += duration
                self.total_distance += distance

                if is_cutting:
                    self.cutting_distance += distance
                else:
                    self.rapid_distance += distance

                current_pos = target

    def get_position_at_time(self, t: float) -> Tuple[np.ndarray, int, float, bool]:
        """Get tool position at a specific time.

        Args:
            t: Time in seconds from start

        Returns:
            Tuple of (position, move_index, move_progress, is_cutting)
        """
        if not self.moves:
            return np.array([0.0, 0.0, 0.0]), 0, 0.0, False

        if t <= 0:
            return self.moves[0].start.copy(), 0, 0.0, self.moves[0].is_cutting

        elapsed = 0.0
        for i, move in enumerate(self.moves):
            if elapsed + move.duration >= t:
                progress = (t - elapsed) / move.duration if move.duration > 0 else 1.0
                return move.interpolate(progress), i, progress, move.is_cutting
            elapsed += move.duration

        # Past end
        return self.moves[-1].end.copy(), len(self.moves) - 1, 1.0, False

    def get_state_at_time(self, t: float) -> ToolPosition:
        """Get complete tool state at a specific time."""
        pos, move_idx, _, is_cutting = self.get_position_at_time(t)

        state = ToolPosition(
            x=pos[0],
            y=pos[1],
            z=pos[2],
            spindle_on=is_cutting,
            feed_rate=self.moves[move_idx].feed_rate if self.moves else 0.0,
        )
        return state

    def generate_frames(self, fps: float = 30.0,
                       start_time: float = 0.0,
                       end_time: Optional[float] = None) -> Iterator[SimulationFrame]:
        """Generate simulation frames for animation.

        Args:
            fps: Frames per second
            start_time: Start time in seconds
            end_time: End time in seconds (None = full simulation)

        Yields:
            SimulationFrame objects
        """
        if end_time is None:
            end_time = self.total_time

        dt = 1.0 / fps
        t = start_time

        while t <= end_time:
            pos, move_idx, progress, is_cutting = self.get_position_at_time(t)

            frame = SimulationFrame(
                time=t,
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                move_index=move_idx,
                progress=progress,
                is_cutting=is_cutting,
                spindle_on=is_cutting,
                total_progress=t / self.total_time if self.total_time > 0 else 0.0,
            )
            yield frame
            t += dt

    def export_animation_data(self, fps: float = 30.0,
                             include_path: bool = True) -> Dict[str, Any]:
        """Export complete animation data for web visualization.

        Args:
            fps: Frames per second
            include_path: Include full toolpath coordinates

        Returns:
            Dictionary with all animation data
        """
        frames = list(self.generate_frames(fps))

        data = {
            "metadata": {
                "fps": fps,
                "total_time": self.total_time,
                "total_distance": self.total_distance,
                "cutting_distance": self.cutting_distance,
                "rapid_distance": self.rapid_distance,
                "move_count": len(self.moves),
                "frame_count": len(frames),
            },
            "frames": [f.to_dict() for f in frames],
        }

        if include_path:
            # Export simplified path for visualization
            path_points = []
            path_types = []  # 0 = rapid, 1 = cutting

            for move in self.moves:
                path_points.append(move.start.tolist())
                path_types.append(0 if move.is_rapid else 1)

            if self.moves:
                path_points.append(self.moves[-1].end.tolist())
                path_types.append(path_types[-1] if path_types else 0)

            data["path"] = {
                "points": path_points,
                "types": path_types,
            }

        return data

    def get_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics."""
        return {
            "total_time_seconds": self.total_time,
            "total_time_minutes": self.total_time / 60,
            "total_distance_mm": self.total_distance,
            "cutting_distance_mm": self.cutting_distance,
            "rapid_distance_mm": self.rapid_distance,
            "cutting_percentage": (self.cutting_distance / self.total_distance * 100)
                                  if self.total_distance > 0 else 0,
            "move_count": len(self.moves),
            "avg_feed_rate": sum(m.feed_rate for m in self.moves if m.is_cutting) /
                            max(1, sum(1 for m in self.moves if m.is_cutting)),
        }

    # Playback control methods
    def play(self) -> None:
        """Start or resume playback."""
        if self.state == SimulationState.FINISHED:
            self.current_time = 0.0
        self.state = SimulationState.PLAYING
        if self._on_state_change:
            self._on_state_change(self.state)

    def pause(self) -> None:
        """Pause playback."""
        self.state = SimulationState.PAUSED
        if self._on_state_change:
            self._on_state_change(self.state)

    def stop(self) -> None:
        """Stop and reset playback."""
        self.state = SimulationState.STOPPED
        self.current_time = 0.0
        if self._on_state_change:
            self._on_state_change(self.state)

    def seek(self, time: float) -> None:
        """Seek to a specific time."""
        self.current_time = max(0.0, min(time, self.total_time))

    def seek_percent(self, percent: float) -> None:
        """Seek to a percentage of total time."""
        self.seek(percent * self.total_time)

    def set_speed(self, speed: float) -> None:
        """Set playback speed multiplier."""
        self.playback_speed = max(0.1, min(100.0, speed))

    def step(self, dt: float) -> Optional[ToolPosition]:
        """Advance simulation by dt seconds (real time).

        Args:
            dt: Real-time delta in seconds

        Returns:
            Current tool position, or None if finished
        """
        if self.state != SimulationState.PLAYING:
            return None

        self.current_time += dt * self.playback_speed

        if self.current_time >= self.total_time:
            self.current_time = self.total_time
            self.state = SimulationState.FINISHED
            if self._on_state_change:
                self._on_state_change(self.state)

        pos = self.get_state_at_time(self.current_time)

        if self._on_position_change:
            self._on_position_change(pos)

        return pos

    def on_position_change(self, callback: Callable[[ToolPosition], None]) -> None:
        """Register callback for position changes."""
        self._on_position_change = callback

    def on_state_change(self, callback: Callable[[SimulationState], None]) -> None:
        """Register callback for state changes."""
        self._on_state_change = callback


@dataclass
class StockVoxels:
    """Voxel representation of stock for material removal simulation."""
    resolution: float           # Voxel size in mm
    grid: np.ndarray           # 3D boolean array (True = material present)
    origin: np.ndarray         # Origin point of grid
    dimensions: Tuple[int, int, int]  # Grid dimensions

    @classmethod
    def from_stock(cls, bounds_min: np.ndarray, bounds_max: np.ndarray,
                   resolution: float = 0.5) -> StockVoxels:
        """Create voxel grid from stock bounds.

        Args:
            bounds_min: Minimum corner [x, y, z]
            bounds_max: Maximum corner [x, y, z]
            resolution: Voxel size in mm

        Returns:
            StockVoxels instance
        """
        size = bounds_max - bounds_min
        dims = tuple(int(np.ceil(s / resolution)) for s in size)

        grid = np.ones(dims, dtype=bool)

        return cls(
            resolution=resolution,
            grid=grid,
            origin=bounds_min.copy(),
            dimensions=dims,
        )

    def remove_sphere(self, center: np.ndarray, radius: float) -> int:
        """Remove material in a sphere (ball endmill).

        Args:
            center: Sphere center [x, y, z]
            radius: Sphere radius

        Returns:
            Number of voxels removed
        """
        # Convert to grid coordinates
        local_center = (center - self.origin) / self.resolution

        # Find affected voxels
        radius_voxels = radius / self.resolution
        x_min = max(0, int(local_center[0] - radius_voxels - 1))
        x_max = min(self.dimensions[0], int(local_center[0] + radius_voxels + 2))
        y_min = max(0, int(local_center[1] - radius_voxels - 1))
        y_max = min(self.dimensions[1], int(local_center[1] + radius_voxels + 2))
        z_min = max(0, int(local_center[2] - radius_voxels - 1))
        z_max = min(self.dimensions[2], int(local_center[2] + radius_voxels + 2))

        removed = 0
        r2 = radius_voxels ** 2

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                for z in range(z_min, z_max):
                    if self.grid[x, y, z]:
                        d2 = ((x - local_center[0]) ** 2 +
                              (y - local_center[1]) ** 2 +
                              (z - local_center[2]) ** 2)
                        if d2 <= r2:
                            self.grid[x, y, z] = False
                            removed += 1

        return removed

    def remove_cylinder(self, center: np.ndarray, radius: float, height: float) -> int:
        """Remove material in a cylinder (flat endmill).

        Args:
            center: Cylinder bottom center [x, y, z]
            radius: Cylinder radius
            height: Cylinder height (extends upward)

        Returns:
            Number of voxels removed
        """
        local_center = (center - self.origin) / self.resolution
        radius_voxels = radius / self.resolution
        height_voxels = height / self.resolution

        x_min = max(0, int(local_center[0] - radius_voxels - 1))
        x_max = min(self.dimensions[0], int(local_center[0] + radius_voxels + 2))
        y_min = max(0, int(local_center[1] - radius_voxels - 1))
        y_max = min(self.dimensions[1], int(local_center[1] + radius_voxels + 2))
        z_min = max(0, int(local_center[2]))
        z_max = min(self.dimensions[2], int(local_center[2] + height_voxels + 1))

        removed = 0
        r2 = radius_voxels ** 2

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                d2 = (x - local_center[0]) ** 2 + (y - local_center[1]) ** 2
                if d2 <= r2:
                    for z in range(z_min, z_max):
                        if self.grid[x, y, z]:
                            self.grid[x, y, z] = False
                            removed += 1

        return removed

    def get_volume(self) -> float:
        """Get remaining material volume in mm³."""
        voxel_count = np.sum(self.grid)
        voxel_volume = self.resolution ** 3
        return float(voxel_count * voxel_volume)

    def get_removed_volume(self, original_volume: float) -> float:
        """Get removed material volume."""
        return original_volume - self.get_volume()

    def export_surface_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Export remaining stock as mesh vertices and faces.

        Returns:
            Tuple of (vertices, faces) arrays
        """
        from scipy import ndimage

        # Find surface voxels using erosion
        eroded = ndimage.binary_erosion(self.grid)
        surface = self.grid & ~eroded

        # Convert surface voxels to mesh (simplified - just voxel centers)
        indices = np.argwhere(surface)
        vertices = indices * self.resolution + self.origin + self.resolution / 2

        # Create simple cube faces for each surface voxel
        faces = []
        # This is a simplified representation - for proper mesh,
        # use marching cubes algorithm

        return vertices, np.array(faces) if faces else np.zeros((0, 3), dtype=int)


class MaterialRemovalSimulator:
    """Simulate material removal during machining."""

    def __init__(self, stock_bounds: Tuple[np.ndarray, np.ndarray],
                 resolution: float = 0.5):
        """
        Args:
            stock_bounds: (min, max) corners of stock
            resolution: Voxel resolution in mm
        """
        self.stock = StockVoxels.from_stock(
            stock_bounds[0], stock_bounds[1], resolution
        )
        self.initial_volume = self.stock.get_volume()
        self.tool_radius = 3.0
        self.tool_type = "ball"  # ball, flat

    def set_tool(self, diameter: float, tool_type: str = "ball") -> None:
        """Set tool parameters."""
        self.tool_radius = diameter / 2
        self.tool_type = tool_type

    def process_move(self, start: np.ndarray, end: np.ndarray,
                    steps_per_mm: float = 2.0) -> int:
        """Process a cutting move.

        Args:
            start: Start position
            end: End position
            steps_per_mm: Sample points per mm

        Returns:
            Total voxels removed
        """
        distance = np.linalg.norm(end - start)
        if distance < 0.001:
            return 0

        steps = max(2, int(distance * steps_per_mm))
        total_removed = 0

        for i in range(steps):
            t = i / (steps - 1)
            pos = start + t * (end - start)

            if self.tool_type == "ball":
                removed = self.stock.remove_sphere(pos, self.tool_radius)
            else:
                # Flat endmill - remove cylinder
                removed = self.stock.remove_cylinder(
                    pos, self.tool_radius, self.tool_radius * 2
                )
            total_removed += removed

        return total_removed

    def simulate_toolpath(self, moves: List[SimulationMove],
                         progress_callback: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
        """Simulate complete toolpath.

        Args:
            moves: List of simulation moves
            progress_callback: Optional callback for progress (0-1)

        Returns:
            Simulation results dictionary
        """
        total_removed = 0
        cutting_moves = [m for m in moves if m.is_cutting]

        for i, move in enumerate(cutting_moves):
            removed = self.process_move(move.start, move.end)
            total_removed += removed

            if progress_callback:
                progress_callback((i + 1) / len(cutting_moves))

        final_volume = self.stock.get_volume()
        removed_volume = self.initial_volume - final_volume

        return {
            "initial_volume_mm3": self.initial_volume,
            "final_volume_mm3": final_volume,
            "removed_volume_mm3": removed_volume,
            "removal_percentage": removed_volume / self.initial_volume * 100,
            "voxels_removed": total_removed,
        }

    def get_current_stock_mesh(self) -> Dict[str, Any]:
        """Get current stock state as mesh data."""
        vertices, faces = self.stock.export_surface_mesh()
        return {
            "vertices": vertices.tolist(),
            "faces": faces.tolist(),
            "volume": self.stock.get_volume(),
        }


def create_simulation_from_toolpath(toolpath, safe_z: float = 10.0) -> SimulationEngine:
    """Create a simulation engine from a toolpath.

    Args:
        toolpath: Toolpath object
        safe_z: Safe Z height

    Returns:
        Configured SimulationEngine
    """
    engine = SimulationEngine()
    engine.load_from_toolpath(toolpath, safe_z)
    return engine


def create_simulation_from_gcode(gcode_path: str) -> SimulationEngine:
    """Create a simulation engine from a G-code file.

    Args:
        gcode_path: Path to G-code file

    Returns:
        Configured SimulationEngine
    """
    with open(gcode_path, 'r') as f:
        lines = f.readlines()

    engine = SimulationEngine()
    engine.load_from_gcode(lines)
    return engine
