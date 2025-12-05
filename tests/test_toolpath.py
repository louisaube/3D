"""Tests for toolpath generation module."""

import numpy as np
import pytest

from pycam3d.toolpath import (
    Tool,
    ToolType,
    Toolpath,
    ToolpathPoint,
    Strategy,
)


class TestTool:
    """Tests for Tool class."""

    def test_ball_tool(self):
        """Test creating ball nose tool."""
        tool = Tool.ball(diameter=6.0, length=50.0)

        assert tool.type == ToolType.BALL
        assert tool.diameter == 6.0
        assert tool.length == 50.0
        assert "ball" in tool.name.lower()

    def test_flat_tool(self):
        """Test creating flat end mill."""
        tool = Tool.flat(diameter=10.0)

        assert tool.type == ToolType.FLAT
        assert tool.diameter == 10.0

    def test_bull_tool(self):
        """Test creating bull nose tool."""
        tool = Tool.bull(diameter=12.0, corner_radius=2.0)

        assert tool.type == ToolType.BULL
        assert tool.diameter == 12.0
        assert tool.corner_radius == 2.0

    def test_cone_tool(self):
        """Test creating conical tool."""
        tool = Tool.cone(diameter=6.0, angle=30.0)

        assert tool.type == ToolType.CONE
        assert tool.angle == 30.0

    def test_custom_name(self):
        """Test custom tool name."""
        tool = Tool(
            type=ToolType.BALL,
            diameter=6.0,
            name="My Custom Tool"
        )

        assert tool.name == "My Custom Tool"

    def test_auto_name(self):
        """Test automatic tool name generation."""
        tool = Tool.ball(diameter=6.0)
        assert "6.0" in tool.name or "6" in tool.name


class TestToolpathPoint:
    """Tests for ToolpathPoint class."""

    def test_init(self):
        """Test point initialization."""
        point = ToolpathPoint(x=10.0, y=20.0, z=5.0)

        assert point.x == 10.0
        assert point.y == 20.0
        assert point.z == 5.0
        assert point.rapid is False

    def test_rapid_point(self):
        """Test rapid point."""
        point = ToolpathPoint(x=0, y=0, z=10, rapid=True)
        assert point.rapid is True

    def test_as_tuple(self):
        """Test conversion to tuple."""
        point = ToolpathPoint(x=1.0, y=2.0, z=3.0)
        t = point.as_tuple()

        assert t == (1.0, 2.0, 3.0)

    def test_as_array(self):
        """Test conversion to numpy array."""
        point = ToolpathPoint(x=1.0, y=2.0, z=3.0)
        arr = point.as_array()

        assert isinstance(arr, np.ndarray)
        assert np.allclose(arr, [1.0, 2.0, 3.0])


class TestToolpath:
    """Tests for Toolpath class."""

    def test_init(self):
        """Test toolpath initialization."""
        tp = Toolpath()

        assert len(tp) == 0
        assert tp.feed_rate == 1000.0
        assert tp.safe_z == 10.0

    def test_add_point(self):
        """Test adding points."""
        tp = Toolpath()
        tp.add_point(0, 0, 10)
        tp.add_point(10, 0, 0)
        tp.add_point(10, 10, 0)

        assert len(tp) == 3

    def test_iteration(self):
        """Test iterating over points."""
        tp = Toolpath()
        tp.add_point(0, 0, 0)
        tp.add_point(10, 10, 10)

        points = list(tp)
        assert len(points) == 2
        assert points[0].x == 0
        assert points[1].x == 10

    def test_get_bounds(self):
        """Test getting bounding box."""
        tp = Toolpath()
        tp.add_point(0, 0, 0)
        tp.add_point(10, 20, 5)
        tp.add_point(-5, 10, 15)

        min_bounds, max_bounds = tp.get_bounds()

        assert np.allclose(min_bounds, [-5, 0, 0])
        assert np.allclose(max_bounds, [10, 20, 15])

    def test_get_bounds_empty(self):
        """Test bounds of empty toolpath."""
        tp = Toolpath()
        min_bounds, max_bounds = tp.get_bounds()

        assert np.allclose(min_bounds, [0, 0, 0])
        assert np.allclose(max_bounds, [0, 0, 0])

    def test_get_total_length(self):
        """Test calculating total length."""
        tp = Toolpath()
        tp.add_point(0, 0, 0)
        tp.add_point(10, 0, 0)
        tp.add_point(10, 10, 0)

        length = tp.get_total_length()

        # 10 + 10 = 20
        assert np.isclose(length, 20.0)

    def test_get_total_length_3d(self):
        """Test length calculation in 3D."""
        tp = Toolpath()
        tp.add_point(0, 0, 0)
        tp.add_point(3, 4, 0)  # Distance = 5 (3-4-5 triangle)

        length = tp.get_total_length()
        assert np.isclose(length, 5.0)

    def test_optimize(self):
        """Test toolpath optimization (removing collinear points)."""
        tp = Toolpath()
        # Add collinear points
        for x in range(11):
            tp.add_point(x, 0, 0)

        initial_count = len(tp)
        tp.optimize()

        # Should reduce to just start and end
        assert len(tp) < initial_count
        # Should keep first and last
        assert tp.points[0].x == 0
        assert tp.points[-1].x == 10


class TestStrategy:
    """Tests for Strategy enum."""

    def test_strategies_exist(self):
        """Test that all strategies are defined."""
        assert Strategy.PARALLEL_X is not None
        assert Strategy.PARALLEL_Y is not None
        assert Strategy.ZIGZAG_X is not None
        assert Strategy.ZIGZAG_Y is not None
        assert Strategy.WATERLINE is not None
        assert Strategy.SPIRAL is not None

    def test_strategy_values(self):
        """Test strategy string values."""
        assert Strategy.PARALLEL_X.value == "parallel_x"
        assert Strategy.WATERLINE.value == "waterline"
