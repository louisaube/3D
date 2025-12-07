"""
Web API and 3D visualization server for PyCAM3D.

Provides:
- REST API for mesh processing and toolpath generation
- Interactive 3D visualization with Three.js
- Real-time toolpath preview
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING

import numpy as np

# Import FastAPI types for annotations (lazy import actual modules in create_app)
try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Define stubs for type checking
    UploadFile = Any
    File = None

logger = logging.getLogger(__name__)


# Define request model at module level for FastAPI annotation resolution
if FASTAPI_AVAILABLE:
    class ToolpathRequest(BaseModel):
        """Request model for toolpath generation."""
        mesh_id: str
        strategy: str = "iso-scallop"
        tool_type: str = "ball"
        tool_diameter: float = 6.0
        stepover: float = 0.15
        feed_rate: float = 1000.0
        spindle_rpm: int = 12000


# HTML template with Three.js visualization
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PyCAM3D - 3D Toolpath Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e8e8e8;
            overflow: hidden;
        }
        #container { display: flex; height: 100vh; }
        #viewer { flex: 1; position: relative; }
        #sidebar {
            width: 380px;
            background: rgba(20, 20, 35, 0.95);
            backdrop-filter: blur(10px);
            border-left: 1px solid rgba(255,255,255,0.1);
            padding: 20px;
            overflow-y: auto;
        }
        h1 {
            font-size: 1.5rem;
            margin-bottom: 20px;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        h2 { font-size: 1rem; color: #888; margin: 20px 0 10px; text-transform: uppercase; letter-spacing: 1px; }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .form-group { margin-bottom: 14px; }
        label { display: block; font-size: 0.85rem; color: #aaa; margin-bottom: 6px; }
        input, select {
            width: 100%;
            padding: 10px 14px;
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 8px;
            background: rgba(0,0,0,0.3);
            color: #fff;
            font-size: 0.95rem;
            transition: all 0.2s;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #00d4ff;
            box-shadow: 0 0 0 3px rgba(0,212,255,0.15);
        }
        input[type="range"] { padding: 0; height: 6px; }
        .range-value { float: right; color: #00d4ff; font-weight: 600; }
        button {
            width: 100%;
            padding: 14px;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 10px;
        }
        .btn-primary {
            background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
            color: #000;
        }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,212,255,0.4); }
        .btn-secondary {
            background: rgba(255,255,255,0.1);
            color: #fff;
        }
        .btn-success {
            background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
            color: #000;
        }
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .stat-item {
            background: rgba(0,212,255,0.1);
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value { font-size: 1.4rem; font-weight: 700; color: #00d4ff; }
        .stat-label { font-size: 0.75rem; color: #888; margin-top: 4px; }
        #status {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.8);
            padding: 12px 24px;
            border-radius: 30px;
            font-size: 0.9rem;
            opacity: 0;
            transition: opacity 0.3s;
        }
        #status.show { opacity: 1; }
        .loading {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid #00d4ff;
            border-top-color: transparent;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 8px;
            vertical-align: middle;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        #drop-zone {
            border: 2px dashed rgba(255,255,255,0.2);
            border-radius: 12px;
            padding: 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        #drop-zone:hover, #drop-zone.dragover {
            border-color: #00d4ff;
            background: rgba(0,212,255,0.1);
        }
        #drop-zone svg { width: 48px; height: 48px; margin-bottom: 10px; opacity: 0.5; }
        .hidden { display: none !important; }
        #controls-hint {
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(0,0,0,0.6);
            padding: 10px 16px;
            border-radius: 8px;
            font-size: 0.8rem;
            color: #888;
        }
    </style>
</head>
<body>
    <div id="container">
        <div id="viewer"></div>
        <div id="sidebar">
            <h1>PyCAM3D</h1>

            <div id="upload-section" class="card">
                <div id="drop-zone">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="17 8 12 3 7 8"/>
                        <line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                    <div>Drop STL/OBJ file here</div>
                    <div style="font-size: 0.8rem; color: #666; margin-top: 5px;">or click to browse</div>
                    <input type="file" id="file-input" accept=".stl,.obj,.ply" style="display:none">
                </div>
            </div>

            <div id="mesh-info" class="card hidden">
                <h2>Mesh Info</h2>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="stat-vertices">-</div>
                        <div class="stat-label">Vertices</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="stat-faces">-</div>
                        <div class="stat-label">Faces</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="stat-size">-</div>
                        <div class="stat-label">Size (mm)</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="stat-watertight">-</div>
                        <div class="stat-label">Watertight</div>
                    </div>
                </div>
            </div>

            <div id="toolpath-section" class="card hidden">
                <h2>Toolpath Settings</h2>

                <div class="form-group">
                    <label>Strategy</label>
                    <select id="strategy">
                        <option value="iso-scallop">Iso-Scallop (Adaptive)</option>
                        <option value="spiral">Spiral (Continuous)</option>
                        <option value="parallel">Parallel Lines</option>
                        <option value="waterline">Waterline</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Tool Type</label>
                    <select id="tool-type">
                        <option value="ball">Ball End Mill</option>
                        <option value="flat">Flat End Mill</option>
                        <option value="bull">Bull Nose</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Tool Diameter <span class="range-value" id="tool-dia-val">6</span> mm</label>
                    <input type="range" id="tool-diameter" min="1" max="25" value="6" step="0.5">
                </div>

                <div class="form-group">
                    <label>Stepover <span class="range-value" id="stepover-val">15</span>%</label>
                    <input type="range" id="stepover" min="5" max="50" value="15">
                </div>

                <div class="form-group">
                    <label>Feed Rate <span class="range-value" id="feed-val">1000</span> mm/min</label>
                    <input type="range" id="feed-rate" min="100" max="5000" value="1000" step="100">
                </div>

                <div class="form-group">
                    <label>Spindle Speed <span class="range-value" id="spindle-val">12000</span> RPM</label>
                    <input type="range" id="spindle-rpm" min="1000" max="24000" value="12000" step="1000">
                </div>

                <button class="btn-primary" id="generate-btn">
                    Generate Toolpath
                </button>
            </div>

            <div id="result-section" class="card hidden">
                <h2>Toolpath Result</h2>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="stat-points">-</div>
                        <div class="stat-label">Points</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="stat-length">-</div>
                        <div class="stat-label">Length (mm)</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="stat-time">-</div>
                        <div class="stat-label">Est. Time</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="stat-lines">-</div>
                        <div class="stat-label">G-code Lines</div>
                    </div>
                </div>
                <button class="btn-success" id="download-btn">
                    Download G-code
                </button>
                <button class="btn-secondary" id="animate-btn">
                    Animate Toolpath
                </button>
            </div>
        </div>
    </div>

    <div id="status"></div>
    <div id="controls-hint">🖱 Rotate | Scroll: Zoom | Shift+Drag: Pan</div>

    <script type="importmap">
    {
        "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
        }
    }
    </script>

    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
        import { STLLoader } from 'three/addons/loaders/STLLoader.js';
        import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

        // Three.js setup
        let scene, camera, renderer, controls;
        let meshObject, toolpathLine, toolMarker;
        let currentMeshId = null;
        let gcodeData = null;
        let animationId = null;

        function init() {
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);

            const viewer = document.getElementById('viewer');
            camera = new THREE.PerspectiveCamera(45, viewer.clientWidth / viewer.clientHeight, 0.1, 10000);
            camera.position.set(100, 100, 100);

            try {
                renderer = new THREE.WebGLRenderer({ antialias: true });
            } catch (e) {
                viewer.innerHTML = '<div style="color: #ff6b6b; padding: 20px; text-align: center;">WebGL not available. Please use a modern browser with WebGL support.</div>';
                console.error('WebGL initialization failed:', e);
                return;
            }
            renderer.setSize(viewer.clientWidth, viewer.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            viewer.appendChild(renderer.domElement);

            controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;

            // Lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(50, 100, 50);
            scene.add(directionalLight);

            const backLight = new THREE.DirectionalLight(0x4488ff, 0.3);
            backLight.position.set(-50, -50, -50);
            scene.add(backLight);

            // Grid
            const grid = new THREE.GridHelper(200, 20, 0x444444, 0x333333);
            scene.add(grid);

            // Axes
            const axesHelper = new THREE.AxesHelper(30);
            scene.add(axesHelper);

            window.addEventListener('resize', onWindowResize);
            animate();
        }

        function onWindowResize() {
            const viewer = document.getElementById('viewer');
            camera.aspect = viewer.clientWidth / viewer.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(viewer.clientWidth, viewer.clientHeight);
        }

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }

        function showStatus(msg, duration = 3000) {
            const status = document.getElementById('status');
            status.innerHTML = msg;
            status.classList.add('show');
            setTimeout(() => status.classList.remove('show'), duration);
        }

        // File upload
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');

        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) uploadFile(e.target.files[0]);
        });

        async function uploadFile(file) {
            showStatus('<span class="loading"></span>Uploading mesh...');

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/api/upload', { method: 'POST', body: formData });
                const data = await response.json();

                if (data.error) {
                    showStatus('Error: ' + data.error);
                    return;
                }

                currentMeshId = data.mesh_id;
                
                console.log('Upload response:', data);
                console.log('Vertices count:', data.vertices ? data.vertices.length : 'undefined');
                console.log('Faces count:', data.faces ? data.faces.length : 'undefined');
                
                try {
                    displayMesh(data);
                    showStatus('Mesh loaded successfully!');
                    document.getElementById('mesh-info').classList.remove('hidden');
                    document.getElementById('toolpath-section').classList.remove('hidden');
                } catch (displayErr) {
                    console.error('Display error:', displayErr);
                    showStatus('Error displaying mesh: ' + displayErr.message, 5000);
                }

            } catch (err) {
                console.error('Upload error:', err);
                showStatus('Upload failed: ' + err.message);
            }
        }

        function displayMesh(data) {
            // Check if renderer is available
            if (!renderer) {
                throw new Error('3D viewer not initialized. WebGL may not be supported.');
            }
            
            // Remove existing mesh
            if (meshObject) scene.remove(meshObject);

            // Validate data
            if (!data.vertices || !data.faces) {
                throw new Error('Invalid mesh data received from server');
            }
            
            console.log('Creating mesh with', data.vertices.length, 'vertices and', data.faces.length, 'faces');

            // Create geometry from vertices and faces
            const geometry = new THREE.BufferGeometry();
            const vertices = new Float32Array(data.vertices.flat());
            const indices = new Uint32Array(data.faces.flat());

            geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
            geometry.setIndex(new THREE.BufferAttribute(indices, 1));
            geometry.computeVertexNormals();

            const material = new THREE.MeshPhongMaterial({
                color: 0x00aaff,
                shininess: 80,
                side: THREE.DoubleSide,
            });

            meshObject = new THREE.Mesh(geometry, material);
            scene.add(meshObject);

            // Center camera
            geometry.computeBoundingBox();
            const box = geometry.boundingBox;
            const center = new THREE.Vector3();
            box.getCenter(center);
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);

            camera.position.set(center.x + maxDim, center.y + maxDim, center.z + maxDim);
            controls.target.copy(center);

            // Update stats
            document.getElementById('stat-vertices').textContent = data.stats.vertex_count.toLocaleString();
            document.getElementById('stat-faces').textContent = data.stats.face_count.toLocaleString();
            document.getElementById('stat-size').textContent = `${size.x.toFixed(1)}x${size.y.toFixed(1)}x${size.z.toFixed(1)}`;
            document.getElementById('stat-watertight').textContent = data.stats.is_watertight ? '✓' : '✗';
        }

        // Range input updates
        document.querySelectorAll('input[type="range"]').forEach(input => {
            const valSpan = document.getElementById(input.id.replace('-', '-') + '-val') ||
                           document.getElementById(input.id.replace('tool-diameter', 'tool-dia') + '-val');
            if (valSpan) {
                input.addEventListener('input', () => valSpan.textContent = input.value);
            }
        });

        // Generate toolpath
        document.getElementById('generate-btn').addEventListener('click', async () => {
            if (!currentMeshId) return;

            showStatus('<span class="loading"></span>Generating toolpath...');
            document.getElementById('generate-btn').disabled = true;

            const params = {
                mesh_id: currentMeshId,
                strategy: document.getElementById('strategy').value,
                tool_type: document.getElementById('tool-type').value,
                tool_diameter: parseFloat(document.getElementById('tool-diameter').value),
                stepover: parseFloat(document.getElementById('stepover').value) / 100,
                feed_rate: parseFloat(document.getElementById('feed-rate').value),
                spindle_rpm: parseInt(document.getElementById('spindle-rpm').value),
            };

            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(params)
                });
                const data = await response.json();

                if (data.error) {
                    showStatus('Error: ' + data.error);
                    return;
                }

                displayToolpath(data.toolpath);
                gcodeData = data.gcode;

                document.getElementById('stat-points').textContent = data.stats.points.toLocaleString();
                document.getElementById('stat-length').textContent = data.stats.length.toFixed(0);
                document.getElementById('stat-time').textContent = data.stats.estimated_time;
                document.getElementById('stat-lines').textContent = data.stats.gcode_lines.toLocaleString();

                document.getElementById('result-section').classList.remove('hidden');
                showStatus('Toolpath generated!');

            } catch (err) {
                showStatus('Generation failed: ' + err.message);
            } finally {
                document.getElementById('generate-btn').disabled = false;
            }
        });

        function displayToolpath(points) {
            if (toolpathLine) scene.remove(toolpathLine);

            const rapidPoints = [];
            const cutPoints = [];
            let currentPoints = cutPoints;

            for (const pt of points) {
                if (pt.rapid) {
                    if (currentPoints.length > 0) currentPoints = [];
                    rapidPoints.push(new THREE.Vector3(pt.x, pt.z, pt.y));  // Y-up
                } else {
                    currentPoints.push(new THREE.Vector3(pt.x, pt.z, pt.y));
                }
            }

            // Cutting path (blue)
            const cutGeom = new THREE.BufferGeometry().setFromPoints(
                points.filter(p => !p.rapid).map(p => new THREE.Vector3(p.x, p.z, p.y))
            );
            const cutMat = new THREE.LineBasicMaterial({ color: 0x00ff88, linewidth: 2 });
            toolpathLine = new THREE.Line(cutGeom, cutMat);
            scene.add(toolpathLine);
        }

        // Download G-code
        document.getElementById('download-btn').addEventListener('click', () => {
            if (!gcodeData) return;
            const blob = new Blob([gcodeData], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'toolpath.nc';
            a.click();
            URL.revokeObjectURL(url);
        });

        // Animate toolpath
        let simulationData = null;
        let animationFrameIndex = 0;
        let isAnimating = false;
        let animationSpeed = 1.0;

        document.getElementById('animate-btn').addEventListener('click', async () => {
            if (!currentMeshId) return;

            if (isAnimating) {
                stopAnimation();
                return;
            }

            showStatus('<span class="loading"></span>Loading simulation...');

            try {
                const response = await fetch(`/api/simulation/${currentMeshId}`);
                const data = await response.json();

                if (data.error) {
                    showStatus('Error: ' + data.error);
                    return;
                }

                simulationData = data;
                startAnimation();
                showStatus('Animation started! Click button again to stop.');
                document.getElementById('animate-btn').textContent = 'Stop Animation';

            } catch (err) {
                showStatus('Simulation failed: ' + err.message);
            }
        });

        function startAnimation() {
            if (!simulationData || !simulationData.frames) return;

            isAnimating = true;
            animationFrameIndex = 0;

            // Create tool marker if not exists
            if (!toolMarker) {
                const markerGeom = new THREE.SphereGeometry(3, 16, 16);
                const markerMat = new THREE.MeshPhongMaterial({ color: 0xff4444, emissive: 0x441111 });
                toolMarker = new THREE.Mesh(markerGeom, markerMat);
                scene.add(toolMarker);
            }
            toolMarker.visible = true;

            animateFrame();
        }

        function animateFrame() {
            if (!isAnimating || !simulationData) return;

            const frames = simulationData.frames;
            if (animationFrameIndex >= frames.length) {
                animationFrameIndex = 0;  // Loop
            }

            const frame = frames[animationFrameIndex];

            // Update tool marker position (swap Y and Z for Three.js)
            toolMarker.position.set(frame.position[0], frame.position[2], frame.position[1]);

            // Change color based on cutting state
            if (frame.is_cutting) {
                toolMarker.material.color.setHex(0xff4444);
                toolMarker.material.emissive.setHex(0x441111);
            } else {
                toolMarker.material.color.setHex(0x44ff44);
                toolMarker.material.emissive.setHex(0x114411);
            }

            animationFrameIndex += Math.ceil(animationSpeed);

            // Update progress in status
            const progress = (frame.total_progress * 100).toFixed(1);
            const timeStr = frame.time.toFixed(1);
            document.getElementById('status').innerHTML = `Simulating: ${progress}% (${timeStr}s)`;
            document.getElementById('status').classList.add('show');

            animationId = requestAnimationFrame(animateFrame);
        }

        function stopAnimation() {
            isAnimating = false;
            if (animationId) {
                cancelAnimationFrame(animationId);
                animationId = null;
            }
            if (toolMarker) {
                toolMarker.visible = false;
            }
            document.getElementById('animate-btn').textContent = 'Animate Toolpath';
            document.getElementById('status').classList.remove('show');
        }

        init();
    </script>
</body>
</html>
"""


def create_app():
    """Create FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI required: pip install fastapi uvicorn python-multipart")

    import trimesh
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="PyCAM3D", description="3D CAM Toolpath Generator")

    # Add CORS middleware for Replit proxy/iframe support
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store uploaded meshes temporarily
    mesh_store: Dict[str, Any] = {}

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTML_TEMPLATE

    @app.post("/api/upload")
    async def upload_mesh(file: UploadFile = File(...)):
        """Upload and analyze a mesh file."""
        try:
            # Save to temp file
            suffix = Path(file.filename).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            # Load with trimesh
            mesh = trimesh.load_mesh(tmp_path)

            # Generate ID and store
            mesh_id = str(uuid.uuid4())[:8]
            mesh_store[mesh_id] = {
                "mesh": mesh,
                "path": tmp_path,
            }

            # Get stats
            stats = {
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.faces),
                "is_watertight": bool(mesh.is_watertight),
                "bounds_min": mesh.bounds[0].tolist(),
                "bounds_max": mesh.bounds[1].tolist(),
            }

            return JSONResponse({
                "mesh_id": mesh_id,
                "filename": file.filename,
                "stats": stats,
                "vertices": mesh.vertices.tolist(),
                "faces": mesh.faces.tolist(),
            })

        except Exception as e:
            logger.exception("Upload failed")
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.post("/api/generate")
    async def generate_toolpath(request: ToolpathRequest):
        """Generate toolpath for uploaded mesh."""
        try:
            if request.mesh_id not in mesh_store:
                return JSONResponse({"error": "Mesh not found"}, status_code=404)

            mesh_data = mesh_store[request.mesh_id]
            mesh = mesh_data["mesh"]
            stl_path = mesh_data["path"]

            # Import required modules
            import opencamlib as ocl
            from pycam3d.toolpath import Tool, Toolpath
            from pycam3d.gcode import GCodeWriter

            # Load STL for OpenCAMLib
            stl_surf = ocl.STLSurf()
            ocl.STLReader(stl_path, stl_surf)

            bounds = (mesh.bounds[0], mesh.bounds[1])

            # Create tool
            if request.tool_type == "ball":
                tool = Tool.ball(diameter=request.tool_diameter)
            elif request.tool_type == "flat":
                tool = Tool.flat(diameter=request.tool_diameter)
            else:
                tool = Tool.bull(diameter=request.tool_diameter, corner_radius=request.tool_diameter * 0.1)

            # Generate toolpath based on strategy
            if request.strategy == "iso-scallop":
                from pycam3d.strategies import IsoScallopGenerator
                generator = IsoScallopGenerator(mesh, stl_surf, bounds)
                toolpath = generator.generate(
                    tool=tool,
                    target_scallop=0.02,
                    z_safe=mesh.bounds[1][2] + 10,
                    direction="x"
                )
            elif request.strategy == "spiral":
                from pycam3d.strategies import SpiralGenerator
                generator = SpiralGenerator(mesh, stl_surf, bounds)
                toolpath = generator.generate_3d_spiral(
                    tool=tool,
                    stepover=request.tool_diameter * request.stepover,
                    z_safe=mesh.bounds[1][2] + 10,
                )
            else:
                # Default parallel
                from pycam3d.toolpath import ToolpathGenerator, Strategy
                generator = ToolpathGenerator(mesh, stl_path)
                toolpath = generator.generate_parallel(
                    tool=tool,
                    stepover=request.tool_diameter * request.stepover,
                    strategy=Strategy.ZIGZAG_X,
                )

            toolpath.feed_rate = request.feed_rate

            # Store toolpath for simulation
            mesh_data["toolpath"] = toolpath

            # Generate G-code
            writer = GCodeWriter()
            gcode = writer.generate(toolpath, spindle_rpm=request.spindle_rpm)

            # Calculate stats
            total_length = toolpath.get_total_length()
            estimated_time_min = total_length / request.feed_rate

            return JSONResponse({
                "toolpath": [
                    {"x": p.x, "y": p.y, "z": p.z, "rapid": p.rapid}
                    for p in toolpath.points
                ],
                "gcode": str(gcode),
                "stats": {
                    "points": len(toolpath),
                    "length": total_length,
                    "estimated_time": f"{estimated_time_min:.1f} min",
                    "gcode_lines": len(gcode),
                }
            })

        except Exception as e:
            logger.exception("Generation failed")
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/simulation/{mesh_id}")
    async def get_simulation(mesh_id: str, fps: float = 30.0):
        """Get simulation animation data for a generated toolpath."""
        try:
            if mesh_id not in mesh_store:
                return JSONResponse({"error": "Mesh not found"}, status_code=404)

            mesh_data = mesh_store[mesh_id]

            # Check if we have a toolpath stored
            if "toolpath" not in mesh_data:
                return JSONResponse({"error": "No toolpath generated yet"}, status_code=400)

            from pycam3d.simulation import SimulationEngine

            toolpath = mesh_data["toolpath"]
            mesh = mesh_data["mesh"]

            # Create simulation
            engine = SimulationEngine()
            engine.load_from_toolpath(toolpath, safe_z=mesh.bounds[1][2] + 10)

            # Export animation data
            animation_data = engine.export_animation_data(fps=fps)

            return JSONResponse(animation_data)

        except Exception as e:
            logger.exception("Simulation failed")
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    return app


def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the web server."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("Uvicorn required: pip install uvicorn")

    app = create_app()
    print(f"\n  PyCAM3D Web Interface")
    print(f"  Open: http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
