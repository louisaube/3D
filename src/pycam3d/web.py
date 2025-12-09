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
        .btn-small {
            width: auto;
            padding: 8px 12px;
            font-size: 0.85rem;
            margin: 0;
        }
        .btn-group {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .btn-icon {
            width: 36px;
            height: 36px;
            padding: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
        }
        .scale-input-group {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .scale-input-group input[type="number"] {
            width: 80px;
            text-align: center;
        }
        .scale-presets {
            display: flex;
            gap: 6px;
            margin-top: 8px;
            flex-wrap: wrap;
        }
        .scale-presets button {
            flex: 1;
            min-width: 60px;
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

            <div id="view-controls" class="card hidden">
                <h2>View & Transform</h2>

                <div class="form-group">
                    <label>Camera Views</label>
                    <div class="btn-group">
                        <button class="btn-secondary btn-small" onclick="resetView()">Reset</button>
                        <button class="btn-secondary btn-small" onclick="setView('top')">Top</button>
                        <button class="btn-secondary btn-small" onclick="setView('front')">Front</button>
                        <button class="btn-secondary btn-small" onclick="setView('side')">Side</button>
                        <button class="btn-secondary btn-small" onclick="setView('iso')">Iso</button>
                    </div>
                </div>

                <div class="form-group">
                    <label>Scale Factor</label>
                    <div class="scale-input-group">
                        <input type="number" id="scale-factor" value="1.0" min="0.001" max="1000" step="0.1">
                        <button class="btn-secondary btn-small" onclick="applyScale()">Apply</button>
                    </div>
                    <div class="scale-presets">
                        <button class="btn-secondary btn-small" onclick="setScale(0.1)">0.1x</button>
                        <button class="btn-secondary btn-small" onclick="setScale(0.5)">0.5x</button>
                        <button class="btn-secondary btn-small" onclick="setScale(2)">2x</button>
                        <button class="btn-secondary btn-small" onclick="setScale(10)">10x</button>
                        <button class="btn-secondary btn-small" onclick="setScale(25.4)">in→mm</button>
                    </div>
                </div>

                <div class="form-group">
                    <label>Target Size (mm)</label>
                    <div class="scale-input-group">
                        <input type="number" id="target-size" placeholder="e.g. 100" min="0.1" step="1">
                        <select id="target-axis">
                            <option value="max">Max</option>
                            <option value="x">X</option>
                            <option value="y">Y</option>
                            <option value="z">Z</option>
                        </select>
                        <button class="btn-secondary btn-small" onclick="scaleToSize()">Fit</button>
                    </div>
                </div>
            </div>

            <div id="toolpath-section" class="card hidden">
                <h2>Toolpath Settings</h2>

                <button class="btn-success" id="smart-btn" style="margin-bottom: 15px;">
                    Smart Strategy (Auto-Optimize)
                </button>

                <div id="smart-plan" class="hidden" style="margin-bottom: 15px;">
                    <div style="background: rgba(0,255,136,0.1); border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                        <h3 style="margin: 0 0 10px 0; color: #00ff88; font-size: 0.95rem;">Recommended Plan</h3>
                        <div id="smart-plan-content"></div>
                    </div>
                    <div class="btn-group">
                        <button class="btn-primary btn-small" id="apply-smart-btn">Apply & Generate All</button>
                        <button class="btn-secondary btn-small" id="cancel-smart-btn">Manual Mode</button>
                    </div>
                </div>

                <div id="manual-settings">
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

        // Store initial mesh data for reset and scaling
        let meshData = null;
        let meshCenter = new THREE.Vector3();
        let meshSize = new THREE.Vector3();
        let initialCameraPos = new THREE.Vector3();
        let initialControlsTarget = new THREE.Vector3();

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

        let statusTimeout = null;
        function showStatus(msg, duration = 3000) {
            const status = document.getElementById('status');
            status.innerHTML = msg;
            status.classList.add('show');
            if (statusTimeout) clearTimeout(statusTimeout);
            if (duration > 0) {
                statusTimeout = setTimeout(() => status.classList.remove('show'), duration);
            }
        }
        function hideStatus() {
            const status = document.getElementById('status');
            status.classList.remove('show');
            if (statusTimeout) clearTimeout(statusTimeout);
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
            console.log('Starting upload for file:', file.name, 'size:', file.size, 'bytes');
            showStatus('<span class="loading"></span>Uploading mesh...', 0); // Keep visible

            const formData = new FormData();
            formData.append('file', file);

            try {
                console.log('Sending fetch request to /api/upload...');
                const response = await fetch('/api/upload', { method: 'POST', body: formData });
                console.log('Response received, status:', response.status);

                if (!response.ok) {
                    const errorText = await response.text();
                    console.error('Server error:', response.status, errorText);
                    showStatus('Server error: ' + response.status + ' - ' + errorText, 5000);
                    return;
                }

                console.log('Parsing JSON response...');
                const data = await response.json();
                console.log('Upload response:', data);

                if (data.error) {
                    console.error('API error:', data.error);
                    showStatus('Error: ' + data.error, 5000);
                    return;
                }

                currentMeshId = data.mesh_id;
                console.log('Mesh ID:', currentMeshId);
                console.log('Vertices count:', data.vertices ? data.vertices.length : 'undefined');
                console.log('Faces count:', data.faces ? data.faces.length : 'undefined');

                try {
                    console.log('Calling displayMesh...');
                    displayMesh(data);
                    showStatus('Mesh loaded successfully!');
                    document.getElementById('mesh-info').classList.remove('hidden');
                    document.getElementById('view-controls').classList.remove('hidden');
                    document.getElementById('toolpath-section').classList.remove('hidden');
                    console.log('Display complete');
                } catch (displayErr) {
                    console.error('Display error:', displayErr);
                    showStatus('Error displaying mesh: ' + displayErr.message, 5000);
                }

            } catch (err) {
                console.error('Upload error:', err);
                showStatus('Upload failed: ' + err.message, 5000);
            }
        }

        function displayMesh(data) {
            // Validate data
            if (!data.vertices || !data.faces) {
                throw new Error('Invalid mesh data received from server');
            }
            
            console.log('Creating mesh with', data.vertices.length, 'vertices and', data.faces.length, 'faces');
            
            // If WebGL is not available, just update stats and return
            if (!renderer || !scene) {
                console.warn('WebGL not available - displaying stats only');
                const bounds_min = data.stats.bounds_min;
                const bounds_max = data.stats.bounds_max;
                const sizeX = bounds_max[0] - bounds_min[0];
                const sizeY = bounds_max[1] - bounds_min[1];
                const sizeZ = bounds_max[2] - bounds_min[2];
                
                document.getElementById('stat-vertices').textContent = data.stats.vertex_count.toLocaleString();
                document.getElementById('stat-faces').textContent = data.stats.face_count.toLocaleString();
                document.getElementById('stat-size').textContent = `${sizeX.toFixed(1)}×${sizeY.toFixed(1)}×${sizeZ.toFixed(1)}`;
                document.getElementById('stat-watertight').textContent = data.stats.is_watertight ? '✓' : '✗';
                return;
            }
            
            // Remove existing mesh
            if (meshObject) scene.remove(meshObject);

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

            // Center camera on mesh
            geometry.computeBoundingBox();
            const box = geometry.boundingBox;
            const center = new THREE.Vector3();
            box.getCenter(center);
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);

            console.log('Mesh bounds:', box.min, box.max);
            console.log('Mesh center:', center);
            console.log('Mesh size:', size, 'maxDim:', maxDim);

            // Store mesh data for reset and scaling
            meshData = data;
            meshCenter.copy(center);
            meshSize.copy(size);

            // Adjust camera clipping planes based on mesh size
            const distance = maxDim * 2;
            camera.near = maxDim * 0.001;  // 0.1% of mesh size
            camera.far = maxDim * 100;     // 100x mesh size
            camera.updateProjectionMatrix();

            // Position camera to see the whole mesh
            camera.position.set(center.x + distance, center.y + distance, center.z + distance);
            camera.lookAt(center);

            // Update orbit controls target and sync
            controls.target.copy(center);
            controls.update();

            // Store initial camera position for reset
            initialCameraPos.copy(camera.position);
            initialControlsTarget.copy(controls.target);

            // Update stats
            document.getElementById('stat-vertices').textContent = data.stats.vertex_count.toLocaleString();
            document.getElementById('stat-faces').textContent = data.stats.face_count.toLocaleString();
            document.getElementById('stat-size').textContent = `${size.x.toFixed(1)}x${size.y.toFixed(1)}x${size.z.toFixed(1)}`;
            document.getElementById('stat-watertight').textContent = data.stats.is_watertight ? '✓' : '✗';
        }

        // View control functions
        function resetView() {
            if (!meshObject) return;
            camera.position.copy(initialCameraPos);
            controls.target.copy(initialControlsTarget);
            controls.update();
            console.log('View reset to initial position');
        }

        function setView(type) {
            if (!meshObject) return;
            const maxDim = Math.max(meshSize.x, meshSize.y, meshSize.z);
            const distance = maxDim * 2.5;

            switch(type) {
                case 'top':
                    camera.position.set(meshCenter.x, meshCenter.y + distance, meshCenter.z);
                    break;
                case 'front':
                    camera.position.set(meshCenter.x, meshCenter.y, meshCenter.z + distance);
                    break;
                case 'side':
                    camera.position.set(meshCenter.x + distance, meshCenter.y, meshCenter.z);
                    break;
                case 'iso':
                default:
                    camera.position.set(
                        meshCenter.x + distance * 0.7,
                        meshCenter.y + distance * 0.7,
                        meshCenter.z + distance * 0.7
                    );
                    break;
            }
            camera.lookAt(meshCenter);
            controls.target.copy(meshCenter);
            controls.update();
            console.log('View set to:', type);
        }

        // Scale functions
        function setScale(factor) {
            document.getElementById('scale-factor').value = factor;
        }

        async function applyScale() {
            const factor = parseFloat(document.getElementById('scale-factor').value);
            if (!factor || factor <= 0 || !currentMeshId) {
                showStatus('Invalid scale factor', 3000);
                return;
            }
            await scaleMesh(factor);
        }

        async function scaleToSize() {
            const targetSize = parseFloat(document.getElementById('target-size').value);
            const axis = document.getElementById('target-axis').value;
            if (!targetSize || targetSize <= 0 || !currentMeshId) {
                showStatus('Enter a valid target size', 3000);
                return;
            }

            let currentSize;
            switch(axis) {
                case 'x': currentSize = meshSize.x; break;
                case 'y': currentSize = meshSize.y; break;
                case 'z': currentSize = meshSize.z; break;
                default: currentSize = Math.max(meshSize.x, meshSize.y, meshSize.z);
            }

            const factor = targetSize / currentSize;
            document.getElementById('scale-factor').value = factor.toFixed(4);
            await scaleMesh(factor);
        }

        async function scaleMesh(factor) {
            showStatus('<span class="loading"></span>Scaling mesh...', 0);
            try {
                const response = await fetch('/api/scale', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mesh_id: currentMeshId, factor: factor })
                });
                const data = await response.json();
                if (data.error) {
                    showStatus('Error: ' + data.error, 5000);
                    return;
                }
                // Update mesh display with new data
                displayMesh(data);
                showStatus(`Mesh scaled by ${factor.toFixed(3)}x`, 3000);
                // Update stats
                document.getElementById('stat-size').textContent =
                    `${meshSize.x.toFixed(1)}x${meshSize.y.toFixed(1)}x${meshSize.z.toFixed(1)}`;
            } catch (err) {
                console.error('Scale error:', err);
                showStatus('Scale failed: ' + err.message, 5000);
            }
        }

        // Smart Strategy functions
        let smartPlanData = null;

        document.getElementById('smart-btn').addEventListener('click', async () => {
            if (!currentMeshId) return;

            showStatus('<span class="loading"></span>Analyzing mesh geometry...', 0);
            document.getElementById('smart-btn').disabled = true;

            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mesh_id: currentMeshId })
                });
                const data = await response.json();

                if (data.error) {
                    showStatus('Analysis error: ' + data.error, 5000);
                    return;
                }

                smartPlanData = data;
                displaySmartPlan(data);
                showStatus('Analysis complete! Review the recommended plan.', 3000);

            } catch (err) {
                console.error('Analysis error:', err);
                showStatus('Analysis failed: ' + err.message, 5000);
            } finally {
                document.getElementById('smart-btn').disabled = false;
            }
        });

        function displaySmartPlan(plan) {
            const container = document.getElementById('smart-plan-content');

            // Tool pool header with selection info
            let html = '<div style="background: rgba(0,212,255,0.1); border-radius: 6px; padding: 10px; margin-bottom: 12px;">';
            html += `<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">`;
            html += `<span style="font-weight: 600; color: #00d4ff; font-size: 0.9rem;">🔧 Pool de Fraises</span>`;
            html += `<span style="background: ${plan.total_tools <= plan.max_tools ? '#00ff88' : '#ff6b6b'}; color: #000; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 700;">${plan.total_tools}/${plan.max_tools} MAX</span>`;
            html += `</div>`;

            // Show available tools in compact format
            html += `<div style="font-size: 0.7rem; color: #888;">`;
            const toolsByType = {};
            (plan.available_tools || []).forEach(t => {
                if (!toolsByType[t.type]) toolsByType[t.type] = [];
                toolsByType[t.type].push(t.diameter);
            });
            const typeLabels = {flat: 'Plates', ball: 'Boules', bull: 'Toriques'};
            Object.keys(toolsByType).forEach(type => {
                html += `<div>${typeLabels[type] || type}: ${toolsByType[type].sort((a,b)=>a-b).join(', ')}mm</div>`;
            });
            html += `</div></div>`;

            // Region analysis summary
            html += '<div style="font-size: 0.8rem; color: #888; margin-bottom: 10px;">';
            html += `<div>Surface: ${plan.region_analysis.flat_percent?.toFixed(0) || 0}% flat, `;
            html += `${(plan.region_analysis.gentle_curve_percent + plan.region_analysis.moderate_curve_percent)?.toFixed(0) || 0}% curved, `;
            html += `${plan.region_analysis.sharp_feature_percent?.toFixed(0) || 0}% detail</div>`;
            html += '</div>';

            // Selected operations header
            html += '<div style="font-size: 0.75rem; color: #00ff88; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 1px;">Fraises sélectionnées</div>';

            // Operations
            html += '<div style="display: flex; flex-direction: column; gap: 8px;">';

            const phaseColors = {
                'roughing': '#ff6b6b',
                'semi_finish': '#ffd93d',
                'finish': '#6bcf6b',
                'detail': '#6b9fff'
            };

            const phaseLabels = {
                'roughing': 'ROUGHING',
                'semi_finish': 'SEMI-FINISH',
                'finish': 'FINISH',
                'detail': 'DETAIL'
            };

            plan.operations.forEach((op, idx) => {
                const color = phaseColors[op.phase] || '#888';
                html += `<div style="background: rgba(255,255,255,0.05); padding: 8px; border-radius: 6px; border-left: 3px solid ${color};">`;
                html += `<div style="display: flex; justify-content: space-between; align-items: center;">`;
                html += `<span style="font-weight: 600; color: ${color}; font-size: 0.75rem;">${phaseLabels[op.phase]}</span>`;
                html += `<span style="font-size: 0.7rem; color: #666;">${op.estimated_time_percent}% time</span>`;
                html += `</div>`;
                html += `<div style="font-size: 0.85rem; margin-top: 4px;">`;
                html += `<strong>${op.tool.name}</strong> - ${op.strategy} @ ${op.stepover_percent}%`;
                html += `</div>`;
                html += `</div>`;
            });

            html += '</div>';

            // Summary
            html += `<div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1); font-size: 0.8rem;">`;
            html += `<div style="color: #00ff88;">Est. time reduction: ${plan.estimated_time_reduction}</div>`;
            html += `<div style="color: #00d4ff;">Quality improvement: ${plan.quality_improvement}</div>`;
            html += `</div>`;

            container.innerHTML = html;

            // Show plan, hide manual settings
            document.getElementById('smart-plan').classList.remove('hidden');
            document.getElementById('manual-settings').style.opacity = '0.5';
            document.getElementById('manual-settings').style.pointerEvents = 'none';
        }

        document.getElementById('cancel-smart-btn').addEventListener('click', () => {
            document.getElementById('smart-plan').classList.add('hidden');
            document.getElementById('manual-settings').style.opacity = '1';
            document.getElementById('manual-settings').style.pointerEvents = 'auto';
            smartPlanData = null;
        });

        document.getElementById('apply-smart-btn').addEventListener('click', async () => {
            if (!smartPlanData || !smartPlanData.operations.length) {
                showStatus('No plan to apply', 3000);
                return;
            }

            showStatus('<span class="loading"></span>Generating multi-tool toolpath...', 0);
            document.getElementById('apply-smart-btn').disabled = true;

            // For now, apply the first finishing operation
            // TODO: Generate all operations and combine G-code
            const finishOp = smartPlanData.operations.find(op =>
                op.phase === 'finish' || op.phase === 'semi_finish'
            ) || smartPlanData.operations[0];

            // Set form values from smart plan
            document.getElementById('strategy').value = finishOp.strategy;
            document.getElementById('tool-type').value = finishOp.tool.type;
            document.getElementById('tool-diameter').value = finishOp.tool.diameter;
            document.getElementById('stepover').value = finishOp.stepover_percent;

            // Update display values
            document.getElementById('tool-dia-val').textContent = finishOp.tool.diameter;
            document.getElementById('stepover-val').textContent = finishOp.stepover_percent;

            // Trigger generate
            document.getElementById('generate-btn').click();
            document.getElementById('apply-smart-btn').disabled = false;
        });

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

    # Max faces for visualization (prevents browser lag)
    MAX_DISPLAY_FACES = 50000

    def decimate_for_display(mesh, max_faces=MAX_DISPLAY_FACES):
        """Decimate mesh for fast browser display."""
        if len(mesh.faces) <= max_faces:
            return mesh
        # Use trimesh's simplify_quadric_decimation if available
        try:
            ratio = max_faces / len(mesh.faces)
            simplified = mesh.simplify_quadric_decimation(int(len(mesh.faces) * ratio))
            logger.info(f"Decimated mesh: {len(mesh.faces)} -> {len(simplified.faces)} faces")
            return simplified
        except Exception:
            # Fallback: random face sampling
            indices = np.random.choice(len(mesh.faces), max_faces, replace=False)
            return mesh.submesh([indices], append=True)

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

            # Generate ID and store FULL mesh for analysis
            mesh_id = str(uuid.uuid4())[:8]
            mesh_store[mesh_id] = {
                "mesh": mesh,
                "path": tmp_path,
            }

            # Get stats from FULL mesh
            stats = {
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.faces),
                "is_watertight": bool(mesh.is_watertight),
                "bounds_min": mesh.bounds[0].tolist(),
                "bounds_max": mesh.bounds[1].tolist(),
            }

            # Decimate for DISPLAY only
            display_mesh = decimate_for_display(mesh)

            return JSONResponse({
                "mesh_id": mesh_id,
                "filename": file.filename,
                "stats": stats,
                "vertices": display_mesh.vertices.tolist(),
                "faces": display_mesh.faces.tolist(),
                "decimated": len(display_mesh.faces) < len(mesh.faces),
            })

        except Exception as e:
            logger.exception("Upload failed")
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.post("/api/scale")
    async def scale_mesh(request: dict):
        """Scale a mesh by a given factor."""
        try:
            mesh_id = request.get("mesh_id")
            factor = request.get("factor", 1.0)

            if not mesh_id or mesh_id not in mesh_store:
                return JSONResponse({"error": "Mesh not found"}, status_code=404)

            if not factor or factor <= 0:
                return JSONResponse({"error": "Invalid scale factor"}, status_code=400)

            mesh_data = mesh_store[mesh_id]
            mesh = mesh_data["mesh"]

            # Scale the mesh vertices
            mesh.vertices *= factor

            # Update stored mesh
            mesh_store[mesh_id]["mesh"] = mesh

            # Get updated stats
            stats = {
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.faces),
                "is_watertight": bool(mesh.is_watertight),
                "bounds_min": mesh.bounds[0].tolist(),
                "bounds_max": mesh.bounds[1].tolist(),
            }

            logger.info(f"Mesh {mesh_id} scaled by factor {factor}")

            # Decimate for display
            display_mesh = decimate_for_display(mesh)

            return JSONResponse({
                "mesh_id": mesh_id,
                "stats": stats,
                "vertices": display_mesh.vertices.tolist(),
                "faces": display_mesh.faces.tolist(),
            })

        except Exception as e:
            logger.exception("Scale failed")
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.post("/api/analyze")
    async def analyze_mesh_smart(request: dict):
        """Analyze mesh and generate smart multi-tool machining plan."""
        try:
            mesh_id = request.get("mesh_id")

            if not mesh_id or mesh_id not in mesh_store:
                return JSONResponse({"error": "Mesh not found"}, status_code=404)

            mesh_data = mesh_store[mesh_id]
            mesh = mesh_data["mesh"]

            # Import and run smart strategy analysis
            from pycam3d.smart_strategy import analyze_mesh_for_smart_strategy

            logger.info(f"Running smart strategy analysis for mesh {mesh_id}")
            plan = analyze_mesh_for_smart_strategy(mesh, mesh_id)

            return JSONResponse(plan.to_dict())

        except Exception as e:
            logger.exception("Analysis failed")
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
