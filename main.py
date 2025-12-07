#!/usr/bin/env python3
"""
PyCAM3D Web Server - Replit Entry Point

Starts the FastAPI web server configured for Replit environment.
"""

import uvicorn
from src.pycam3d.web import create_app

if __name__ == "__main__":
    app = create_app()
    
    print("\n" + "="*60)
    print("  PyCAM3D - 3D Toolpath Generation & Visualization")
    print("  Server starting on http://0.0.0.0:5000")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )
