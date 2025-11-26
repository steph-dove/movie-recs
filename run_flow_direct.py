#!/usr/bin/env python3
"""
Direct runner for MovieRAGFlow that ensures output is visible.
This uses Metaflow's CLI but ensures output is visible.
"""
import sys
import os

if __name__ == "__main__":
    # Ensure unbuffered output
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # Import and run via Metaflow's CLI
    from flow import MovieRAGFlow
    MovieRAGFlow()

