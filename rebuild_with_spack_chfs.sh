#!/bin/bash
# Rebuild IOR with spack CHFS (without manual build libraries)

set -euo pipefail

cd /work/0/NBB/rmaeda/workspace/rust/benchfs/ior_integration/ior

# Load spack CHFS
source /work/NBB/rmaeda/spack/share/spack/setup-env.sh
spack load chfs

# Get CHFS location
CHFS_DIR=$(spack location -i chfs)

# Set PKG_CONFIG_PATH to use spack CHFS first (NOT /work/NBB/rmaeda/.local)
export PKG_CONFIG_PATH="$CHFS_DIR/lib/pkgconfig:/home/NBB/rmaeda/.local/lib/pkgconfig:$PKG_CONFIG_PATH"

echo "=== PKG_CONFIG_PATH ==="
echo "$PKG_CONFIG_PATH" | tr ':' '\n' | head -10
echo ""

echo "=== CHFS from pkg-config ==="
pkg-config --cflags --libs chfs
echo ""

# Clean previous build
rm -f src/ior Makefile config.status config.log 2>/dev/null || true

# Configure
echo "=== Running configure ==="
./configure

# Build
echo "=== Building ==="
make -j4

# Verify linking
echo ""
echo "=== Verifying IOR libraries ==="
ldd src/ior | grep -E "(chfs|margo|mercury|fabric)"

echo ""
echo "=== Build complete ==="
