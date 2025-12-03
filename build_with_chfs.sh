#!/bin/bash
set -e

# Load spack and mochi-margo
source /work/NBB/rmaeda/spack/share/spack/setup-env.sh
spack load mochi-margo

# Add CHFS and BenchFS to PKG_CONFIG_PATH
export PKG_CONFIG_PATH="/work/NBB/rmaeda/.local/lib/pkgconfig:/work/0/NBB/rmaeda/workspace/rust/benchfs/target/release:${PKG_CONFIG_PATH}"

echo "PKG_CONFIG_PATH=$PKG_CONFIG_PATH"
echo ""
echo "Checking packages..."
pkg-config --exists chfs && echo "chfs: found" || echo "chfs: NOT found"
pkg-config --exists benchfs && echo "benchfs: found" || echo "benchfs: NOT found"
echo ""

# Clean and rebuild
make clean || true
autoreconf -i
./configure
make -j4
