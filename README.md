# Specomp

Specomp is a python library that provides fast lossless and lossy HSI (hyperspectral image) compression algorithm implementations. Currently, there are no open source packages tailored to hsi compression that have all three of these qualities: easy setup, high compression-ratios and high bit rate. Specomp does.

## Why not traditional image compression algorithms?

Where RGB image compression algorithms like JPEG rely mostly on exploiting spatial redundancy in images, hyperspectral images have enormous redundancy along their spectral dimension. This creates the need for custom-algorithms that exploit this.

Additionally, lossless algorithms need to preserve important features in pixel spectra so that the results of analysis are not affected by compression. The lossless algorithms in this package make guarantees about the scale and type of information lost.

Finally, most RGB image compression algorithms just don't work images with more than 3 or 4 channels without custom implementations, chunking or preprocessing. We(I) want scientists and developers to be able to (un)compress images with a few lines of code.

## Development is ongoing

Phase 1: Python prototyping

Phase 2: Rust optimization.