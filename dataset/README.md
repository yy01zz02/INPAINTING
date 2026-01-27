# Benchmark Datasets

This directory contains sample data for the inpainting benchmarks used in the project.

## Directory Structure

*   **/brushbench**: Samples from BrushBench.
    *   `masked_images/`: Input images with missing regions.
    *   `masks/`: Binary masks indicating the inpainting area.
    *   `originals/`: Ground truth original images.
    *   `prompts/`: Text prompts for the inpainting task.
*   **/editbench**: Samples from EditBench.
    *   Contains samples with various attribute edits (e.g., color, object, style).
*   **/fluxbench**: Samples from FluxBench.

## Source

To obtain the full datasets, please refer to the following official repositories:

*   **BrushBench**: [GitHub](https://github.com/TencentARC/BrushNet)
*   **EditBench**: [Official Website](https://imagen.research.google/editor/)
*   **FluxBench**: [Official Website](https://bfl.ai/blog/24-11-21-tools)
