#!/usr/bin/env python3
"""
NeuroMontage - Brain Overlay Mosaic Generator

Create resolution-based brain mosaics and GIFs with optional segmentation overlays.
Supports 3D/4D NIfTI files with automatic layout optimization and adaptive sampling.

Features:
- No upscaling (preserves native resolution)
- Adaptive slice sampling for large datasets
- Maximum output dimensions: 4000×4000 pixels
- Automatic quality-driven resolution selection

:Authors:
    MDS
    MGH / Harvard Medical School

:Version: 2.2
:Date: 2025-10-08
:License: MIT

:Usage Examples:

  # Brain-only mosaic with automatic resolution
  python3 neuromontage.py -b brain_T1.nii.gz -o brain_mosaic.jpg --start_slice 10 --end_slice 50

  # Force 4K output (may result in smaller slices)
  python3 neuromontage.py -b brain_T1.nii.gz -o brain_mosaic_4k.jpg --resolution 4k

  # 4D volume super mosaic with adaptive sampling
  python3 neuromontage.py -b diffusion_4d.nii.gz -o diffusion_super_mosaic.jpg

  # Single segmentation with overlay
  python3 neuromontage.py -b brain_T1.nii.gz -s cumulative_segmentation.nii.gz -o mosaic_overlay.jpg --alpha 0.7

  # Alternating rows (brain-only in odd rows, brain+overlay in even rows)
  python3 neuromontage.py -b brain_T1.nii.gz -s segmentation.nii.gz -o mosaic_alternate.jpg --alternate

:Requires:
    numpy, nibabel, matplotlib, Pillow, scikit-image
"""

__version__ = "2.2"
__date__ = "2025-10-08"
__author__ = "MDS"
__organization__ = "MGH/HMS"
__description__ = "NeuroMontage - Brain Overlay Mosaic Generator"

import argparse
import nibabel as nib
from mosaic_3d import create_mosaic
from mosaic_4d import create_super_mosaic_4d
from gif_builder import create_gif


def parse_args():
    parser = argparse.ArgumentParser(
        description='NeuroMontage: resolution-based mosaic/GIF of brain slices with optional segmentation overlays.',
        epilog='For detailed documentation, see README.md'
    )
    parser.add_argument(
        '-b', '--brain_file', type=str, required=True,
        help='Structural brain NIfTI (3D or 4D).'
    )
    parser.add_argument(
        '-s', '--input_files', type=str, nargs='+', required=False, default=None,
        help=(
            'Optional: One or more lesion/segmentation NIfTI files (only for 3D brain files). '
            '• If not provided, creates a brain-only visualization. '
            '• If exactly 1 file is provided and --outline is not set, it is used as a colormap overlay. '
            '• Otherwise, each input is treated as a lesion mask with colored contour outlines.'
        )
    )
    parser.add_argument(
        '--outline', action='store_true',
        help='Force outline mode even if only one input file is provided.'
    )
    parser.add_argument(
        '--highlight', action='store_true',
        help='Draw a red border around any slice that contains segmentation.'
    )
    parser.add_argument(
        '--alternate', action='store_true',
        help='Alternate rows: odd rows show brain-only, even rows show brain+segmentation overlay. Only works with segmentation files.'
    )
    parser.add_argument(
        '-o', '--output_file', type=str, required=True,
        help='Output file path: .jpg for a single mosaic, .gif for an animated GIF.'
    )
    parser.add_argument(
        '--resolution', type=str, default='auto',
        choices=['auto', 'hd', '2k', '4k'],
        help=(
            'Output resolution: '
            'auto (quality-driven, recommended - may adaptively sample slices), '
            'HD (1920×1080), '
            '2K (2560×1440), '
            '4K (3840×2160). '
            'Default: auto. '
            'Note: Preset resolutions never upscale and may downsample significantly for large datasets.'
        )
    )
    parser.add_argument(
        '--gif', action='store_true',
        help='Produce an animated GIF (iterate through all valid slices). Only for 3D data.'
    )
    parser.add_argument(
        '--duration', type=float, default=10.0,
        help='GIF duration in seconds (only used if --gif is set). Default=10.0'
    )
    parser.add_argument(
        '--slice_step', type=int, default=1,
        help=(
            'Step between slices in mosaic (default=1, use every slice). '
            'Note: In auto resolution mode, additional adaptive sampling may be applied. '
            'Only for 3D data.'
        )
    )
    parser.add_argument(
        '--start_slice', type=int, default=None,
        help='First slice index (inclusive). Default=0. Only for 3D data.'
    )
    parser.add_argument(
        '--end_slice', type=int, default=None,
        help='Last slice index (inclusive). Default=last slice. Only for 3D data.'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.7,
        help='Overlay transparency [0..1] (only in single-input overlay mode). Default=0.7'
    )
    parser.add_argument(
        '--colormap', type=str, default='viridis',
        help='Matplotlib colormap for the segmentation overlay (only in single-input overlay mode). Default=viridis'
    )
    parser.add_argument(
        '--log_scale', action='store_true',
        help='Use logarithmic normalization on the segmentation overlay (only if exactly 1 input file without --outline).'
    )
    parser.add_argument(
        '--threshold', type=float, nargs='+', default=[0.0],
        help=(
            'One or more thresholds. '
            '• If a single value is provided, it is applied to all segmentations. '
            '• If N values are provided for N input_files, each threshold applies to the corresponding file. '
            '• In overlay mode, threshold determines voxels > threshold to show. '
            '• In outline mode, each mask is binarized at its threshold to find contours.'
        )
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load brain data to check dimensionality
    brain_nii = nib.load(args.brain_file)
    brain_data_full = brain_nii.get_fdata()

    # Route to appropriate handler
    if args.gif:
        if len(brain_data_full.shape) == 4:
            raise ValueError("[NeuroMontage] GIF creation not supported for 4D data. Use mosaic mode instead.")
        
        print("[NeuroMontage] Creating animated GIF...")
        create_gif(
            brain_file=args.brain_file,
            input_files=args.input_files,
            output_file=args.output_file,
            gif_duration=args.duration,
            alpha=args.alpha,
            colormap_name=args.colormap,
            log_scale=args.log_scale,
            thresholds=args.threshold,
            outline=args.outline,
            highlight=args.highlight
        )
    else:
        if len(brain_data_full.shape) == 4:
            if args.input_files is not None and len(args.input_files) > 0:
                print("[NeuroMontage] Warning: Segmentation overlays not supported for 4D data. Creating brain-only super mosaic.")
            
            print("[NeuroMontage] Creating 4D super mosaic...")
            print("[NeuroMontage] Note: In auto mode, adaptive sampling may be applied to maintain quality.")
            create_super_mosaic_4d(
                brain_file=args.brain_file,
                output_file=args.output_file,
                resolution=args.resolution
            )
        else:
            print("[NeuroMontage] Creating 3D mosaic...")
            if args.resolution == 'auto':
                print("[NeuroMontage] Note: Auto resolution mode may adaptively sample slices for large datasets.")
            create_mosaic(
                brain_file=args.brain_file,
                input_files=args.input_files,
                output_file=args.output_file,
                resolution=args.resolution,
                slice_step=args.slice_step,
                thresholds=args.threshold,
                alpha=args.alpha,
                colormap_name=args.colormap,
                log_scale=args.log_scale,
                start_slice=args.start_slice,
                end_slice=args.end_slice,
                outline=args.outline,
                highlight=args.highlight,
                alternate=args.alternate
            )


if __name__ == '__main__':
    main()