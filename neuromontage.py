#!/usr/bin/env python3
"""
NeuroMontage - Brain Overlay Mosaic Generator

Create resolution-based brain mosaics and animations (GIF/WebP) with optional segmentation overlays.
Supports 3D/4D NIfTI files with automatic layout optimization and adaptive sampling.

Features:
- No upscaling, except slices smaller than 100px (whole-number factor, no interpolation)
- Adaptive slice sampling for large datasets
- Maximum output dimensions: 3840×2160 pixels (3D), 3840 pixels wide (4D)
- Automatic quality-driven resolution selection

:Authors:
    MDS
    MGH / Harvard Medical School

:License: MIT

:Usage Examples:

  # Brain-only mosaic with automatic resolution
  python3 neuromontage.py -b brain_T1.nii.gz -o brain_mosaic.jpg --start_slice 10 --end_slice 50

  # 4K output (slices are enlarged or reduced to fill the frame)
  python3 neuromontage.py -b brain_T1.nii.gz -o brain_mosaic_4k.jpg --resolution 4k

  # 4D volume super mosaic with adaptive sampling
  python3 neuromontage.py -b diffusion_4d.nii.gz -o diffusion_super_mosaic.jpg

  # 4D volume with segmentation overlay
  python3 neuromontage.py -b diffusion_4d.nii.gz -s lesion_mask.nii.gz -o diffusion_overlay.jpg --alpha 0.7

  # Single segmentation with overlay
  python3 neuromontage.py -b brain_T1.nii.gz -s cumulative_segmentation.nii.gz -o mosaic_overlay.jpg --alpha 0.7

  # Alternating rows (brain-only in odd rows, brain+overlay in even rows)
  python3 neuromontage.py -b brain_T1.nii.gz -s segmentation.nii.gz -o mosaic_alternate.jpg --alternate

:Requires:
    numpy, nibabel, matplotlib, Pillow, scikit-image
"""

__version__ = "2.3.0.17"
__date__ = "2026-11-09"
__author__ = "MDS"
__organization__ = "MGH/HMS"
__description__ = "NeuroMontage - Brain Overlay Mosaic Generator"

import argparse
import os
import sys
import nibabel as nib
from mosaic_3d import create_mosaic
from mosaic_4d import create_super_mosaic_4d
from gif_builder import create_gif
from image_utils import load_nifti_robust


def parse_args():
    parser = argparse.ArgumentParser(
        description='NeuroMontage: resolution-based mosaic/animation of brain slices with optional segmentation overlays.',
        epilog='For detailed documentation, see README.md'
    )
    parser.add_argument(
        '--version', action='version', version=f'NeuroMontage {__version__} ({__date__})'
    )
    parser.add_argument(
        '-b', '--brain_file', type=str, required=True,
        help='Structural brain NIfTI (3D or 4D).'
    )
    parser.add_argument(
        '-s', '--input_files', type=str, nargs='+', required=False, default=None,
        help=(
            'Optional: One or more lesion/segmentation NIfTI files. '
            '• If not provided, creates a brain-only visualization. '
            '• If exactly 1 file is provided and --outline is not set, it is used as a colormap overlay. '
            '• Otherwise, each input is treated as a lesion mask with colored contour outlines. '
            '• For 4D brain data: 3D segmentation applies to all volumes; 4D segmentation provides per-volume masks.'
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
        '--colors', type=str, nargs='+', default=None,
        help=('Outline mode: one color per mask, as palette name (orange, skyblue, green, yellow, '
              'blue, vermillion, pink, black) or #RRGGBB. Default: palette order.')
    )
    parser.add_argument(
        '--line-styles', type=str, nargs='+', default=None,
        help=('Outline mode: one line style per mask (solid, dashed, dotted, long-dash, short-dash, '
              'very-long-dash, medium-dash, dense-dash). Default: in that order.')
    )
    parser.add_argument(
        '--alternate', action='store_true',
        help='Alternate rows: odd rows show brain-only, even rows show brain+segmentation overlay. Only works with segmentation files.'
    )
    parser.add_argument(
        '-o', '--output_file', type=str, required=True,
        help=('Output file path: .jpg, .jpeg, .png or .webp for a mosaic; .gif or .webp (with --animate) '
              'for an animation. .webp gives the smallest files (lossless when outlines are drawn).')
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
            'Note: Presets fill the frame. They enlarge slices without interpolation and may downsample significantly for large datasets. The output is padded to exactly the preset size.'
        )
    )
    parser.add_argument(
        '--animate', '--gif', dest='animate', action='store_true',
        help=('Produce an animation through all slices (.gif or .webp output). Only for 3D data. '
              'Optional for .gif: a .gif output file always gives an animation. --gif is an alias.')
    )
    parser.add_argument(
        '--duration', type=float, default=10.0,
        help='Animation duration in seconds (only used for animations). Default=10.0'
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
        '--rows', type=int, default=None,
        help=(
            'Fixed number of rows in the mosaic grid (columns are derived automatically). '
            'In --alternate mode the value must be even; odd values are rounded down. '
            'Only for 3D mosaics. Default: automatic layout.'
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
        '--colormap', type=str, default=None,
        help=('Matplotlib colormap for the segmentation overlay (only in single-input overlay mode). '
              'Default: Okabe-Ito colors for label maps (up to 7 whole-number values), viridis otherwise. '
              'If set, it is also used for label maps.')
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
    print(f"[NeuroMontage] v{__version__} ({__date__})")

    # The output file extension decides the output type
    ext = os.path.splitext(args.output_file)[1].lower()
    if ext == '.gif':
        args.animate = True
    elif args.animate and ext != '.webp':
        print(f"[NeuroMontage] Error: --animate needs an output file ending in .gif or .webp (got '{ext or 'no extension'}').",
              file=sys.stderr)
        return 1
    elif ext not in ('.jpg', '.jpeg', '.png', '.webp'):
        print(f"[NeuroMontage] Error: output file must end in .jpg, .jpeg, .png, .webp or .gif (got '{ext or 'no extension'}').",
              file=sys.stderr)
        return 1

    # Load brain data to check dimensionality (with robust error handling)
    try:
        brain_nii, brain_data_full = load_nifti_robust(args.brain_file)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[NeuroMontage] Error loading {args.brain_file}: {e}", file=sys.stderr)
        return 1

    # Route to appropriate handler
    if args.rows is not None and (args.animate or len(brain_data_full.shape) == 4):
        print("[NeuroMontage] Warning: --rows only applies to 3D mosaics. Ignoring --rows.")

    if args.animate:
        if len(brain_data_full.shape) == 4:
            print("[NeuroMontage] Error: animations are not supported for 4D data. "
                  "Use a .jpg, .png or .webp output file instead.", file=sys.stderr)
            return 1
        
        print("[NeuroMontage] Creating animation...")
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
            highlight=args.highlight,
            outline_colors=args.colors,
            line_styles=args.line_styles
        )
    else:
        if len(brain_data_full.shape) == 4:
            print("[NeuroMontage] Creating 4D super mosaic...")
            if args.resolution == 'auto':
                print("[NeuroMontage] Note: In auto mode, adaptive sampling may be applied to maintain quality.")
            
            create_super_mosaic_4d(
                brain_file=args.brain_file,
                output_file=args.output_file,
                input_files=args.input_files,
                resolution=args.resolution,
                thresholds=args.threshold,
                alpha=args.alpha,
                colormap_name=args.colormap,
                log_scale=args.log_scale,
                outline=args.outline,
                highlight=args.highlight,
                alternate=args.alternate,
                outline_colors=args.colors,
                line_styles=args.line_styles
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
                alternate=args.alternate,
                rows=args.rows,
                outline_colors=args.colors,
                line_styles=args.line_styles
            )


if __name__ == '__main__':
    sys.exit(main())