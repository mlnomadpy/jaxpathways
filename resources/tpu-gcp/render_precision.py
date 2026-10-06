"""Render error and storage from an actual precision_profile.py report."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = json.loads(args.report.read_text())
    modes = ['fp32', 'bf16', 'int8']
    colors = {'balanced': '#245bbd', 'outlier': '#bb541e'}
    rows = {(row['case'], row['mode']): row for row in report['results']}
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (error, storage) = plt.subplots(2, 1, figsize=(8, 9), layout='constrained')
    for case, color in colors.items():
        error.plot(modes, [rows[case, mode]['relative_l2_error'] for mode in modes],
                   marker='o', linewidth=2, color=color, label=case)
    error.set_yscale('log')
    error.set_ylabel('Relative L2 output error (log scale)')
    error.set_xlabel('Operand policy; output remains FP32')
    error.set_title('Outliers change the quantization error')
    error.grid(axis='y', alpha=.2)
    error.legend()
    sizes = [rows['balanced', mode]['stored_operand_bytes'] for mode in modes]
    bars = storage.bar(modes, sizes, color=['#334d70', '#527aa9', '#82a2c8'])
    storage.bar_label(bars, labels=[f'{size:,}' for size in sizes], padding=4, fontsize=11)
    storage.set_ylim(0, max(sizes) * 1.2)
    storage.set_ylabel('Stored operand bytes, including INT8 scales')
    storage.set_xlabel('Does not include outputs or runtime peak memory')
    storage.set_title('Smaller storage is a separate measurement')
    # Keep the title truthful if this script is reused with a different shape.
    dims = report['dimensions']
    fig.suptitle(f"Recorded {report['platform'].upper()} dot: {dims['batch']} × {dims['features']} @ "
                 f"{dims['features']} × {dims['outputs']}\nSynthetic inputs; output remains FP32", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
