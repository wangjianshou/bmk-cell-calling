import argparse
from .cells import save_feature, callcells, barcode_umi_plot, writeQC

def parseArgs():
    parser = argparse.ArgumentParser('')
    parser.add_argument('--matrix', '-m', required=True, help='raw feature barcode matrix')
    parser.add_argument('--expect-cells', '-e', default=3000, required=False,
                        type=int, help='number of expected cells')
    parser.add_argument('--basedir', '-d', default='.', help='output direction')
    return parser.parse_args()


def main():
    args = parseArgs()
    expect_cells = args.expect_cells
    basedir = args.basedir
    matrixDir = args.matrix
    matrix, filtered_matrix = callcells(matrixDir, basedir, expect_cells)
    cell_barcodes = filtered_matrix.bcs[:]
    counts_per_bc = matrix.get_counts_per_bc()
    raw_barcodes = matrix.bcs[:]
    barcode_umi_plot(counts_per_bc, raw_barcodes, cell_barcodes, basedir)
    writeQC(filtered_matrix, basedir)
    print('completed!')

if __name__=='__main__':
    main()
