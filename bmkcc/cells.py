import os
import numpy as np
import argparse
from plotly import graph_objects as go
from bmkcc.cellranger.matrix import CountMatrix
from bmkcc.cellranger.cell_calling import find_nonambient_barcodes
from bmkcc.cellranger.io import open_maybe_gzip
from bmkcc.cellranger.webshim.data import compute_sort_order,compute_plot_segments
from bmkcc.cellranger.webshim.common import build_plot_data_dict

def parseArgs():
    parser = argparse.ArgumentParser('')
    parser.add_argument('--matrix', '-m', required=True, help='raw feature barcode matrix')
    parser.add_argument('--expect-cells', '-e', default=3000, required=False,
                        type=int, help='number of expected cells')
    parser.add_argument('--basedir', '-d', default='.', help='output direction')
    return parser.parse_args()

def save_feature(feature_ref, base_dir, compress):
    out_features = os.path.join(base_dir, "features.tsv.gz")
    with open_maybe_gzip(out_features, 'w') as f:
        tmp = ['\t'.join([i.id, i.name, i.feature_type]) for i in feature_ref.feature_defs]
        f.write('\n'.join(tmp))


def callcells(matrixDir, basedir, expect_cells):
    matrix = CountMatrix.from_v3_mtx(matrixDir) 
    matrix.m = matrix.m.astype('int64')
    matrix.tocsc()
    umis_per_bc = matrix.get_counts_per_bc()
    bc_order = np.argsort(umis_per_bc)

    thres = np.quantile(umis_per_bc[bc_order][-expect_cells:], 0.99)
    orig_cell_idx = np.flatnonzero(umis_per_bc > thres/10)
    orig_cell_bcs = matrix.bcs[orig_cell_idx]

    r = find_nonambient_barcodes(matrix, orig_cell_bcs) 
    if r is None:
        cells = orig_cell_idx
    else:
        cells = np.concatenate([r.eval_bcs[r.is_nonambient], orig_cell_idx])
    print('number of cells: {}'.format(cells.shape[0]))
    filtered_matrix = matrix.select_barcodes(cells)

    filtered_matrix.save_mex(os.path.join(basedir, 'filtered_feature_bc_matrix'),
                             save_feature, compress=True)
    return matrix, filtered_matrix

def barcode_umi_plot(counts_per_bc, raw_barcodes, cell_barcodes, basedir='.'):
    srt_order = compute_sort_order(counts_per_bc, raw_barcodes, cell_barcodes)
    sorted_bc = raw_barcodes[srt_order]
    sorted_counts = counts_per_bc[srt_order]
    plot_segments = compute_plot_segments(sorted_bc, sorted_counts, cell_barcodes)
    plot_data = [build_plot_data_dict(i, sorted_counts) for i in plot_segments]
    layout = go.Layout(xaxis={'type':'log', 'title':'Barcodes'}, 
                       yaxis={'type':'log', 'title':'UMI counts'},
                       width=800, height=600)
    fig =  go.Figure(plot_data, layout=layout)
    fig.write_html(os.path.join(basedir, 'rank_barcode_umi.html'),
                   config= {'displaylogo': False})
    fig.write_image(os.path.join(basedir, 'rank_barcode_umi.png'),
                    engine='kaleido', scale=5)
    fig.write_image(os.path.join(basedir, 'rank_barcode_umi.pdf'),
                    engine='kaleido')
    return fig

def writeQC(matrix, basedir):
    Estimated_Number_of_Cells = matrix.get_shape()[1]
    Median_Genes_per_Cell = int(np.median(matrix.get_numfeatures_per_bc()))
    Total_Genes_Detected = (matrix.get_counts_per_feature()>0).sum()
    Median_UMI_Counts_per_Cell = int(np.median(matrix.get_counts_per_bc()))
    r = (f'Estimated_Number_of_Cells\t{Estimated_Number_of_Cells}\n'
         f'Median_Genes_per_Cell\t{Median_Genes_per_Cell}\n'
         f'Total_Genes_Detected\t{Total_Genes_Detected}\n'
         f'Median_UMI_Counts_per_Cell\t{Median_UMI_Counts_per_Cell}\n')
    with open(os.path.join(basedir, 'filter_qc.txt'), 'w') as f:
        f.write(r)

def cellQC(matrix):
    r = {}
    r['NumberofCells'] = matrix.get_shape()[1]
    r['MedianGenes'] = int(np.median(matrix.get_numfeatures_per_bc()))
    r['TotalGenesDetected'] = (matrix.get_counts_per_feature()>0).sum()
    r['MedianUMICountsperCell'] = int(np.median(matrix.get_counts_per_bc()))
    return r




if __name__=='__main__':
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
    print('completed!\n')
