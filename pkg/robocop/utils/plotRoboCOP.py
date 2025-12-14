# import matplotlib
# matplotlib.use('Agg')
from robocop.utils import visualization
from robocop import get_posterior_binding_probability_df
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import sys
import re
import pickle
import random
import glob
import h5py
import os
from scipy import sparse
import configparser
from robocop.utils.plotMNaseMidpoints import plotMidpointsAx
random.seed(9)

def get_idx(chrm, start, end, coords):
    coords = coords[(coords['chr'] == chrm) & (start <= coords['end']) & (end >= coords['start'])]
    return list(coords.index)

def get_sparse(f, k):
    g = f[k]
    v_sparse = sparse.csr_matrix((g['data'][:],g['indices'][:], g['indptr'][:]), g.attrs['shape'])
    return v_sparse

def get_sparse_todense(f, k):
    v_dense = np.array(get_sparse(f, k).todense())
    if v_dense.shape[0]==1: v_dense = v_dense[0]
    return v_dense

def calc_posterior(allinfofiles, dshared, coords, chrm, start, end):
    idxs = get_idx(chrm, start, end, coords)
    ptable = np.zeros((end - start + 1, dshared['n_states']))
    longCounts = np.zeros(end - start + 1)
    shortCounts = np.zeros(end - start + 1)
    counts = np.zeros(end - start + 1)
    for infofile in allinfofiles:
        for i in range(len(idxs)):
            idx = idxs[i]
            f = h5py.File(infofile, mode = 'r')
            k = 'segment_' + str(idx)
            if k not in f.keys():
                f.close()
                continue
            dshared['info_file'] = f
            dp = get_sparse_todense(f, k+'/posterior') # f[k + '/posterior'][:]
            lc = get_sparse_todense(f, k+'/MNase_long') # f[k + '/MNase_long'][:]
            sc = get_sparse_todense(f, k+'/MNase_short') # f[k + '/MNase_short'][:]
            f.close()
            dp_start = max(0, start - coords.loc[idx]['start'])
            dp_end = min(end - coords.loc[idx]['start'] + 1, coords.loc[idx]['end'] - coords.loc[idx]['start'] + 1)
            ptable_start = max(0, coords.loc[idx]['start'] - start)
            ptable_end = ptable_start + dp_end - dp_start
            ptable[ptable_start : ptable_end] += dp[dp_start : dp_end, :] 
            longCounts[ptable_start : ptable_end] += lc[dp_start : dp_end]
            shortCounts[ptable_start : ptable_end] += sc[dp_start : dp_end]
            counts[ptable_start : ptable_end] += 1 

    if counts[counts > 0].shape != counts.shape:
        print("ERROR: Invalid coordinates " + chrm + ":" + str(start) + "-" + str(end))
        print("Valid coordinates can be found at coords.tsv")
        sys.exit(0)
    ptable = ptable / counts[:, np.newaxis]
    longCounts = longCounts / counts
    shortCounts = shortCounts / counts
    optable = get_posterior_binding_probability_df(dshared, ptable)
    return optable, longCounts, shortCounts
    
def colorMap(outDir):
    if os.path.isfile(outDir + 'dbf_color_map.pkl'):
        dbf_color_map = pickle.load(open(outDir + "dbf_color_map.pkl", "rb"))
        return dbf_color_map

    print("Color map is not defined. Generating color map...")
    pwm = pickle.load(open(outDir + 'pwm.p', "rb"))
    predefined_dbfs = list(pwm.keys())
    # get upper case
    predefined_dbfs = [x for x in predefined_dbfs if x != "unknown"]
    predefined_dbfs = list(set([(x.split("_")[0]).upper() for x in predefined_dbfs]))
    n_tfs = len(predefined_dbfs)
    colorset48 = [(random.random(), random.random(), random.random(), 1.0) for i in range(n_tfs)] 
    nucleosome_color = '0.7'
    
    dbf_color_map = dict(list(zip(predefined_dbfs, colorset48)))
    dbf_color_map['nucleosome'] = nucleosome_color
    dbf_color_map['unknown'] =  '#D3D3D3'

    pickle.dump(dbf_color_map, open(outDir + "dbf_color_map.pkl", "wb"))
    print("Color map saved as", outDir + "dbf_color_map.pkl")
    return dbf_color_map


def plotRegion(gtffile, chrm, start, end, ax):
    a = pd.read_csv(gtffile, sep = "\t", header = None, comment = '#')
    a = a[(a[0] == chrm[3:]) & (a[3] <= end) & (a[4] >= start)]
    transcripts = {}
    # ax = plt.gca()
    for i, r in a.iterrows():
        if r[2] == 'transcript':
            if r[6] == '+':
                  ax.add_patch(patches.Rectangle((r[3], 0.4), r[4] - r[3] + 1, 0.3, color = 'skyblue'))
            else:
                  ax.add_patch(patches.Rectangle((r[3], -0.7), r[4] - r[3] + 1, 0.3, color = 'lightcoral'))
            gene_splits = dict([(g.split()[0], g.split()[1][1:-1]) for g in r[8][:-1].split(';')])
            gene = gene_splits['gene_name'] if 'gene_name' in gene_splits else gene_splits['gene_id']
            if gene not in transcripts:
                transcripts[gene] = (r[3], r[4], r[6])
            else:
                transcripts[gene] = (min(r[3], transcripts[gene][0]), max(r[4], transcripts[gene][1]), r[6])
        elif r[2] == 'exon': 
            if r[6] == '+':
                  ax.add_patch(patches.Rectangle((r[3], 0.1), r[4] - r[3] + 1, 0.9, color = 'skyblue'))
            else:
                  ax.add_patch(patches.Rectangle((r[3], -1), r[4] - r[3] + 1, 0.9, color = 'lightcoral'))
            gene_splits = dict([(g.split()[0], g.split()[1][1:-1]) for g in r[8][:-1].split(';')])
            gene = gene_splits['gene_name'] if 'gene_name' in gene_splits else gene_splits['gene_id']
        
    for t in transcripts:
        if transcripts[t][2] == '+':
            if transcripts[t][0] + 10 < start:
                ax.text(start, 1.2, t, fontsize = 12)
            else:
                ax.text(transcripts[t][0] + 10, 1.2, t, fontsize = 12)
        else:
            if transcripts[t][0] + 10 < start:
                ax.text(start, -2, t, fontsize = 12)
            else:
                ax.text(transcripts[t][0] + 10, -2, t, fontsize = 12)

    ax.set_xlim((start, end))
    ax.set_ylim((-1.9, 1.9))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

def plotOutput(outDir, config, dbf_color_map, optable, chrm, start, end,
               tech, longCounts, shortCounts, save=True, gtffile=None,
               fiber_info=None):
    fragRangeLong = tuple([int(x) for x in re.findall(r'\d+', config.get("main", "fragRangeLong"))])
    fragRangeShort = tuple([int(x) for x in re.findall(r'\d+', config.get("main", "fragRangeShort"))])

    offset = 4 if tech == "ATAC" else 0

    if gtffile is not None:
        nrows = 6 if fiber_info is not None else 4
        ratios = [0.3, 1, 0.5, 1, 0.25, 0.25] if fiber_info else [0.3, 1, 0.5, 1]
        fig, ax = plt.subplots(nrows, 1, figsize=(19, 10), gridspec_kw={'height_ratios': ratios})
        isgtf = 1
    else:
        nrows = 5 if fiber_info is not None else 3
        fig, ax = plt.subplots(nrows, 1, figsize=(19, 9))
        isgtf = 0

    # MNase coverage plot
    # ax[1 + isgtf].plot(list(range(start - 1, end)), longCounts, color='maroon')
    # ax[1 + isgtf].plot(list(range(start - 1, end)), shortCounts, color='blue')

    bamFile = config.get("main", "bamFile")
    shortCounts, longCounts = plotMidpointsAx(
        ax[0 + isgtf], bamFile, chrm, start, end,
        fragRangeShort, fragRangeLong, offset=offset
    )

    # NEW: plot Fiber-seq data if provided
    if fiber_info is not None:
        plotFiberseqAx(ax[3 + isgtf], ax[4 + isgtf], fiber_info, start, end)

    # RoboCOP occupancy
    visualization.plot_occupancy_profile(
        ax[2 + isgtf], op=optable, chromo=chrm,
        coordinate_start=start, threshold=0.1,
        dbf_color_map=dbf_color_map
    )

    # Axis formatting
    for j in range(3 + isgtf):
        ax[j].set_xlim((start, end))
        ax[j].set_xticks([])

    ax[2 + isgtf].set_xlabel(chrm)

    if isgtf:
        plotRegion(gtffile, chrm, start, end, ax[0])

    if save:
        os.makedirs(outDir + 'figures/', exist_ok=True)
        outfile = f"{outDir}figures/robocop_output_{chrm}_{start}_{end}.png"
        plt.savefig(outfile)
        print("Output saved:", outfile)
    else:
        plt.show()

def plot_output(outDir, chrm, start, end, save = True):
    outDir = outDir + '/' if outDir[-1] != '/' else outDir

    configFile = outDir + "config.ini"
    # DEBUG: show exactly where we’re looking
    print("Current working directory:", os.getcwd())
    print("Absolute path to configFile:", os.path.abspath(configFile))
    print("Does it exist?:", os.path.exists(configFile))

    config = configparser.SafeConfigParser()
    config.read(configFile)
    print("configFile (as read):", configFile)
    print("Sections found:", config.sections())
    
    # hmmconfigfile is generated only with robocop_em.py
    # if outputDir is generated using robocop_no_em.py
    # then use the dir path used to run robocop_no_em.py
    # to get the hmmconfig file
    
    hmmconfigfile = config.get("main", "trainDir") + "/HMMconfig.pkl"

    tech = config.get("main", "tech")

    gtfFile = config.get("main", "gtfFile")
    # create file for plotting
    dshared = pickle.load(open(hmmconfigfile, "rb"))

    coords = pd.read_csv(outDir + "coords.tsv", sep = "\t")
    allinfofiles = glob.glob(outDir + 'tmpDir/info*.h5')

    optable, longCounts, shortCounts = calc_posterior(allinfofiles, dshared, coords, chrm, start, end)

    #Need to implement a check to see if fiberseq file is there, then run this next code
    fiber_info = plot_fiberseq(allinfofiles, coords, chrm, start, end, tech="Fiber")

    dbf_color_map = colorMap(outDir)
    plotOutput(outDir, config, dbf_color_map, optable, chrm, start, end, tech,
               longCounts, shortCounts, save, gtffile=gtfFile,
               fiber_info=fiber_info)  # << pass in fiberseq

def plot_fiberseq(allinfofiles, coords, chrm, start, end, tech):
    """
    Load Fiber-seq count arrays (Watson/Crick meth/A) from info files.

    Arguments:
        allinfofiles : list of info_*.h5 files
        coords : pd.DataFrame with segment coordinates
        chrm, start, end : region to plot
        tech : name of assay, e.g. 'fiberseq'

    Returns:
        dict[str, np.ndarray] with keys:
        ['count_meth_watson', 'count_meth_crick',
         'count_A_watson', 'count_A_crick']
    """
    idxs = get_idx(chrm, start, end, coords)

    # Initialize arrays for region length
    region_len = end - start + 1
    result = {
        'count_meth_watson': np.zeros(region_len),
        'count_meth_crick': np.zeros(region_len),
        'count_A_watson': np.zeros(region_len),
        'count_A_crick': np.zeros(region_len),
        'count': np.zeros(region_len)  # keep track how many segments contributed
    }

    # iterate through all info files and all segments that overlap region
    for infofile in allinfofiles:
        for idx in idxs:
            f = h5py.File(infofile, mode='r')
            print(f"\nContents of {infofile}:")
            f.visit(print)
            k = f'segment_{idx}'
            # skip if the segment key is not in file (like calc_posterior)
            if k not in f.keys():
                f.close()
                continue

            try:
                #print(f"{k}/{tech}_count_meth_watson")
                
                cmw = get_sparse_todense(f, f"{k}/{tech}_count_meth_watson")
                cmc = get_sparse_todense(f, f"{k}/{tech}_count_meth_crick")
                caw = get_sparse_todense(f, f"{k}/{tech}_count_A_watson")
                cac = get_sparse_todense(f, f"{k}/{tech}_count_A_crick")
                print('we hereeeee')
                print(cmw)
                print(cmc)
                print(caw)
                print(cac)
            except Exception as e:
                print(f"Warning: could not load Fiber-seq datasets from {infofile}, {k}: {e}")
                print('bob')
                f.close()
                continue
            f.close()

            seg_start = coords.loc[idx]['start']
            seg_end = coords.loc[idx]['end']

            dp_start = max(0, start - seg_start)
            dp_end = min(end - seg_start + 1, seg_end - seg_start + 1)
            p_start = max(0, seg_start - start)
            p_end = p_start + dp_end - dp_start

            result['count_meth_watson'][p_start:p_end] += cmw[dp_start:dp_end]
            result['count_meth_crick'][p_start:p_end] += cmc[dp_start:dp_end]
            result['count_A_watson'][p_start:p_end] += caw[dp_start:dp_end]
            result['count_A_crick'][p_start:p_end] += cac[dp_start:dp_end]
            result['count'][p_start:p_end] += 1

    # Average over contributing segments
    for key in ['count_meth_watson', 'count_meth_crick', 'count_A_watson', 'count_A_crick']:
        mask = result['count'] > 0
        result[key][mask] /= result['count'][mask]

    # Remove count tracking from result before returning
    del result['count']
    return result

def plotFiberseqAx(ax_w, ax_c, fiber_info, start, end):
    """
    Plot Fiber‑seq methyl/A ratios for Watson and Crick strands
    into two separate axes.

    Parameters
    ----------
    ax_w, ax_c : matplotlib Axes
        Axes for Watson and Crick strand ratio plots.
    fiber_info : dict
        Output of plot_fiberseq() with count arrays.
    start, end : int
        Genomic interval (inclusive start, exclusive end).
    """
    x = np.arange(start, end + 1)

    # Compute ratios safely (avoid divide-by-zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_w = np.where(fiber_info["count_A_watson"] > 0,
                           fiber_info["count_meth_watson"] / fiber_info["count_A_watson"],
                           np.nan)
        ratio_c = np.where(fiber_info["count_A_crick"] > 0,
                           fiber_info["count_meth_crick"] / fiber_info["count_A_crick"],
                           np.nan)

    ax_w.scatter(x, ratio_w, s=1 ,color="blue", alpha=0.7, label="Watson meth/A")
    ax_c.scatter(x, ratio_c, s=1, color="orange", alpha=0.7, label="Crick meth/A")

    for ax, title in zip([ax_w, ax_c], ["Watson strand", "Crick strand"]):
        ax.set_xlim(start, end)
        ax.set_ylim(0, 1)        # ratio is 0–1
        ax.set_ylabel("meth/A ratio")
        ax.legend(loc="upper right", frameon=False)
        ax.set_title(title)

    
if __name__ == '__main__':

    if len(sys.argv) != 5:
        print("Usage: python plotRoboCOP.py outDir chr start end")
        sys.exit(0)

    outDir = (sys.argv)[1]
    chrm = (sys.argv)[2]
    start = int((sys.argv)[3])
    end = int((sys.argv)[4])
    plot_output(outDir, chrm, start, end)
