import pandas as pd
import matplotlib.pyplot as plt

def plotHist1d_DataFrame(data_df, colname, outdir, xscale, yscale,
                         xmin=None, xmax=None, ymin=None, ymax=None):
    fig, ax = plt.subplots()
    range_tpl = None
    if (xmin != None) and (xmax != None):
        range_tpl = (xmin, xmax)

    data_df[colname].hist(bins=200, grid=True, label=colname, range=range_tpl)
    plt.xlabel(colname)
    plt.ylabel('frequency')
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    outfile_full = (outdir + "/" + 
                    f"akari_stat_{colname}.png")
    plt.savefig(outfile_full,
                bbox_inches='tight',
                pad_inches=0.1)
    print(outfile_full)
    plt.clf()
    plt.close()
    del fig
    del ax


def plotScatter_DataFrame(data_df, colname1, colname2, outdir,
                          xscale="linear", yscale="linear",
                          xmin=None, xmax=None, ymin=None, ymax=None):
    fig, ax = plt.subplots()
    plt.scatter(data_df[colname1], data_df[colname2])
    plt.xlabel(colname1)
    plt.ylabel(colname2)
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    outfile_full = (outdir + "/" + 
                    f"akari_stat_{colname1}_{colname2}.png")
    plt.savefig(outfile_full,
                bbox_inches='tight',
                pad_inches=0.1)
    print(outfile_full)
    plt.clf()
    plt.close()
    del fig
    del ax


