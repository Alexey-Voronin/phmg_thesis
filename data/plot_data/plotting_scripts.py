import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
from common import set_figure, fig_size


FONT_SIZE = 17

def get_frame(file_name):
    df = pd.read_pickle(file_name)

    for name in [#'resids',
                 'nu', 'Re', #'ncells',
                 'udofs', 'pdofs']:
        try:
            df.pop(name)
        except:
            pass

    df = df.reset_index(drop=True)

    return df

def plot_all(data, save_name=None, disc="TH", time_per_iter=True, warm=False):
    ncols = len(data.keys())
    nrows = 3 if time_per_iter else 2

    fs  = fig_size.singlefull
    set_figure(width=0.5*fs['height']*ncols, height=0.7*fs['height']*nrows)
    fig = plt.figure()

    gs = GridSpec(nrows, ncols + 1, figure=fig,
                  width_ratios=([1]*ncols+[0.05]))


    warm_str = "Warm" if warm else "Warm-up"

    plot_field(data,  #save_name='alfi_rel',
             vmax=None, fig_gs=(fig, gs), row=0)
    plot_relatives(data, field_name=f"SNESSolve(time):{warm_str}", fig_gs=(fig, gs), row=1)
    if time_per_iter:
        plot_relatives(data, field_name=f"SNESSolve(time):{warm_str}", fig_gs=(fig, gs), row=2, per_iter_cost=True)

    # shared labels
    fig.text(-0.09, 0.5, '$\\bf{order}$ ($k$)', va='center', rotation='vertical', fontsize=FONT_SIZE)
    fig.text(0.5, -0.03, '$\\bf{refinements}$', ha='center', fontsize=FONT_SIZE)

    # row labels
    if time_per_iter:
        fig.text(-0.03, 0.85, 'iterations', va='center', rotation='vertical', fontsize=FONT_SIZE)
        fig.text(-0.03, 0.5, 'rel. time to solution', va='center', rotation='vertical', fontsize=FONT_SIZE)
        fig.text(-0.03, 0.2, 'rel. time per iter.', va='center', rotation='vertical', fontsize=FONT_SIZE)
    else:
        fig.text(-0.03, 0.75, 'iterations', va='center', rotation='vertical', fontsize=FONT_SIZE)
        fig.text(-0.03, 0.25, 'rel. time to solution', va='center', rotation='vertical', fontsize=FONT_SIZE)

    if save_name:
        plt.savefig(f'{save_name}.pdf', bbox_inches='tight')


def plot_field(data, field_name='linear_iter',
               title=None, fontsize=FONT_SIZE,
               save_name=None, vmax=None,
               invert_clr_bar=True,rel_plot=False, fig_gs=False, row=0):
    # Determine global min and max for the color scale
    all_values = np.concatenate([df.pivot(index='order', columns='ref', values=field_name).values.ravel() for df in data.values()])
    vmin, vmax = min(all_values), vmax or max(all_values)

    nframes = len(data.keys())
    ncols = nframes

    col_offset = 1 if rel_plot else 0

    if not fig_gs:
        fs  = fig_size.singlefull
        set_figure(width=0.5*fs['height']*ncols, height=0.7*fs['height'])
        fig = plt.figure()

        ncols     += col_offset
        gs = GridSpec(1, ncols + 1, figure=fig,
                      width_ratios=([1]*ncols+[0.05]))
    else:
        fig, gs = fig_gs

    cmap='RdYlGn' if not invert_clr_bar else 'winter_r' #'RdYlGn'

    if not invert_clr_bar:
        vcenter = 1
        if vmin < vcenter and vcenter < vmax:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter,  vmax=vmax)
        elif vmin < vcenter:
            # ValueError: vmin, vcenter, and vmax must be in ascending order
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter,  vmax=(vcenter+1e-8))
        elif vcenter < vmax:
            norm = TwoSlopeNorm(vmin=vcenter, vcenter=(vcenter+1e-8),  vmax=vmax)
        else:
            print('NO CENTRERING of colorbar!!!')
            norm = None #TwoSlopeNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    if col_offset:
        col = 0
        ax = fig.add_subplot(gs[row, col])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.axis('off')

    for i, (name, df) in enumerate(data.items()):
        col = i + col_offset
        ax = fig.add_subplot(gs[row, col])


        pivot_df = df.pivot(index='order', columns='ref', values=field_name)
        sns.heatmap(pivot_df, annot=True, cmap=cmap, fmt="g",
                    linewidths=.5, annot_kws={"size": fontsize}, ax=ax,
                    vmin=vmin, vmax=vmax, cbar=False, norm=norm)
        ax.invert_yaxis()

        if row == 0:
            ax.set_title(f'{name}', fontsize=fontsize)

        if fig_gs:
            ax.set_ylabel("")
            if i > 0:
                ax.set_yticks([])
            if row == 0 and i > 0:
                ax.set_xticks([])
        elif i == 0:
            ax.set_ylabel("order", fontsize=fontsize)
        else:
            ax.set_ylabel("")
            ax.set_yticks([])


        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=fontsize)
        ax.set_xlabel("")


    if i == 0 and not fig_gs:
        fig.text(0.475, 0.01, 'refinements', ha='center', va='center', fontsize=fontsize)
    if title:
        plt.suptitle(title, fontsize=fontsize)

    # Add shared color bar
    cbar_ax = fig.add_subplot(gs[row, -1])  # Color bar in the last column of GridSpec
    norm = plt.Normalize(vmin=vmin, vmax=vmax) if field_name == 'linear_iter' else norm
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)


    from matplotlib.ticker import MaxNLocator
    cbar = fig.colorbar(sm, cax=cbar_ax)

    if field_name == 'linear_iter':
        def format_func(value, tick_number):
            return f"{int(value)}"
    else:
        def format_func(value, tick_number):
            return f"{value:.1f}"
    from matplotlib.ticker import FuncFormatter
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_func))

    if not fig_gs:
        if save_name:
            plt.savefig(f'{save_name}.pdf', bbox_inches='tight')
        plt.show()


"""
def plot_relatives(data, title=None, field_name = "linear_iter",
                   save_name=None, invert_clr_bar=False, per_iter_cost=False, fig_gs=False, row=0):
    print('Relative cost per iteration') if per_iter_cost else print('Relative cost per run')
    merged_data = {}
    keys = list(data.keys())
    k1 = keys[0]
    df1= data[k1]


    df_hi = df1[df1['order'] > 5][field_name]
    print(f"{k1:10s}: {np.sum(df_hi):5.2f} |  {np.sum(df1[field_name]):5.2f}")

    for k2 in keys[1:]:
        df2= data[k2]
        merged_name = f'{k2}'
        # first four columns from dfs are shared
        ratio_frame = df1[['order', 'ref', 'ndofs']].copy()
        # Calculate the ratios

        df_hi = df2[df2['order'] > 5][field_name]
        print(f"{k2:10s}: {np.sum(df_hi):5.2f} |  {np.sum(df2[field_name]):5.2f}")

        if not per_iter_cost:
            ratio_frame[field_name]  = np.round(df1[field_name]/df2[field_name],2)
        else:
            q1 = df1[field_name]/df1["linear_iter"]
            q2 = df2[field_name]/df2["linear_iter"]
            ratio_frame[field_name]  = np.round(q1/q2,1)

        merged_data[merged_name] = ratio_frame

    plot_field(merged_data, field_name,  title=title,
               save_name=save_name, invert_clr_bar=invert_clr_bar, rel_plot=True, fig_gs=fig_gs, row=row)

"""

def plot_relatives(data, title=None, field_name="linear_iter",
                   save_name=None, invert_clr_bar=False, per_iter_cost=False, fig_gs=False, row=0):
    print('Relative cost per iteration' if per_iter_cost else 'Relative cost per run')

    # Collect timing and ratio information
    timing_info = []
    merged_data = {}
    base_df = data[next(iter(data))]  # Assume first item as base for comparison

    for k, df in data.items():
        sum_hi = np.sum(df[df['order'] > 5][field_name])
        sum_total = np.sum(df[field_name])
        timing_info.append((k, sum_hi, sum_total))

        if k != next(iter(data)):  # Skip the base dataframe
            ratio_frame = df[['order', 'ref', 'ndofs']].copy()
            if per_iter_cost:
                q1 = base_df[field_name] / base_df["linear_iter"]
                q2 = df[field_name] / df["linear_iter"]
                ratio = np.round(q1/q2, 1)
            else:
                ratio = np.round(base_df[field_name] / df[field_name], 2)
            ratio_frame[field_name] = ratio
            merged_data[k] = ratio_frame

    # Sort the timing information by high order sum then by total sum
    timing_info.sort(key=lambda x: (x[1], x[2]))

    # Print the sorted timing information
    max_key_length = max(len(k) for k, _, _ in timing_info)
    title="Name"; title2 = "k>=5"; title3="k>2"
    print(f"{title:{max_key_length}} : {title2:6s} | {title3:6s}")
    for k, sum_hi, sum_total in timing_info:
        #print(f"{k:40s}: {sum_hi:5.2f} |  {sum_total:5.2f}")
        print(f"{k:{max_key_length}} : {sum_hi:6.2f} | {sum_total:6.2f}")

    # The rest of the function would use 'merged_data' as needed
    plot_field(merged_data, field_name,  title=None,
               save_name=save_name, invert_clr_bar=invert_clr_bar, rel_plot=True, fig_gs=fig_gs, row=row)

