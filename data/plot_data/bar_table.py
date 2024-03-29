# from bar_plots import *

import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter
import numpy as np
import pandas as pd
from common import *
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def adjust_color(color, factor):
    """
    Adjust the color by a certain factor.
    Lighten if factor > 1, darken if factor < 1.
    """
    converter = ColorConverter()
    rgb = converter.to_rgb(color)  # Convert hex to RGB
    # Adjust each channel
    adjusted = tuple(min(max(0, channel * factor), 1) for channel in rgb)
    # Convert back to hex
    return '#%02x%02x%02x' % (int(adjusted[0] * 255), int(adjusted[1] * 255), int(adjusted[2] * 255))

def plot_fraction(data_dict, save_name=None, with_setup=False, th_disc=True):

    #######################################################################
    #  plot setup
    ncols = len(list(data_dict.keys()))
    nrows = 1
    fs = fig_size.singlefull
    set_figure(width=fs["height"]*(ncols*1.), height=1.*fs["height"])
    fig, axs_all = plt.subplots(nrows, ncols, sharey="row", sharex="col")
    # Font stuff
    plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts,amsmath}'
    })
    text_font_size = 20
    number_font_size = 14

    #########################################################################
    # Exctract Data

    # First item in the dictionary is always reference solver
    ref_name  = list(data_dict.keys())[0]
    ref_frame = list(data_dict.values())[0]
    # exctract these values for the table
    table_data = {'T(solve)' : [], 'T(setup)' : [], 'T(rlx)/T(solve)' : [], 'T(setup)/T(total)' : [], 'iterations' : []}
    # plot formatting
    ymax = -1

    for solver_name, data, ax in zip(data_dict.keys(), data_dict.values(), axs_all.ravel()):
        ##########################################################################################
        # Prepare data for stacked bar plot
        stacked_data = []
        col_table_data = {'T(solve)' : [], 'T(setup)' : [], 'T(rlx)/T(solve)' : [], 'T(setup)/T(total)' : []}


        # Use warm relaxation timings because they are not polluted by the calls to setup routines.
        rlx_type_expected = f'ASMPatch(times):Warm'
        # phmg and hmg solver use ASMPatch. afbf solver uses PatchPC
        rlx_type  = rlx_type_expected if rlx_type_expected in data.columns else f'PatchPC(times):Warm'

        try:
            max_lvls = max(len(l) if not np.isnan(l).any() else 0 for l in data[rlx_type])
        except:
            max_lvls = 0
        """ref timings are form the reference solvers (e.g. {TH: hmg, SV: afbf(hmg)}.
        """
        for (ref_c_slv, ref_w_slv, c_setup, c_solve, w_solve,
        rlx_times_list, w_prolong, w_restrict, w_jacobian) in zip(
                                                              ref_frame[f'SNESSolve(time):Warm-up'],
                                                              ref_frame[f'SNESSolve(time):Warm'],
                                                              data[f'PCSetUp(time):Warm-up'],
                                                              data[f'SNESSolve(time):Warm-up'],
                                                              data[f'SNESSolve(time):Warm'],
                                                              data[rlx_type], #coarse to fine
                                                              data['MatMultAdd(time):Warm'],
                                                              data['MatMultTranspose(time):Warm'],
                                                              data['MatMult(time):Warm']
        ):

            # setup = SNESSolve(warm-up) - SNESSolve(warm)
            # This works only when we don't rebuild anything for the 2nd solve.
            # Counting induvidual PCSetUp is too hard because they are nested..
            setup_time = c_solve - w_solve
            #setup_time = c_setup # c_solve - w_solve
            # Pad with zeros because phMG(gradual) for lower-order discretization
            # will have fewer levels than higher-order preconditioners.
            # Do not change the order !!!
            rlx_times_list = [] if  np.isnan(rlx_times_list).any() else rlx_times_list
            rlx_time   = [0]*(max_lvls - len(rlx_times_list)) + rlx_times_list
            grid_tranfer_time = w_prolong+w_restrict

            # PETSc double counts MatMult when afbf solver is used. Just count it toward
            # other for now..
            w_jacobian = w_jacobian #if rlx_type == rlx_type_expected else 0,

            other_time = w_solve - np.sum(rlx_time) - grid_tranfer_time - w_jacobian
            ref_time   = ref_c_slv if with_setup else ref_w_slv
            setup_time_list = [setup_time] if with_setup else []
            stacked_data.append(np.array(setup_time_list + [
                                        other_time,
                                        w_jacobian,
                                        grid_tranfer_time] + rlx_time) / ref_time)

            # table data
            col_table_data['T(solve)'].append(w_solve)
            col_table_data['T(setup)'].append(setup_time)
            col_table_data['T(rlx)/T(solve)'].append(np.sum(rlx_time)/col_table_data['T(solve)'][-1])
            col_table_data['T(setup)/T(total)'].append(setup_time/c_solve)
            # formatting
            ymax = max(ymax, np.sum(stacked_data[-1]))

        table_data['T(solve)'].append(col_table_data['T(solve)'])
        table_data['T(setup)'].append(col_table_data['T(setup)'])
        table_data['T(rlx)/T(solve)'].append(col_table_data['T(rlx)/T(solve)'])
        table_data['T(setup)/T(total)'].append(col_table_data['T(setup)/T(total)'])
        table_data['iterations'].append(data['linear_iter'].values)
        ##########################################################################################
        # Legend labels and colors
        x_labels = [f"{o}" for o, r in zip(data['order'], data['ref'])]
        legends = [#'setup',
                   'other solve', 'Jacobian', 'grid transfer'] +\
                           [f"rlx($\ell={i}$)" for i in range(max_lvls)][::-1] #\ell=0 is the finest level!!!



        base_color = 'tab:purple'
        color_map = {
            'other solve': 'tab:olive',
            'Jacobian': 'tab:brown',
            'grid transfer': 'tab:blue',
            'setup': 'red',
            # h
            'rlx($\ell=0$)': adjust_color(base_color, 0.7),
            'rlx($\ell=1$)': adjust_color(base_color, 1.0),
            'rlx($\ell=2$)': adjust_color(base_color, 1.3),
            'rlx($\ell=3$)': adjust_color(base_color, 1.6),
            'rlx($\ell=4$)': adjust_color(base_color, 1.9),
            'rlx($\ell=5$)': adjust_color(base_color, 2.2),
            #'rlx($\ell=0$)': adjust_color(base_color, 1.9),
            #'rlx($\ell=1$)': adjust_color(base_color, 1.6),
            #'rlx($\ell=2$)': adjust_color(base_color, 1.3),
            #'rlx($\ell=3$)': adjust_color(base_color, 1.),
            #'rlx($\ell=4$)': adjust_color(base_color, 0.7),
            #'rlx($\ell=5$)': adjust_color(base_color, 0.5),
        }
        colors = [color_map[legend] for legend in legends]
        #############################################################################################
        # Plotting



        # Convert to a DataFrame for easier manipulation
        df_stacked = pd.DataFrame(stacked_data).T  # Transpose to have each row represent a layer of the stack
        #df_stacked = np.nan_to_num(df_stacked, nan=0)
        df_stacked = df_stacked.fillna(0)
        bottom = np.zeros(len(x_labels))  # Starting point for the bottom of each bar
        for idx, (row_idx, row) in enumerate(df_stacked.iterrows()):
            ax.bar(x_labels, row, bottom=bottom, label=legends[idx], color=colors[idx])
            bottom += row  # Update bottom for the next layer

        ax.set_title(f'{solver_name}', fontsize=text_font_size,y=1.05)
        ax.grid()

        # Set tick labels with specific font size
        ax.tick_params(axis='both', which='major', labelsize=12)

    ###########################################################################################################
    # Prepare data for the table
    orders = np.arange(3, 11).repeat(4).reshape(-1, 4)

    tmp = np.array(table_data['T(solve)']).T
    slv_scaled = tmp[:, 0].reshape(-1, 1)/tmp

    tmp = np.array(table_data['T(setup)']).T
    setup_scaled = tmp[:, 0].reshape(-1, 1)/tmp

    tmp = np.array(table_data['T(solve)']).T+ np.array(table_data['T(setup)']).T
    tt_scaled = tmp[:, 0].reshape(-1, 1)/tmp

    rlx_over_solve = np.array(table_data['T(rlx)/T(solve)']).T

    setup_over_total = np.array(table_data['T(setup)/T(total)']).T

    iters_all = np.array(table_data['iterations']).T


    # Compute Norm Bounds
    scale_up = 1.5 # deep reds and greens look too dark
    scale_down = 0.6 # deep reds and greens look too dark

    all_vals_scaled = np.stack([slv_scaled,setup_scaled,tt_scaled]).ravel()
    all_vals_scaled = all_vals_scaled[~np.isnan(all_vals_scaled)]
    norm_scaled_bounds = (np.min(all_vals_scaled)*scale_down, np.max(all_vals_scaled)*scale_up)

    all_vals_iter = iters_all.ravel()
    # non/poorly convergent methods sckew the color-map.
    all_vals_iter = all_vals_iter[np.where(all_vals_iter < 40)]
    norm_iter_bounds = (np.min(all_vals_iter)*0.9, np.max(all_vals_iter)*1.3)

    all_setup_over_total = setup_over_total.ravel()
    all_setup_over_total = all_setup_over_total[~np.isnan(all_setup_over_total)]
    norm_setup_over_total_bounds = (np.min(all_setup_over_total),
                                    np.max(all_setup_over_total)*scale_up)

    all_rlx_over_solve = rlx_over_solve.ravel()
    all_rlx_over_solve = all_rlx_over_solve[~np.isnan(all_rlx_over_solve)]
    norm_rlx_over_solve_bounds = (np.min(all_rlx_over_solve)*scale_down,
                                  np.max(all_rlx_over_solve)*scale_up)

    # Function to format numbers to at most two digits
    def format_numbers(arr):
        def custom_format(x):
            if 0 < x < 1:
                return f"{x:.2f}".lstrip('0')  # Strip leading '0' for numbers between 0 and 1
            elif x == 0:
                return "0"  # Ensure 0 is just "0"
            elif x < 10:
                return f"{x:.2g}"
            else:
                return f"{x:.0f}"
        formatted = np.vectorize(custom_format)(arr)
        return formatted

    plt.tight_layout(pad=4.0, h_pad=4.0, w_pad=2.0)

    row_order = np.array([0, 6, 3, 2, 1, 4])

    table_row_labels_all = [
                        r'\textbf{order} ($k$)', # 0
                        'Rel. T(solve)',         # 1
                        'Rel. T(setup)',         # 2
                        'Rel. T(total)',         # 3
                        'Iterations',            # 4
                        'T(rlx)/T(solve)',       # 5
                        'T(setup)/T(total)',     # 6
                       ]

    for j, ax in enumerate(axs_all):
        x_labels = [f"{o}" for o, r in zip(data_dict[list(data_dict.keys())[j]]['order'], data_dict[list(data_dict.keys())[j]]['ref'])]

        # Apply the formatting function to each column of data
        formatted_tt_scaled = format_numbers(tt_scaled[:, j])
        formatted_setup_scaled = format_numbers(setup_scaled[:, j])
        formatted_slv_scaled = format_numbers(slv_scaled[:, j])
        formatted_rlx_over_solve = format_numbers(rlx_over_solve[:, j])
        formatted_setup_over_total = format_numbers(setup_over_total[:, j])
        formatter_orders = [r'\textbf{%d}' % o for o in orders[:,j]]
        formatter_iters = [r'%d' % i for i in iters_all[:,j]]

        cell_text = np.vstack([formatter_orders,
                               formatted_slv_scaled,
                               formatted_setup_scaled,
                               formatted_tt_scaled,
                               formatter_iters,
                               formatted_rlx_over_solve,
                               formatted_setup_over_total,
                              ])

        cell_text = cell_text[row_order]
        table_row_labels = [ table_row_labels_all[i] for i in row_order]

        if j == 0:
            row_labels = table_row_labels
        else:
            row_labels = [""] * len(table_row_labels)  # Keep the structure, but no visible labels


        def post_process(v):
            if v == '0':
                return '-'
            elif v == '80': # max_iter
                return r'80\texttt{+}'
            else:
                return v

        # do not modify the cell_text array with non-int strings, as that will
        # break the cell-coloring functionality later
        cell_text = np.where(cell_text == 'nan', '0', cell_text)
        # Create the table directly below the current subplot
        table = ax.table(
            cellText=[[post_process(c) for c in row_list] for row_list in cell_text],
            rowLabels=row_labels,
            #colLabels=x_labels,
            cellLoc='center',
            loc='bottom',
            bbox=[0.03, -0.63, 0.93, 0.60] # x0, y0, width, height
        )
        table.auto_set_font_size(False)
        table.set_fontsize(text_font_size-5)
        table.scale(1, 1.5)

        for key, cell in table.get_celld().items():
           cell.set_edgecolor((0.8, 0.8, 0.8, 0.5)) # gray, 50% transparent


        ###################################
        # Adjust cell colors based on value
        cells = table.get_celld()

        # Normalize object to map cell values to [0, 1]
        norm_scaled = matplotlib.colors.TwoSlopeNorm(vmin=norm_scaled_bounds[0],
                                                     vcenter=1,
                                                     vmax=norm_scaled_bounds[1])
        cmap_scaled = plt.get_cmap('RdYlGn')
        mappables_scaled = ScalarMappable(norm=norm_scaled, cmap=cmap_scaled)

        norm_scaled = Normalize(vmin=norm_iter_bounds[0], vmax=norm_iter_bounds[1])
        cmap_scaled = plt.get_cmap('RdYlGn_r')
        mappables_iter = ScalarMappable(norm=norm_scaled, cmap=cmap_scaled)

        norm_scaled = Normalize(vmin=norm_setup_over_total_bounds[0], vmax=2.1)#norm_setup_over_total_bounds[1])
        cmap_scaled = plt.get_cmap('binary')
        mappables_setup_over_total = ScalarMappable(norm=norm_scaled, cmap=cmap_scaled)

        norm_scaled = Normalize(vmin=norm_rlx_over_solve_bounds[0], vmax=norm_rlx_over_solve_bounds[1])
        cmap_scaled = plt.get_cmap('cool')
        mappables_all_rlx_over_solve = ScalarMappable(norm=norm_scaled, cmap=cmap_scaled)

        for i, row in enumerate(cell_text):  # Iterate over rows in cell_text
            if table_row_labels[i] in [r'\textbf{order} ($k$)']:
                continue
            elif table_row_labels[i] in ['T(setup)/T(total)']:
                mappables = mappables_setup_over_total
            elif table_row_labels[i] in ['T(rlx)/T(solve)']:
                mappables = mappables_all_rlx_over_solve
            elif table_row_labels[i] in ['Iterations']:
                mappables = mappables_iter
            else:
                mappables = mappables_scaled

            for j, cell_str in enumerate(row):  # Iterate over columns/values in the row
                cell = cell_str.astype(float)

                if (not np.isnan(cell) and cell > 0) and cell != 80:
                    color = mappables.to_rgba(cell)
                else:
                    color = "white"

                table[(i, j)].set_facecolor(color)
                # Adjust text color for readability
                text_color = 'black' #if cell != 80 else 'red'
                table[(i, j)].get_text().set_color(text_color)

    plt.subplots_adjust(bottom=0.2)
    #######################################################################################
    # legend and other formatting

    # Place legend outside the plot with a specific font size
    axs_all = np.ravel(axs_all)

    # Collect handles and labels from all plots
    handles, labels = [], []
    for ax in axs_all:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:  # Check for duplicates
                handles.append(handle)
                labels.append(label)
    # Create a unified legend
    # currently hadnles and labels are in a weird order.
    # The following code prescribed top-down label/bar order of the legend.
    sorted_rlx_labels_handles = sorted([(label, handles[i]) for i, label in enumerate(labels) if "rlx" in label],
                                      key=lambda x: int(x[0][x[0].find('=')+1:x[0].find('$', x[0].find('='))]))
    sorted_rlx_labels, sorted_rlx_handles = zip(*sorted_rlx_labels_handles)
    # Keeping the non-rlx labels and their handles in place
    non_rlx_labels_handles = [(label, handles[i]) for i, label in enumerate(labels) if "rlx" not in label]
    # Unpacking non-rlx labels and handles
    non_rlx_labels, non_rlx_handles = zip(*non_rlx_labels_handles)
    # Combining sorted rlx labels and handles with non-rlx labels and handles
    sorted_labels = list(sorted_rlx_labels) + list(non_rlx_labels)[::-1]
    sorted_handles = list(sorted_rlx_handles) + list(non_rlx_handles)[::-1]

    # Legend on the right and ylabel titled:
    #axs_all[0].set_ylabel(f'Relative Time', fontsize=text_font_size)
    #axs_all[-1].legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    # Legend on the left
    #axs_all[0].legend(sorted_handles, sorted_labels, loc='center left',
    axs_all[0].legend(sorted_handles, sorted_labels, loc='center left',
                      bbox_to_anchor=(-0.68, 0.5), fontsize=14)

    for ax in axs_all:
        ax.set_ylim((0, ymax*1.05))
        ax.set_xticklabels([])

    #disc = '$\\boldsymbol{\\mathbb{P}}_k/\\mathbb{P}_{k-1}$' if th_disc else '$\\boldsymbol{\\mathbb{P}}_k/\\mathbb{P}_{k-1}^{disc}$'
    #fig.supxlabel('Discretization order, {}', fontsize=text_font_size)
    if save_name:
        plt.savefig(f'{save_name}.pdf')
    plt.show()

