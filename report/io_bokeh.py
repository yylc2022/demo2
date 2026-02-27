from bokeh.io import show, output_file, reset_output, output_notebook, save
from bokeh.layouts import gridplot, column
from bokeh.models import RangeTool, NumberFormatter, StringFormatter, DateFormatter
from bokeh.models.sources import ColumnDataSource
from bokeh.models.widgets import DataTable, TableColumn, Div
from bokeh.palettes import Category20
from bokeh.plotting import figure
import os
import pandas as pd
import numpy as np
import copy

AUTHOR = "Anliang Song"
VERSION = '2.0.0'

IndexThreshold = 50000
MiddleThreshold = 200
ColumnThreshold = 22

def np_as_datetime(x):
    return np.array(x, dtype=np.datetime64)

def plot_div(str):
    return Div(text="<b>{title}</b>".format(title=str), width=1200, height=25)

def get_format(df, i):
    typestr = str(type(df.iat[0, i]))
    if 'float' in typestr:
        return NumberFormatter(format="0.000000")
    elif 'ime' in typestr:
        return DateFormatter(format = '%Y-%m-%d')
    else:
        return StringFormatter()

# Now only accept data from double type
def plot_table(df):
    # If table has too many columns for display, transpose it so that it fits
    truncated_note = None
    if df.shape[1] > ColumnThreshold:
        df = df.T

    # If too many rows, truncate to MiddleThreshold and remember note
    if df.shape[0] > MiddleThreshold:
        truncated_note = f"(showing first {MiddleThreshold} rows of {df.shape[0]})"
        df = df.iloc[:MiddleThreshold, :]

    # now safe to build table
    assert(df.shape[0] <= MiddleThreshold)
    assert(df.shape[1] <= ColumnThreshold)

    # Because bokeh cannot show index, we need to save index to first column
    index_series = pd.Series(df.index)
    index_str = 'class'
    df = df.reset_index().iloc[:,1:]
    df.insert(loc = 0, column = index_str, value = index_series)

    source = ColumnDataSource(df)
    columns = [TableColumn(field=c, title=c, 
        formatter=get_format(df, i)) for i, c in enumerate(df.columns.tolist())]
    # height calculation: at least 100
    height = max(120, (df.shape[0]+1)*25)
    table_figure = DataTable(source=source, columns=columns, fit_columns=True,
        editable=False, height = height, width = (df.shape[1]+1)*88)
    if truncated_note:
        # Prepend a small Div to indicate truncation
        return column([Div(text=truncated_note, width=1200, height=20), table_figure])
    return table_figure

# now enforce set index as datetime, x axis
# now each column one line
def plot_line_old(df):
    assert(df.shape[0] <= IndexThreshold)
    assert(df.shape[1] <= ColumnThreshold)
    x_axis_type = "datetime"
    dts = np_as_datetime(df.index)
    range_start = df.shape[0] // -5
    big = figure(height=600, width=1200,  
        tools = "", toolbar_location=None, 
        x_axis_type=x_axis_type, x_axis_location="above",
        background_fill_color="#efefef", 
        x_range=(dts[range_start], dts[-1]))
    tot = len(df.columns.tolist())
    for i, c in enumerate(df.columns.tolist()):
        big.line(dts, df[c], 
            color=Category20[max(3, tot)][i], legend_label=str(c))
    big.yaxis.axis_label = "Value"
    small = figure(height=260, width=1200, y_range=big.y_range,
            x_axis_type=x_axis_type, y_axis_type=None,
            tools = "", toolbar_location=None, 
            background_fill_color="#efefef")
    range_rool = RangeTool(x_range=big.x_range)
    range_rool.overlay.fill_color = "navy"
    range_rool.overlay.fill_alpha = 0.2
    for i, c in enumerate(df.columns.tolist()):
        small.line(dts, df[c], 
            color=Category20[max(3, tot)][i])
    small.ygrid.grid_line_color = None
    small.add_tools(range_rool)
    small.toolbar.active_multi = range_rool
    return gridplot([[big], [small]])

def plot_line(df):
    assert(df.shape[0] <= IndexThreshold)
    assert(df.shape[1] <= ColumnThreshold)
    dts = np_as_datetime(df.index)
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select"
    p2 = figure(height=500, width=1000,  
        tools=TOOLS, x_axis_type = "datetime")
    
    tot = len(df.columns.tolist())
    for i, c in enumerate(df.columns.tolist()):
        p2.line(dts, df[c], 
            line_color=Category20[max(3, tot)][i], legend_label=str(c), line_width=2)
    p2.yaxis.axis_label = "Value"
    return p2

# We can only accept index size be one
# Each column one bar
def plot_bar(df):
    assert(df.shape[0] == 1)
    assert(df.shape[1] <= ColumnThreshold)
    bar_str_list = [str(x) for x in df.columns.tolist()]
    source = ColumnDataSource(data=dict(
        bars=bar_str_list, 
        values=list(df.iloc[0,:]), 
        colors=Category20[max(3,df.shape[1])]))
    bar = figure(x_range=bar_str_list, 
        y_range=(np.min(df.iloc[0,:])-0.1, np.max(df.iloc[0,:])+0.1),
        height=350, tools="pan,wheel_zoom,box_zoom,reset,save")
    bar.vbar(x='bars', top='values', width=0.5, color='colors', 
        legend_label="bars", source=source)
    bar.xgrid.grid_line_color = None
    bar.legend.orientation = "horizontal"
    bar.legend.location = "top_center"
    return bar

# Ori: plot_figure(df, graph_type = None, df = None)
def plot(title, display_data_list = []):
    plot_list = [plot_div(title)]
    for item in display_data_list:
        display_type = item[0]
        display_df = item[1]
        if(display_type == 'table'):
            plot_list.append(plot_table(display_df))
        elif(display_type == 'line'):
            plot_list.append(plot_line(display_df))
        elif(display_type == 'bar'):
            plot_list.append(plot_bar(display_df))
        else:
            continue
    return plot_list

def save_html(save_path, print_fig_list):
    folder_path = os.path.dirname(os.path.abspath(save_path))
    if(os.path.isdir(folder_path)):
        pass
    else:
        os.makedirs(folder_path)
    whole_figure = column(print_fig_list)
    output_file(save_path)
    save(obj=whole_figure, filename=save_path, title="output")
    reset_output()

def save_xlsx(save_path, df_dict):
    folder_path = os.path.dirname(os.path.abspath(save_path))
    if(os.path.isdir(folder_path)):
        pass
    else:
        os.makedirs(folder_path)
    writer = pd.ExcelWriter(save_path, engine='xlsxwriter')   
    for sheet in df_dict:
        dataframe = df_dict[sheet]
        dataframe.to_excel(writer, sheet_name=sheet[:31], startrow=0 , startcol=0)   
    writer.save()

