from copy import deepcopy
from typing import Optional, List, overload, Callable
from pathlib import Path
import textwrap
from ..typing import PathLike

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils import _select_func_by_1st_arg_type
from .utils import Params
from .train import RNNtraining
from .disrnn import get_bottlenecks_latent, get_bottlenecks_update, order_bottlenecks
from disRNN_MP.rnn.train_db import ModelTrainee
from disRNN_MP.analysis.disrnn_static_analy import disRNN_params_analyzer

def _plotly_text_wrap(txt:str, width = 80):
    splt = textwrap.wrap(txt, width=width)
    return '<br>'.join(splt)


def px2subplot(fig: go.Figure, p: go.Figure, row: int, col: int, secondary_y: bool = False):
    """copy a plotly.express plot to a subplot of another figure

    assume both figures are of XYcoordinate

    Args:
        fig (go.Figure): the figure that contains the subplots
        p (go.Figure): the plotly.express plot
        row (int), col (int): indicating which subplot
        secondary_y (bool, optional): whether copy the plot to 2nd y-axis of the subplot. Defaults to False.

    Returns:
        _type_: _description_
    """
    # ===================== add the trace data
    for trace in p.data:
        fig.append_trace(trace, row, col)

    # obtain the keys for x and y axis of `fig`
    xkey, ykey = fig._grid_ref[row-1][col-1][int(secondary_y)].layout_keys
    # ===================== copy x and y axis settings to the subplot

    if 'xaxis' in p._layout and 'yaxis' in p._layout:
        # get p's axes settings
        xaxis = deepcopy(p._layout['xaxis'])
        yaxis = deepcopy(p._layout['yaxis'])

        # incorporate fig's setting for axes
        xaxis.update(fig._layout[xkey])
        yaxis.update(fig._layout[ykey])

        fig.update_layout({xkey: xaxis, ykey: yaxis})
    else:
        UserWarning("'xaxis' or 'yaxis' not found in px trace")

    return fig


def remove_dup_legend(fig: go.Figure) -> go.Figure:
    """remove duplicate legend from a figure
    detection rule is whether the associated traces have same `name`

    """
    names = set()
    fig.for_each_trace(
        lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name))

    return fig


def dashboard_loss(train_data: RNNtraining):
    x = (np.arange(len(train_data.loss_trace)) + 1)
    y = np.array(train_data.loss_trace)
    return x, y

def dashboard_HMs(listOfParams: List[Params], sort_bn: bool = False, start: Optional[int] = None):
    """for plotting bottlenecks for disRNN model
    
    prepare bottlenecks for the data format required by plotly 
    """
    if start:
        # listOfParams = [listOfParams[i] for i in range(len(listOfParams)) if i >= start]
        listOfParams = listOfParams[start:]
    
    bns_latent = get_bottlenecks_latent(listOfParams)
    bns_update = get_bottlenecks_update(listOfParams)

    if sort_bn:
        latent_order, update_order = order_bottlenecks(listOfParams[-1])
        bns_latent = bns_latent[:,latent_order,:]
        bns_update = bns_update[:,latent_order,:][:,:,update_order]
    
    return list(zip(bns_latent, bns_update))

def dashboard_ll(train_data: RNNtraining):

    ll_x = np.cumsum(np.concatenate([np.repeat(ts.steps_per_block, repeats=ts.n_block) for ts in train_data.train_sessions]))
    trainll_y = np.array(train_data.train_history.train_ll)
    testll_y = np.array(train_data.train_history.test_ll)

    return ll_x, trainll_y, testll_y

def _training_progress_plot(
        fig: go.Figure, 
        ll_x: np.ndarray|List[int|float],
        trainll_y: np.ndarray|List[float],
        testll_y: np.ndarray|List[float],
        loss_x: np.ndarray|List[int|float],
        loss_y: np.ndarray|List[float],
        sess_edges: np.ndarray|List[int|float], 
        sess_annot: Optional[np.ndarray|List[str]] = None,
        custom_data = None,
        row: int = 1,
        col: int = 1
    ) -> go.Figure:
    """lower level function to generate training progress plot
    
    showing training, testing likelihood plot together with loss function

    the background color is shaded to indicate different training sessions
    """

    #====== likelihood and loss function traces in 1st subplot

    fig.add_scatter(name="training likelihood", 
        x=ll_x, y=trainll_y, 
        customdata=custom_data, 
        mode="lines+markers", text=sess_annot, secondary_y=False, 
        row=row, col=col)

    fig.add_scatter(name="testing likelihood", 
        x=ll_x, y=testll_y, 
        mode="lines+markers", secondary_y=False, row=row, col=col)

    fig.add_scatter(name="loss function", 
        x=loss_x,
        y=loss_y, 
        line={'color': '#9467bd'},
        mode = "lines", secondary_y=True, row=row, col=col)

    #====== add shaded background
    for i in range(len(sess_edges)-1):
        color = 'yellow' if i % 2 == 1 else 'green'
        fig.add_vrect(
            x0=sess_edges[i], x1=sess_edges[i+1], 
            fillcolor=color, line_width=0, opacity=0.25, layer="below", 
            row=row, col=col, # type: ignore
        )

    #====== change figure style

    loss_yaxis = next(fig.select_traces({'name': "loss function"})).yaxis[1:]
    ll_yaxis = next(fig.select_traces({'name': "training likelihood"})).yaxis[1:]

    fig.update_layout(
        {'yaxis'+loss_yaxis: {'showgrid': False, 'title': "loss function"},
        'yaxis'+ll_yaxis: {'title': "likelihood"}},
        width=800, height=600,
        hovermode = "x unified"
    )

    #

    ind = np.argmax(testll_y)
    fig.add_hline(testll_y[ind], line_dash='dot', 
        annotation_text = "{:.6f}".format(testll_y[ind]),
        annotation_position = "top left",
        annotation = {'font_color': "darkred"}
        )

    return fig

def _disrnn_bottlenecks_plot(
        fig: go.Figure,
        bns_latent,
        bns_update,
        bns_choice = None,
        in_feat_name = None,
        coloraxis_name = 'coloraxis1',
        latent_plt_coord = (2,1),
        update_plt_coord = (2,2),
        choice_plt_coord = (2,3),
    ) -> go.Figure:
    """add bottlenecks plot to the figure

    Args:
        fig (go.Figure): _description_
        bns_latent (_type_): _description_
        bns_update (_type_): _description_
        in_feat_name (_type_, optional): _description_. Defaults to None.
        coloraxis_name (str, optional): _description_. Defaults to 'coloraxis1'.
        latent_plt_coord: the row and col of latent bottleneck subplot
        update_plt_coord: the row and col of update bottleneck subplot

    Returns:
        go.Figure: modified figure
    """

    y_lab = ["l"+str(i+1) for i in range(bns_latent.shape[0])]
    if in_feat_name is not None: # hasattr(train_data.datasets[0], 'input_feature_name') and train_data.datasets[0].input_feature_name is not None:
        x_lab = [*in_feat_name, *y_lab]
    else:
        x_lab = [*["in"+str(i+1) for i in range(bns_update.shape[1] - bns_update.shape[0])], *y_lab]

    #====== latent heatmap plot

    fig.add_trace(go.Heatmap(
        name = "latent bottleneck",
        z=bns_latent, 
        x=[""], 
        y=y_lab,
        coloraxis=coloraxis_name, zmin=0, zmax=1,), 
    row=latent_plt_coord[0],col=latent_plt_coord[1])

    #====== choice heatmap plot
    if bns_choice is not None:
        fig.add_trace(go.Heatmap(
            name = "choice bottleneck",
            z=bns_choice, 
            x=[""], 
            y=y_lab,
            coloraxis=coloraxis_name, zmin=0, zmax=1,), 
        row=choice_plt_coord[0],col=choice_plt_coord[1])

    #====== update heatmap plot

    

    fig.add_trace(go.Heatmap(
        name = "update bottleneck",
        z=bns_update, 
        x=x_lab, y = y_lab,
        coloraxis=coloraxis_name, zmin=0, zmax=1), 
    row=update_plt_coord[0],col=update_plt_coord[1])

    updHM_yaxis = next(fig.select_traces({'name': 'update bottleneck'})).yaxis[1:]

    #====== define coloraxis shared by heatmaps

    # match the colorbar to update heatmap height
    cb_bottom, cb_top = fig.layout['yaxis' + updHM_yaxis]['domain'] # type: ignore
    fig.update_layout({
        coloraxis_name: dict(
            colorscale = 'Oranges', cmin = 0, cmax = 1, 
            colorbar = dict(yanchor = 'bottom', y = cb_bottom, len = cb_top, ypad = 0)
    )})

    return fig

def _export_disrnn_dashboard_html(
        fig: go.Figure, 
        outdir: Path = Path('.'), 
        name: str = 'dashboard.html',
        customData_container: str | None = None,
    ) -> None: 

    # all possible cases
    z_indexs = {
        "list": {
            True: "[pointData.customdata[0], pointData.customdata[1], pointData.customdata[2]]",
            False: "[pointData.customdata[0], pointData.customdata[1]]",
        },
        "dict": {
            True: "[pointData.customdata['latent'], pointData.customdata['update'], pointData.customdata['choice']]",
            False: "[pointData.customdata['latent'], pointData.customdata['update']]",
        }
    }

    if len(list(fig.select_traces(selector={'name' : "choice bottleneck"}))) > 0:
        include_choice_bottleneck = True
    else:
        include_choice_bottleneck = False

    if isinstance(customData_container, str):
        match customData_container.lower():
            case "list":
                z_index = z_indexs['list'][include_choice_bottleneck]
            case "dictionary"|"dict":
                z_index = z_indexs['dict'][include_choice_bottleneck]
            case _:
                raise ValueError(f"the indicated `customData_container` ({customData_container}) is not supported. Supported are: 'list', 'dictionary'('dict')")
    else:
        custom_data_sample = next(fig.select_traces(selector={'name' : "training likelihood"}))['customdata'][0]
        if isinstance(custom_data_sample, dict):
            z_index = z_indexs['dict'][include_choice_bottleneck]
        elif isinstance(custom_data_sample, list):
            z_index = z_indexs['list'][include_choice_bottleneck]
        else:
            raise ValueError(f"the custom data type '{type(custom_data_sample)}' is not supported")

    # assume the first customdata is latent HM, second is update HM
    js_callback = """

    // var plot = document.getElementsByClassName("plotly-graph-div js-plotly-plot")[0]
    var plot = document.getElementById('{plot_id}')

    console.log(plot.id)
    console.log(plot.data)

    var TID_trainll = plot.data.findIndex(d => d.name == 'training likelihood')
    var TID_latentHM = plot.data.findIndex(d => d.name == 'latent bottleneck')
    var TID_updateHM = plot.data.findIndex(d => d.name == 'update bottleneck')
    var TID_choiceHM = plot.data.findIndex(d => d.name == 'choice bottleneck')

    plot.on('plotly_hover', function(data) {

        var pointData = data.points.find(p => p.curveNumber == TID_trainll);
        if (pointData) {
            let patch = {
                z: {z_indexing}
            }
            Plotly.restyle(plot, patch, [TID_latentHM, TID_updateHM, TID_choiceHM])
        }
    })

    """.replace("{z_indexing}", z_index)

    fig.write_html(outdir / name, post_script = js_callback)

def _mt_disrnn_dashboard_figure(trainee:ModelTrainee) -> go.Figure:
    """create disRNN dashboard from ModelTrainee

    Args:
        trainee (ModelTrainee): the data

    Returns:
        go.Figure: the created figure
    """

    recs = trainee.records
    analyzer = disRNN_params_analyzer.from_modelTrainee(trainee)

    if analyzer.has_separate_choice_bottleneck:
        funcs = {
            "latent": lambda p: analyzer.get_latent_sigma(p).reshape(-1, 1), 
            "update": lambda p: np.transpose(analyzer.get_update_sigma(p)),
            "choice": lambda p: analyzer.get_choice_sigma(p).reshape(-1, 1), 
        }
    else:
        funcs = {
            "latent": lambda p: analyzer.get_latent_sigma(p).reshape(-1, 1), 
            "update": lambda p: np.transpose(analyzer.get_update_sigma(p)),
        }

    custom_data = [{key: 1 - f(rec.parameter) for key, f in funcs.items()} for rec in recs]

    try:
        trainee.materialize()
        in_feat_name = trainee.sessions[0].train_dataset.input_feature_name
    except Exception as e:
        print(f'cannot read input feature name from dataset due to error: {e}')
        in_feat_name = None

    if analyzer.has_separate_choice_bottleneck:
        fig = make_subplots(
            rows=2, cols=3,
            specs=[[{"colspan": 3, 'secondary_y': True}, None, None], [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}]],
            column_widths=[0.2, 0.6, 0.2],
            subplot_titles=("likelihood","latent bottleneck", "update bottleneck", "choice bottleneck"))

        _training_progress_plot(
            fig = fig,
            ll_x = np.array([rec.step for rec in recs]),
            trainll_y = np.array([rec.train_metric for rec in recs]),
            testll_y = np.array([rec.test_metric for rec in recs]),
            sess_annot = (np.repeat(
                    [ts.name for ts in trainee.sessions], 
                [len(ts.records) for ts in trainee.sessions])  
                if hasattr(trainee.sessions[0], 'name') 
                else None
            ),
            loss_y = np.array([lt.value for lt in trainee.loss_trace]),
            loss_x = np.array([lt.step for lt in trainee.loss_trace]),
            sess_edges = np.cumsum([1, *[se.n_step for se in trainee.sessions]]),
            custom_data = custom_data,
        )

        _disrnn_bottlenecks_plot(
            fig = fig,
            bns_latent = custom_data[-1]['latent'],
            bns_update = custom_data[-1]['update'],
            bns_choice = custom_data[-1]['choice'],
            in_feat_name = in_feat_name,
        )
    else:
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"colspan": 2, 'secondary_y': True}, None], [{'type': 'heatmap'}, {'type': 'heatmap'}]],
            column_widths=[0.2, 0.8],
            subplot_titles=("likelihood","latent bottleneck", "update bottleneck"))

        _training_progress_plot(
            fig = fig,
            ll_x = np.array([rec.step for rec in recs]),
            trainll_y = np.array([rec.train_metric for rec in recs]),
            testll_y = np.array([rec.test_metric for rec in recs]),
            sess_annot = (np.repeat(
                    [ts.name for ts in trainee.sessions], 
                [len(ts.records) for ts in trainee.sessions])  
                if hasattr(trainee.sessions[0], 'name') 
                else None
            ),
            loss_y = np.array([lt.value for lt in trainee.loss_trace]),
            loss_x = np.array([lt.step for lt in trainee.loss_trace]),
            sess_edges = np.cumsum([1, *[se.n_step for se in trainee.sessions]]),
            custom_data = custom_data,
        )

        _disrnn_bottlenecks_plot(
            fig = fig,
            bns_latent = custom_data[-1]['latent'],
            bns_update = custom_data[-1]['update'],
            in_feat_name = in_feat_name,
        )

    fig.update_layout(title_text = 
        _plotly_text_wrap(trainee.description, 80) if trainee.description else None)

    return fig

def _rt_disrnn_dashboard_figure(train_data: RNNtraining, sort_bn: bool = False):

    # latent_sigmas, update_sigmas = disrnn.sort_bottlenecks(train_data.params) # init value

    ll_x, trainll_y, testll_y = dashboard_ll(train_data)
    customdata = dashboard_HMs(train_data.train_history.params, sort_bn)
    loss_x, loss_y = dashboard_loss(train_data)
    curr_bns_latent, curr_bns_update = customdata[-1]

    sess_block_schedule = [ts.n_block for ts in train_data.train_sessions]

    # sess_edges = np.concatenate(([1], ll_x[np.cumsum(sess_block_schedule)-1]))
    # the following method will reflect the actual session blocks (when a session was terminated before it is finished)
    sess_edges = np.concatenate(([1], ll_x[[ts.block_ids[-1] for ts in train_data.train_sessions]]))

    # sess_names = [ts.name for ts in train_data.train_sessions] if hasattr(train_data.train_sessions[0], 'name') else None

    texts = np.repeat([ts.name for ts in train_data.train_sessions], sess_block_schedule) if hasattr(train_data.train_sessions[0], 'name') else None


    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2, 'secondary_y': True}, None], [{'type': 'heatmap'}, {'type': 'heatmap'}]],
        column_widths=[0.2, 0.8],
        subplot_titles=("likelihood","latent bottleneck", "update bottleneck"))

    #====== likelihood and loss function traces in 1st subplot

    fig.add_scatter(name="training likelihood", 
        x=ll_x, y=trainll_y, 
        customdata=customdata, # to help with updating the heatmap
        mode="lines+markers", text=texts, secondary_y=False, row=1,col=1)

    fig.add_scatter(name="testing likelihood", 
        x=ll_x, y=testll_y, 
        mode="lines+markers", secondary_y=False, row=1,col=1)

    fig.add_scatter(name="loss function", 
        x=loss_x,
        y=loss_y, 
        mode = "lines", secondary_y=True, row=1,col=1)

    #====== add shaded background
    for i in range(len(sess_edges)-1):
        color = 'yellow' if i % 2 == 1 else 'green'
        fig.add_vrect(
            x0=sess_edges[i], x1=sess_edges[i+1], 
            fillcolor=color, line_width=0, opacity=0.25, layer="below",
            row=1, col=1, ) # type: ignore

    #====== latent heatmap plot

    fig.add_trace(go.Heatmap(
            name = "latent bottleneck",
            z=curr_bns_latent, 
            x=[""], 
            coloraxis="coloraxis1", zmin=0, zmax=1,), 
        row=2,col=1)

    #====== update heatmap plot

    y_lab = ["l"+str(i+1) for i in range(curr_bns_latent.shape[0])]
    if hasattr(train_data.datasets[0], 'input_feature_name') and train_data.datasets[0].input_feature_name is not None:
        x_lab = [*train_data.datasets[0].input_feature_name, *y_lab]
    else:
        x_lab = [*["in"+str(i+1) for i in range(curr_bns_update.shape[1] - curr_bns_update.shape[0])], *y_lab]

    fig.add_trace(go.Heatmap(
            name = "update bottleneck",
            z=curr_bns_update, 
            x=x_lab, y = y_lab,
            coloraxis="coloraxis1", zmin=0, zmax=1), 
        row=2,col=2)

    updHM_yaxis = next(fig.select_traces({'name': 'update bottleneck'})).yaxis[1:]

    #====== define coloraxis shared by heatmaps

    # match the colorbar to update heatmap height
    cb_bottom, cb_top = fig.layout['yaxis' + updHM_yaxis]['domain'] # type: ignore
    fig.update_layout(coloraxis1=dict(
        colorscale = 'Oranges', cmin = 0, cmax = 1, 
        colorbar = dict(yanchor = 'bottom', y = cb_bottom, len = cb_top, ypad = 0)
    ))

    #====== change figure style

    loss_yaxis = next(fig.select_traces({'name': "loss function"})).yaxis[1:]
    ll_yaxis = next(fig.select_traces({'name': "training likelihood"})).yaxis[1:]

    fig.update_layout(
        {'yaxis'+loss_yaxis: {'showgrid': False, 'title': "loss function"},
        'yaxis'+ll_yaxis: {'title': "likelihood"}},
        width=1200, height=800,
        hovermode = "x unified"
    )

    return fig

def _mt_training_progress_plot(trainee:ModelTrainee) -> go.Figure:
    recs = trainee.records

    fig = _training_progress_plot(
        fig = make_subplots(specs=[[{"secondary_y": True}]]),
        ll_x = np.array([rec.step for rec in recs]),
        trainll_y = np.array([rec.train_metric for rec in recs]),
        testll_y = np.array([rec.test_metric for rec in recs]),
        sess_annot = (np.repeat(
                [ts.name for ts in trainee.sessions], 
            [len(ts.records) for ts in trainee.sessions])  
            if hasattr(trainee.sessions[0], 'name') 
            else None
        ),
        loss_y = np.array([lt.value for lt in trainee.loss_trace]),
        loss_x = np.array([lt.step for lt in trainee.loss_trace]),
        sess_edges = np.cumsum([1, *[se.n_step for se in trainee.sessions]]),
        )
    
    fig.update_layout(title_text = 
        _plotly_text_wrap(trainee.description, 80) if trainee.description else None)

    return fig

def _rt_training_progress_plot(train_data: RNNtraining, outdir: Optional[PathLike] = None):
    """visualize the training progress that is recorded in a RNNtraining instance
    will plot training, testing likelihood and loss function
    Args:
        train_data: the RNNtraining instance
        outdir: [optional] if provided, write a html and a svg figure to the directory
    """
    
    ll_x, trainll_y, testll_y = dashboard_ll(train_data)
    loss_x, loss_y = dashboard_loss(train_data)

    sess_block_schedule = [ts.n_block for ts in train_data.train_sessions]

    texts = (np.repeat([
            ts.name for ts in train_data.train_sessions], # type: ignore
        sess_block_schedule)  # type: ignore
        if hasattr(train_data.train_sessions[0], 'name') 
        else None
    )

    # the following method will reflect the actual session blocks (when a session was terminated before it is finished)
    sess_edges = np.concatenate(([1], ll_x[[ts.block_ids[-1] for ts in train_data.train_sessions]]))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    #====== likelihood and loss function traces in 1st subplot

    fig.add_scatter(name="training likelihood", 
        x=ll_x, y=trainll_y, 
        mode="lines+markers", text=texts, secondary_y=False)

    fig.add_scatter(name="testing likelihood", 
        x=ll_x, y=testll_y, 
        mode="lines+markers", secondary_y=False)

    fig.add_scatter(name="loss function", 
        x=loss_x,
        y=loss_y, 
        line={'color': '#9467bd'},
        mode = "lines", secondary_y=True)

    #====== add shaded background
    for i in range(len(sess_edges)-1):
        color = 'yellow' if i % 2 == 1 else 'green'
        fig.add_vrect(
            x0=sess_edges[i], x1=sess_edges[i+1], 
            fillcolor=color, line_width=0, opacity=0.25, layer="below",
        )

    #====== change figure style

    loss_yaxis = next(fig.select_traces({'name': "loss function"})).yaxis[1:]
    ll_yaxis = next(fig.select_traces({'name': "training likelihood"})).yaxis[1:]

    fig.update_layout(
        {'yaxis'+loss_yaxis: {'showgrid': False, 'title': "loss function"},
        'yaxis'+ll_yaxis: {'title': "likelihood"}},
        width=800, height=600,
        hovermode = "x unified"
    )

    if outdir is not None:
        fig.write_html(Path(outdir) / 'training_progress.html')
        fig.write_image(Path(outdir) / 'training_progress.svg')
    
    return fig

def _rt_disrnn_dashboard_html(train_data: RNNtraining, outdir: Path, sort_bn: bool = False): 

    fig = disrnn_dashboard_figure(train_data, sort_bn = sort_bn)
    _export_disrnn_dashboard_html(fig=fig, outdir=outdir, customData_container='list')

def _mt_disrnn_dashboard_html(
        trainee:ModelTrainee, 
        outdir: Path, 
        name: str = 'dashboard.html', 
    ) -> None: 

    fig = disrnn_dashboard_figure(trainee)
    _export_disrnn_dashboard_html(fig=fig, outdir=outdir, name = name, customData_container="dict")



# =========== public interface

@overload
def training_progress_plot(trainee:ModelTrainee) -> go.Figure:
    ...

@overload
def training_progress_plot(train_data: RNNtraining, outdir: Optional[PathLike] = None) -> go.Figure:
    ...

def training_progress_plot(*args, **kwargs) -> go.Figure:
    return _select_func_by_1st_arg_type(
        {
            ModelTrainee: _mt_training_progress_plot,
            RNNtraining: _rt_training_progress_plot,
        },
        args, kwargs
    )

@overload
def disrnn_dashboard_figure(trainee:ModelTrainee) -> go.Figure:
    """generate disRNN dashboard figure from ModelTrainee instance"""
    ...

@overload
def disrnn_dashboard_figure(train_data: RNNtraining, sort_bn: bool = False) -> go.Figure:
    """generate disRNN dashboard figure from RNNtraining instance"""
    ...

def disrnn_dashboard_figure(*args, **kwargs) -> go.Figure:

    return _select_func_by_1st_arg_type(
        {
            ModelTrainee: _mt_disrnn_dashboard_figure,
            RNNtraining: _rt_disrnn_dashboard_figure,
        },
        args, kwargs
    )

@overload
def disrnn_dashboard_html(trainee:ModelTrainee, outdir: Path, name: str = 'dashboard.html', customData_container: str = "list",) -> None:
    ...

@overload
def disrnn_dashboard_html(train_data: RNNtraining, outdir: Path, sort_bn: bool = False) -> None:
    ...

def disrnn_dashboard_html(*args, **kwargs) -> None: 

    return _select_func_by_1st_arg_type(
        {
            ModelTrainee: _mt_disrnn_dashboard_html,
            RNNtraining: _rt_disrnn_dashboard_html,
        },
        args, kwargs
    )


