from pathlib import Path
from typing import Union

import cloudpickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from .train import RNNtraining
from . import train
from .disrnn_analy import plot_update_rules

def serialize(train_data: RNNtraining, outdir: Path, name = "RNNtraining_obj"):
    """save RNNtraining instance to cloudpickle format

    Args:
        train_data (RNNtraining): _description_
        outdir (Path): output directory
        name (str, optional): _description_. Defaults to "RNNtraining_obj".
    """
    cloudpickle.register_pickle_by_value(train)
    with open(outdir/(name + ".cloudpickle"), 'wb') as handle:
        cloudpickle.dump(train_data, handle)

def unpack(fp: Union[str, Path]) -> RNNtraining:
    """unpack couldpickle-d RNNtraining instance from file"""
    with open(fp, 'rb') as f:
        res = cloudpickle.load(f)
    return res

def export_pdfpages(figs, pdf_path):
    """export multiple matplotlib figures to multi-page pdf"""
    with PdfPages(pdf_path) as pdf: 
            for f in figs:
                plt.figure(f)
                pdf.savefig()
                plt.close()

def loss_trace_plot(train_data: RNNtraining, outdir: Path):
    plt.figure()
    plt.semilogy(np.array(train_data.loss_trace), color='black')
    plt.xlabel('Training Step')
    plt.ylabel('Mean Loss')
    plt.savefig(outdir / "model_trainning_loss_trace.svg")

# def disrnn_bottlenecks_plot(train_data: RNNtraining, outdir: Path):
#     if train_data.datasets[0].input_feature_name:
#         disrnn.plot_bottlenecks(train_data.params, obs_names=train_data.datasets[0].input_feature_name)
#     plt.savefig(outdir / "bottlenecks.svg")

def disrnn_update_rules_plot(train_data: RNNtraining, outdir: Path):
    figs = plot_update_rules(train_data.params, train_data.eval_model)
    export_pdfpages(figs, outdir / 'update_rules.pdf')

# def disrnn_plotly_bottlenecks_plot(train_data: RNNtraining, outdir: Path):
#     latent_sigmas, update_sigmas = disrnn.sort_bottlenecks(train_data.params)
#     px.imshow(1 - update_sigmas, zmin=0, zmax=1, color_continuous_scale="Oranges")

