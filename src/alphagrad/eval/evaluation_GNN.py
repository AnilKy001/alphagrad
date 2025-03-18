import matplotlib
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jrand

import pandas as pd
import numpy as np

import os
from typing import Callable, Sequence, Union
from chex import Array
from tqdm import tqdm
import time
import timeit

from graphax.perf import plot_performance_with_GNN, plot_performance_over_size_jax_GNN, measure
from graphax.core import jacve
from graphax.examples import (
    RoeFlux_1d,
    RobotArm_6DOF,
    HumanHeartDipole,
    PropaneCombustion,
    BlackScholes,
    RoeFlux_3d,
    Encoder
)

Order = Union[str, Sequence[int]]

experiments = [
    RoeFlux_1d
]

font = {"family": "sans-serif",
        "weight": "normal",
        "size": 18}

line_width = 1

def measure_exec_time(f: Callable, args: Sequence[Array], elim_order: Order, num_samples: int, argnums: Sequence[int]):
    grad_f = jax.jit(jacve(
        fun=f,
        order=elim_order,
        argnums=argnums
    ))

    def sample_time(xs, number=10, repeat=10):
        # Using timeit to measure execution time
        timer = timeit.Timer(
            lambda: jax.block_until_ready(grad_f(*xs))
        )
        # Run multiple timing measurements and take the best (minimum) time
        execution_times = timer.repeat(repeat=repeat, number=number)
        
        return min(execution_times[1:]) / (number-1)
    
    measurements = [sample_time(args)*1000 for i in tqdm(range(num_samples))] # conversion to ms

    measurements = jnp.array(measurements)[10:]
    median = jnp.median(measurements)
    return jnp.array(measurements), median


def exp_RoeFlux_1d(num_samples: int = 1000):

    # Minimal Markowitz elimination order:
    order_mM = [4, 5, 8, 9, 16, 17, 25, 27, 31, 33, 38, 43, 44, 45, 69, 84, 1, 2,
            10, 13, 18, 21, 26, 28, 32, 34, 37, 39, 42, 47, 50, 53, 57, 59, 
            62, 64, 66, 67, 68, 71, 73, 75, 76, 77, 80, 81, 83, 85, 86, 87, 
            91, 92, 95, 11, 14, 19, 22, 51, 54, 58, 60, 63, 65, 72, 79, 88, 
            90, 93, 96, 3, 6, 7, 15, 29, 40, 56, 61, 74, 78, 82, 48, 89, 94, 
            23, 35, 46, 24, 70, 41, 98, 100, 12, 20, 30, 49, 52, 55, 36]

    # Alphagrad with transformers in PPO network elimination order:
    order_trans = [8, 82, 27, 66, 7, 78, 76, 13, 48, 42, 68, 86, 95, 4, 59, 28, 77, 54, 1, 
         94, 5, 58, 72, 93, 75, 31, 53, 33, 57, 90, 44, 25, 89, 88, 84, 96, 74, 
         92, 83, 91, 45, 51, 81, 80, 11, 10, 85, 43, 22, 73, 19, 71, 6, 18, 17, 
         79, 47, 50, 52, 21, 37, 38, 55, 49, 69, 35, 65, 29, 64, 16, 9, 60, 15, 
         61, 23, 87, 70, 67, 24, 46, 63, 39, 2, 62, 3, 41, 40, 32, 26, 34, 56, 
         30, 14, 98, 36, 12, 20, 100]

    # Alphagrad with GNN in PPO network elimination order:
    order_GNN = [
        96, 9, 54, 39, 38, 43, 95, 8, 84, 88, 11, 17, 79, 69, 72, 33, 94, 4, 2, 18, 67, 5, 16, 31, 
        27, 63, 48, 83, 71, 68, 58, 81, 86, 82, 78, 75, 44, 42, 45, 25, 87, 77, 80, 53, 65, 57, 61, 
        10, 62, 91, 60, 90, 76, 73, 50, 85, 37, 59, 93, 19, 56, 92, 35, 29, 28, 64, 6, 51, 1, 26, 
        34, 47, 32, 14, 22, 100, 21, 13, 40, 66, 89, 98, 70, 74, 52, 55, 46, 24, 3, 41, 49, 23, 36, 
        30, 7, 12, 20, 15
    ]

    shape = (1024,)
    key = jrand.PRNGKey(1234)
    xs = [.01, .02, .02, .01, .03, .03]
    xs = [jrand.uniform(key, shape)*x for x in xs]
    xs = jax.device_put(xs, jax.devices("cpu")[0])
    argnums = list(range(len(xs)))

    meas_mM, med_mM = measure_exec_time(RoeFlux_1d, xs, order_mM, num_samples=num_samples, argnums=argnums)
    meas_trans, med_trans = measure_exec_time(RoeFlux_1d, xs, order_trans, num_samples=num_samples, argnums=argnums)
    meas_GNN, med_GNN = measure_exec_time(RoeFlux_1d, xs, order_GNN, num_samples=num_samples, argnums=argnums)

    plt.style.use("seaborn-v0_8-colorblind")
    matplotlib.rc("font", **font)
    matplotlib.rcParams['axes.linewidth'] = 1.5

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    ax.set_title("RoeFlux_1d Execution Time Comparison")
    x_pos = jnp.arange(num_samples-10)

    ax.plot(x_pos, meas_mM, markersize=14, label="minimalMarkowitz", linewidth=line_width)
    ax.plot(x_pos, meas_trans, markersize=14, label="PPO_Transformer", linewidth=line_width)
    ax.plot(x_pos, meas_GNN, markersize=14, label="PPO_GNN", linewidth=line_width)
    #ax.plot(t, rtrl, "-*", markersize=14, label="RTRL peak memory usage", linewidth=line_width)

    ax.set_xlabel("Sample number")
    ax.set_ylabel("Execution time [ms]")
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.grid(True)
    ax.tick_params(direction="in", which="both", width=2.)
    ax.legend()

    plt.tight_layout()
    plt.savefig("exec_times_RoeFlux_1d.png")

    return 

def exp_RobotArm_6DOF():

    # Minimal Markowitz elimination order:
    order_mM = []

    # Alphagrad with transformers in PPO network elimination order:
    order_trans = []

    # Alphagrad with GNN in PPO network elimination order:
    order_GNN = []
    return

def exp_HumanHeartDipole():

    # Minimal Markowitz elimination order:
    order_mM = []

    # Alphagrad with transformers in PPO network elimination order:
    order_trans = []

    # Alphagrad with GNN in PPO network elimination order:
    order_GNN = []
    return

def exp_PropaneCombustion():

    # Minimal Markowitz elimination order:
    order_mM = []

    # Alphagrad with transformers in PPO network elimination order:
    order_trans = []

    # Alphagrad with GNN in PPO network elimination order:
    order_GNN = []
    return

def exp_BlackScholes():

    # Minimal Markowitz elimination order:
    order_mM = []

    # Alphagrad with transformers in PPO network elimination order:
    order_trans = []

    # Alphagrad with GNN in PPO network elimination order:
    order_GNN = []
    return

def exp_Encoder():

    # Minimal Markowitz elimination order:
    order_mM = []

    # Alphagrad with transformers in PPO network elimination order:
    order_trans = []

    # Alphagrad with GNN in PPO network elimination order:
    order_GNN = []
    return

def exp_RoeFlux_3d(num_samples: int = 1000):

    # Minimal Markowitz elimination order:
    order_mM = [81, 82, 90, 91, 4, 5, 9, 10, 12, 14, 21, 22, 23, 24, 30, 31, 32, 33, 
            39, 41, 45, 47, 48, 54, 55, 59, 75, 78, 80, 84, 89, 93, 126, 129, 
            132, 133, 3, 6, 25, 28, 34, 37, 40, 42, 49, 56, 58, 62, 64, 65, 67, 
            71, 74, 77, 85, 87, 94, 96, 98, 99, 100, 101, 102, 104, 105, 107, 
            109, 110, 112, 114, 116, 117, 119, 121, 123, 125, 127, 130, 134, 
            136, 138, 26, 29, 35, 38, 46, 57, 61, 68, 72, 83, 88, 92, 97, 111, 
            118, 124, 131, 137, 141, 11, 19, 20, 50, 53, 60, 70, 103, 106, 113, 
            120, 135, 86, 95, 13, 15, 43, 52, 66, 128, 145, 69, 73, 18, 139, 27, 
            36, 140, 108, 122, 115, 1, 51, 76, 79, 63, 2, 44, 16, 7, 8, 143, 17] 

    # Alphagrad with transformers in PPO network elimination order:
    order_trans = [124, 136, 56, 128, 78, 24, 1, 54, 101, 127, 121, 140, 47, 135, 67, 34, 
         111, 32, 100, 119, 99, 114, 125, 141, 122, 45, 65, 59, 117, 89, 116, 
         60, 42, 28, 74, 85, 11, 53, 36, 30, 108, 113, 55, 109, 129, 64, 91, 
         14, 133, 5, 10, 132, 87, 139, 110, 12, 131, 72, 8, 61, 88, 107, 6, 29, 
         57, 96, 118, 105, 71, 77, 112, 66, 75, 84, 143, 123, 90, 94, 137, 104, 
         69, 23, 22, 62, 58, 50, 130, 31, 106, 39, 48, 49, 98, 134, 93, 138, 
         126, 68, 115, 80, 102, 92, 79, 52, 16, 120, 95, 76, 19, 25, 73, 21, 70, 
         38, 35, 20, 86, 41, 4, 103, 43, 27, 3, 40, 9, 83, 13, 18, 37, 51, 46, 
         7, 81, 97, 63, 44, 2, 33, 82, 26, 15, 17, 145] 

    # Alphagrad with GNN in PPO network elimination order:
    order_GNN = []

    shape = (512,)
    key = jrand.PRNGKey(1234)
    xs = [.01, .02, .02, .01, .03, .03]
    xs = [jrand.uniform(key, shape)*x for x in xs]
    xs = jax.device_put(xs, jax.devices("cpu")[0])
    argnums = list(range(len(xs)))

    meas_mM, med_mM = measure_exec_time(RoeFlux_1d, xs, order_mM, num_samples=num_samples, argnums=argnums)
    meas_trans, med_trans = measure_exec_time(RoeFlux_1d, xs, order_trans, num_samples=num_samples, argnums=argnums)
    meas_GNN, med_GNN = measure_exec_time(RoeFlux_1d, xs, order_GNN, num_samples=num_samples, argnums=argnums)

    plt.style.use("seaborn-v0_8-colorblind")
    matplotlib.rc("font", **font)
    matplotlib.rcParams['axes.linewidth'] = 1.5

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    ax.set_title("RoeFlux_3d Execution Time Comparison")
    x_pos = jnp.arange(num_samples-10)

    ax.plot(x_pos, meas_mM, markersize=14, label="minimalMarkowitz", linewidth=line_width)
    ax.plot(x_pos, meas_trans, markersize=14, label="PPO_Transformer", linewidth=line_width)
    ax.plot(x_pos, meas_GNN, markersize=14, label="PPO_GNN", linewidth=line_width)
    #ax.plot(t, rtrl, "-*", markersize=14, label="RTRL peak memory usage", linewidth=line_width)

    ax.set_xlabel("Sample number")
    ax.set_ylabel("Execution time [ms]")
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.grid(True)
    ax.tick_params(direction="in", which="both", width=2.)
    ax.legend()

    plt.tight_layout()
    plt.savefig("exec_times_RoeFlux_3d.png")

    return

"""
line_width = 5.

df_eprop_t = pd.read_csv("./eprop_alif_hd_128.csv", sep=";")
df_bptt_t = pd.read_csv("./bptt_alif_hd_128.csv", sep=";")
df_naive_eprop_t = pd.read_csv("./stupid_eprop_alif_hd_128.csv", sep=";")
df_rtrl_t = pd.read_csv("./rtrl_alif_hd_128.csv", sep=";")

df_eprop_h = pd.read_csv("./eprop_alif_ts_1000.csv", sep=",")
df_bptt_h = pd.read_csv("./bptt_alif_ts_1000.csv", sep=",")
df_naive_eprop_h = pd.read_csv("./stupid_eprop_alif_ts_1000.csv", sep=",")
df_rtrl_h = pd.read_csv("./rtrl_alif_ts_1000.csv", sep=",")

t = df_eprop_t["num_timesteps"]
h = df_eprop_h["num_hidden"]

batch_size = 16
# NOTE: Adjust this to the needs of you measured data!
overhead_t = (batch_size*t*140*8 + batch_size*(20)*8) / (1024*1024*1024)
# overhead_t = 0
overhead_h = (batch_size*700*140*8 + batch_size*(20)*8) / (1024*1024*1024)

bptt_mem_t = df_bptt_t["peak_mem_usage_gb"] - overhead_t
eprop_mem_t = df_eprop_t["peak_mem_usage_gb"] - overhead_t
naive_eprop_mem_t = df_naive_eprop_t["peak_mem_usage_gb"] - overhead_t
rtrl_mem_t = df_rtrl_t["peak_mem_usage_gb"] - overhead_t

bptt_mem_h = df_bptt_h["peak_mem_usage_gb"] - overhead_h
eprop_mem_h = df_eprop_h["peak_mem_usage_gb"] - overhead_h
naive_eprop_mem_h = df_naive_eprop_h["peak_mem_usage_gb"] - overhead_h
rtrl_mem_h = df_rtrl_h["peak_mem_usage_gb"] - overhead_h


bptt_time_t = df_bptt_t["avg_batch_time_ms"] / t
eprop_time_t = df_eprop_t["avg_batch_time_ms"] / t
naive_eprop_time_t = df_naive_eprop_t["avg_batch_time_ms"] / t
rtrl_time_t = df_rtrl_t["avg_batch_time_ms"] / t


bptt_time_h = df_bptt_h["avg_batch_time_ms"] / 1000
eprop_time_h = df_eprop_h["avg_batch_time_ms"] / 1000
naive_eprop_time_h = df_naive_eprop_h["avg_batch_time_ms"] / 1000
rtrl_time_h = df_rtrl_h["avg_batch_time_ms"] / 1000

font = {"family": "sans-serif",
        "weight": "normal",
        "size": 18}
"""
def make_memory_plot_t(t, bptt, eprop, naive_eprop, rtrl):
    plt.style.use("seaborn-v0_8-colorblind")
    matplotlib.rc("font", **font)
    matplotlib.rcParams['axes.linewidth'] = 1.5

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    #ax.set_title("Peak Memory Usage Comparison")

    ax.plot(t, eprop, "-ro", markersize=14, label="Ours", linewidth=line_width)
    ax.plot(t, naive_eprop, "-gs", markersize=14, label="Naïve e-prop", linewidth=line_width)
    ax.plot(t, bptt, "-c^", markersize=14, label="BPTT", linewidth=line_width)
    #ax.plot(t, rtrl, "-*", markersize=14, label="RTRL peak memory usage", linewidth=line_width)

    ax.set_xlabel("Number of time-steps")
    ax.set_ylabel("Peak memory usage [GB]")
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.grid(True)
    ax.tick_params(direction="in", which="both", width=2.)
    ax.legend()

    plt.tight_layout()
    plt.savefig("memory_timestep.png")


def make_time_plot_t(t, bptt, eprop, naive_eprop, rtrl):
    plt.style.use("seaborn-v0_8-colorblind")
    matplotlib.rc("font", **font)
    matplotlib.rcParams['axes.linewidth'] = 1.5

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    #ax.set_title("Evaluation time for single step")

    ax.plot(t, eprop, "-ro", label="Ours", linewidth=line_width, markersize=14)
    ax.plot(t, naive_eprop, "-gs", label="Naïve e-prop", linewidth=line_width, markersize=14)
    ax.plot(t, bptt, "-c^", label="BPTT", linewidth=line_width, markersize=14)
    ax.plot(t, rtrl, "-m*", label="RTRL", linewidth=line_width, markersize=14)

    ax.set_yscale("log")
    ax.set_xlabel("Number of time-steps")
    ax.set_ylabel("Time per step [ms]")
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.grid(True)
    ax.tick_params(direction="in", which="both", width=2.7)
    ax.legend()

    plt.tight_layout()
    plt.savefig("time_timestep.png")

def make_memory_plot_h(h, bptt, eprop, naive_eprop, rtrl):
    plt.style.use("seaborn-v0_8-colorblind")
    matplotlib.rc("font", **font)
    matplotlib.rcParams['axes.linewidth'] = 1.5

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    #ax.set_title("Peak Memory Usage Comparison")

    ax.plot(h, eprop, "-ro", markersize=14, label="Ours", linewidth=line_width)
    ax.plot(h, naive_eprop, "-gs", markersize=14, label="Naïve e-prop", linewidth=line_width)
    ax.plot(h, bptt, "-c^", markersize=14, label="BPTT", linewidth=line_width)
    ax.plot(h, rtrl, "-m*", markersize=14, label="RTRL", linewidth=line_width)

    #ax.set_yscale("log")
    ax.set_ylim(bottom=.0, top=2.)
    ax.set_xlabel("Number of hidden neurons")
    ax.set_ylabel("Peak memory usage [GB]")
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.grid(True)
    ax.tick_params(direction="in", which="both", width=2.)
    ax.legend()

    plt.tight_layout()
    plt.savefig("memory_hidden.png")


def make_time_plot_h(h, bptt, eprop, naive_eprop, rtrl):
    plt.style.use("seaborn-v0_8-colorblind")
    matplotlib.rc("font", **font)
    matplotlib.rcParams['axes.linewidth'] = 1.5

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    #ax.set_title("Evaluation time for single step")

    ax.plot(h, eprop, "-ro", label="Ours", linewidth=line_width, markersize=14)
    ax.plot(h, naive_eprop, "-gs", label="Naïve e-prop", linewidth=line_width, markersize=14)
    ax.plot(h, bptt, "-c^", label="BPTT", linewidth=line_width, markersize=14)
    ax.plot(h, rtrl, "-m*", label="RTRL", linewidth=line_width, markersize=14)

    ax.set_yscale("linear")
    ax.set_xlabel("Number of hidden neurons")
    ax.set_ylabel("Time per step [ms]")


exp_RoeFlux_1d(1000)