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

# from graphax.perf import plot_performance_with_GNN, plot_performance_over_size_jax_with_GNN, measure
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
    
    sample_time_batch = jax.vmap(sample_time, in_axes=(0, None, None))
        
    measurements = [sample_time(args, 10, 10)*1000 for i in tqdm(range(num_samples))] # conversion to ms

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

def exp_Encoder(num_samples: int = 1000):

    # Minimal Markowitz elimination order:
    order_mM = [7, 6, 8, 47, 46, 48, 81, 80, 82, 11, 16, 20, 21, 22, 25, 29, 31, 36, 
            51, 56, 60, 61, 62, 65, 69, 71, 75, 85, 88, 90, 89, 9, 12, 10, 26, 
            37, 38, 39, 49, 52, 50, 66, 76, 77, 78, 79, 83, 86, 84, 87, 24, 35, 
            64, 15, 55, 33, 74, 73, 4, 13, 17, 27, 34, 44, 53, 57, 67, 19, 59, 
            1, 2, 3, 18, 23, 41, 42, 43, 58, 63, 30, 70, 28, 68, 5, 45, 32, 14, 
            54, 72, 40]

    # Alphagrad with transformers in PPO network elimination order:
    order_trans = [60, 81, 8, 59, 58, 41, 69, 85, 1, 46, 25, 51, 37, 17, 56, 22, 12, 75, 
         78, 82, 7, 66, 47, 64, 20, 88, 65, 31, 38, 6, 63, 71, 87, 19, 90, 24, 
         80, 83, 27, 48, 77, 49, 29, 23, 76, 9, 79, 67, 61, 26, 89, 86, 18, 34, 
         39, 84, 74, 70, 30, 36, 35, 72, 50, 73, 68, 62, 57, 28, 5, 55, 13, 11, 
         54, 53, 43, 52, 45, 44, 42, 40, 33, 32, 21, 16, 15, 14, 3, 10, 4, 2]

    # Alphagrad with GNN in PPO network elimination order:
    order_GNN = []

    order_GNN_offset = min(order_GNN) - 1
    order_GNN = [x - order_GNN_offset for x in order_GNN]

    scale_factor = 16

    x = jnp.ones((scale_factor*4, scale_factor*4))
    y = jrand.normal(key, (scale_factor*2, scale_factor*4))

    wq1key, wk1key, wv1key, key = jrand.split(key, 4)
    WQ1 = jrand.normal(wq1key, (scale_factor*4, scale_factor*4))
    WK1 = jrand.normal(wk1key, (scale_factor*4, scale_factor*4))
    WV1 = jrand.normal(wv1key, (scale_factor*4, scale_factor*4))

    wq2key, wk2key, wv2key, key = jrand.split(key, 4)
    WQ2 = jrand.normal(wq2key, (scale_factor*4, scale_factor*4))
    WK2 = jrand.normal(wk2key, (scale_factor*4, scale_factor*4))
    WV2 = jrand.normal(wv2key, (scale_factor*4, scale_factor*4))

    w1key, w2key, b1key, b2key = jrand.split(key, 4)
    W1 = jrand.normal(w1key, (scale_factor*4, scale_factor*4))
    b1 = jrand.normal(b1key, (scale_factor*4,))

    W2 = jrand.normal(w2key, (scale_factor*2, scale_factor*4))
    b2 = jrand.normal(b2key, (scale_factor*2, 1))

    xs = (x, y, WQ1, WQ2, WK1, WK2, WV1, WV2, W1, W2, b1, b2, jnp.array([0.]), jnp.array([1.]), jnp.array([0.]), jnp.array([1.]))

    argnums = list(range(len(xs)))

    meas_mM, med_mM = measure_exec_time(Encoder, xs, order_mM, num_samples=num_samples, argnums=argnums)
    meas_trans, med_trans = measure_exec_time(Encoder, xs, order_trans, num_samples=num_samples, argnums=argnums)
    meas_GNN, med_GNN = measure_exec_time(Encoder, xs, order_GNN, num_samples=num_samples, argnums=argnums)

    plt.style.use("seaborn-v0_8-colorblind")
    matplotlib.rc("font", **font)
    matplotlib.rcParams['axes.linewidth'] = 1.5

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    ax.set_title("Transformer Execution Time Comparison")
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
    plt.savefig("exec_times_Transformer.png")

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
    order_GNN = [133, 94, 80, 106, 28, 118, 121, 83, 146, 64, 114, 113, 111, 107, 97, 
                 37, 122, 140, 109, 61, 38, 85, 150, 143, 77, 137, 86, 79, 129, 53, 
                 48, 119, 34, 142, 31, 59, 66, 39, 29, 33, 52, 105, 30, 70, 125, 43, 
                 89, 10, 98, 126, 76, 60, 134, 101, 35, 69, 26, 115, 131, 82, 36, 65, 
                 100, 58, 27, 95, 130, 99, 96, 139, 9, 144, 138, 63, 62, 8, 110, 92, 
                 141, 87, 124, 73, 11, 44, 72, 50, 117, 14, 88, 42, 112, 51, 46, 116, 
                 15, 91, 17, 90, 41, 128, 19, 132, 102, 135, 18, 123, 24, 54, 45, 71, 
                 75, 20, 25, 93, 103, 108, 78, 13, 57, 12, 16, 120, 21, 47, 56, 67, 
                 136, 127, 148, 81, 23, 74, 84, 55, 145, 104, 40, 6, 68, 32, 7, 22, 49]
    
    order_GNN_offset = min(order_GNN) - 1
    order_GNN = [x - order_GNN_offset for x in order_GNN]

    shape = (512,)
    batchsize = 512
    ul0 = jnp.array([.1])
    ul = jnp.array([.1, .2, .3])
    ul4 = jnp.array([.5])
    ur0 = jnp.array([.2])
    ur = jnp.array([.2, .2, .4])
    ur4 = jnp.array([.6])
    xs = (ul0, ul, ul4, ur0, ur, ur4)
    xs = jax.device_put(xs, jax.devices("cpu")[0])

    argnums = list(range(len(xs)))

    meas_mM, med_mM = measure_exec_time(RoeFlux_3d, xs, order_mM, num_samples=num_samples, argnums=argnums)
    meas_trans, med_trans = measure_exec_time(RoeFlux_3d, xs, order_trans, num_samples=num_samples, argnums=argnums)
    meas_GNN, med_GNN = measure_exec_time(RoeFlux_3d, xs, order_GNN, num_samples=num_samples, argnums=argnums)

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


exp_RoeFlux_3d(1000)