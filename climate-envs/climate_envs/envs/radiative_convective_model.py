import os
import pickle

import climlab
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import tephi
import xarray as xr
from gymnasium import spaces
from matplotlib.gridspec import GridSpec
from metpy.plots import SkewT

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
fp = os.path.join(DATASETS_DIR, "air.mon.ltm.1981-2010.nc")

if not os.path.exists(fp):
    print("Downloading NCEP air data ...")
    ncep_url = (
        "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Monthlies/"
    )
    ncep_air = xr.open_dataset(
        ncep_url + "pressure/air.mon.1981-2010.ltm.nc#mode=bytes",
        use_cftime=True,
    )
    ncep_air.to_netcdf(fp)
else:
    print("Loading NCEP air data ...")
    ncep_air = xr.open_dataset(fp, use_cftime=True)

coslat = np.cos(np.deg2rad(ncep_air.lat))
weight = coslat / coslat.mean(dim="lat")
Tobs = (ncep_air.air * weight).mean(dim=("lat", "lon", "time"))


class RadiativeConvectiveModelEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode=None, locale="uk"):
        self.min_emissivity = 0  # perfect reflector
        self.max_emissivity = 1  # black body

        self.min_adj_lapse_rate = 5.5
        self.max_adj_lapse_rate = 9.8  # dry adiabatic lapse rate

        self.min_temperature = -90
        self.max_temperature = 90

        self.action_space = spaces.Box(
            low=np.array(
                [self.min_emissivity, self.min_adj_lapse_rate],
                dtype=np.float32,
            ),
            high=np.array(
                [self.max_emissivity, self.max_adj_lapse_rate],
                dtype=np.float32,
            ),
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=self.min_temperature,
            high=self.max_temperature,
            shape=(len(Tobs.level),),
            dtype=np.float32,
        )

        assert (
            render_mode is None or render_mode in self.metadata["render_modes"]
        )

        # with open(f"{DATASETS_DIR}/climlab_rcm_200.pkl", "rb") as file:
        #     self.climlab_rcm = pickle.load(file)

        self.locale = locale
        self.render_mode = render_mode
        self.reset()

    def _get_obs(self):
        return self._get_temp().values

    def _get_temp(self, model="RL"):
        if model == "RL":
            rcm = self.rcm
        elif model == "climlab":
            rcm = self.climlab_rcm
        temp = np.concatenate([rcm.Tatm, rcm.Ts], dtype=np.float32)
        temp -= climlab.constants.tempCtoK
        temp = xr.DataArray(temp, coords={"level": Tobs.level.values[::-1]})
        return temp

    def _get_info(self):
        return {"_": None}

    def _get_params(self):
        emissivity = self.rcm.subprocess["Radiation (net)"].emissivity
        adj_lapse_rate = self.rcm.subprocess["Convection"].adj_lapse_rate
        params = np.array([emissivity, adj_lapse_rate], dtype=np.float32)
        return params

    def _get_state(self):
        state = self._get_temp().values
        return state

    def step(self, action):
        emissivity, adj_lapse_rate = action[0], action[1]

        emissivity = np.clip(
            emissivity, self.min_emissivity, self.max_emissivity
        )
        adj_lapse_rate = np.clip(
            adj_lapse_rate, self.min_adj_lapse_rate, self.max_adj_lapse_rate
        )
        # adj_lapse_rate = (self.max_adj_lapse_rate - self.min_adj_lapse_rate) * scipy.special.expit(adj_lapse_rate)
        # adj_lapse_rate += self.min_adj_lapse_rate

        self.rcm.subprocess["Radiation (net)"].emissivity = emissivity
        self.rcm.subprocess["Convection"].adj_lapse_rate = adj_lapse_rate
        self.rcm.step_forward()
        self.climlab_rcm.step_forward()

        Tprofile = self._get_temp().values
        costs = np.mean((Tprofile - Tobs.values[::-1]) ** 2)

        self.state = self._get_state()

        return self._get_obs(), -costs, False, False, self._get_info()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rce_state = climlab.column_state(lev=Tobs.level[1:], water_depth=2.5)
        h2o = climlab.radiation.ManabeWaterVapor(
            state=rce_state, lev=Tobs.level[1:]
        )
        rad = climlab.radiation.RRTMG(
            name="Radiation (net)",
            state=rce_state,
            specific_humidity=h2o.q,
            S0=1365.0,
            timestep=climlab.constants.seconds_per_day,
            albedo=0.25,
        )

        conv = climlab.convection.ConvectiveAdjustment(
            name="Convection",
            state=rce_state,
            adj_lapse_rate=6.5,
            timestep=rad.timestep,
        )

        self.rcm = climlab.couple([rad, conv], name="RCE Model w/ RL")

        # Start from Radiative Equlibrium
        # for n in range(1000):
        #     self.rcm.subprocess["Radiation (net)"].step_forward()

        # Start from Isothermal initial state
        self.rcm.state.Tatm[:] = self.rcm.state.Ts

        # Initialize a climlab RCM model clone
        self.climlab_rcm = climlab.process_like(self.rcm)
        self.climlab_rcm.name = "RCE Model"

        self.state = self._get_state()
        return self._get_obs(), self._get_info()

    def _render_frame(self):
        fig = plt.figure(figsize=(27, 9))
        gs = GridSpec(1, 3, figure=fig)

        params = self._get_params()

        Tprofile_RL = self._get_temp()
        T_diff_RL = Tobs.sel(level=[100, 200]) - Tprofile_RL.sel(
            level=[100, 200]
        )
        T_diff_RL = np.abs(T_diff_RL)

        Tprofile_climlab = self._get_temp(model="climlab")
        T_diff_climlab = Tobs.sel(level=[100, 200]) - Tprofile_climlab.sel(
            level=[100, 200]
        )
        T_diff_climlab = np.abs(T_diff_climlab)

        # Left subplot: emissivity and adj_lapse_rate as bar plots
        ax1 = fig.add_subplot(gs[0, 0])

        ax1_labels = ["Emissivity", "Adj Lapse Rate"]
        ax1_colors = ["tab:blue", "tab:blue"]
        ax1_bars = ax1.bar(ax1_labels, params, color=ax1_colors, width=0.75)
        ax1.set_ylim(0, 10)
        ax1.set_ylabel("Value", fontsize=14)
        ax1.set_title("Parameters")

        # Add values on top of the bars
        for bar in ax1_bars:
            height = bar.get_height()
            ax1.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        if self.locale == "us":

            # Middle subplot: Skew-T plot

            skew = SkewT(fig, subplot=gs[0, 1], rotation=30)
            self._make_skewT(skew, "SkewT-logP")
            self._add_profile(skew, self.rcm)
            self._add_profile(skew, self.climlab_rcm)

        elif self.locale == "uk":

            # Middle subplot: Tephigram

            ax2 = fig.add_subplot(gs[0, 1])
            tephi.ISOBAR_FIXED = [50, 1000, 100]
            tpg = tephi.Tephigram(
                figure=ax2.get_figure(), anchor=[(1000, -10), (70, -25)]
            )
            tpg.plot(
                self._get_tephigram_data(self.rcm),
                label=self.rcm.name,
                color="tab:blue",
            )
            tpg.plot(
                self._get_tephigram_data(self.climlab_rcm),
                label=self.climlab_rcm.name,
                color="tab:orange",
            )
            tpg.plot(
                zip(Tobs.level, Tobs),
                color="black",
                linestyle="-",
                linewidth=2,
                label="Observations",
            )
            tpg.axes.legend()
            ax2.set_title("Tephigram")
            ax2.set_axis_off()

        # Right subplot: emissivity and adj_lapse_rate as bar plots
        ax3 = fig.add_subplot(gs[0, 2])
        ax3_bar_width = 0.4
        ax3_labels = ["T_diff @ 100hPa", "T_diff @ 200 hPa"]
        ax3_ind = np.arange(1, 1 + len(ax3_labels))
        ax3_colors_RL = ["tab:blue", "tab:blue"]
        ax3_bars_RL = ax3.bar(
            ax3_ind - (ax3_bar_width / 2),
            T_diff_RL,
            color=ax3_colors_RL,
            width=ax3_bar_width,
        )
        ax3_colors_climlab = ["tab:orange", "tab:orange"]
        ax3_bars_climlab = ax3.bar(
            ax3_ind + (ax3_bar_width / 2),
            T_diff_climlab,
            color=ax3_colors_climlab,
            width=ax3_bar_width,
        )
        ax3.set_ylim(0, 10)
        ax3.set_ylabel("Difference [$\degree$C]", fontsize=14)
        ax3.set_title("Temperature Differences")
        ax3.set_xticks(ax3_ind)
        ax3.set_xticklabels(ax3_labels)
        ax3.legend(
            (ax3_bars_RL[0], ax3_bars_climlab[0]),
            ("RCE Model w/ RL", "RCE Model"),
        )

        # Add values on top of the bars
        ax3_bars = ax3_bars_RL + ax3_bars_climlab
        for bar in ax3_bars:
            height = bar.get_height()
            ax3.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        return fig

    def _get_tephigram_data(self, rcm):

        temp = np.concatenate([rcm.Tatm, rcm.Ts], dtype=np.float32)
        temp -= climlab.constants.tempCtoK
        levels = np.concatenate([rcm.lev, [1000]], dtype=np.float32)

        df = pd.DataFrame(zip(levels, temp))
        df.columns = ["pressure", "temperature"]

        return df.values

    def _make_skewT(self, skew, title=None):
        skew.plot(
            Tobs.level,
            Tobs,
            color="black",
            linestyle="-",
            linewidth=2,
            label="Observations",
        )
        skew.ax.set_ylim(1050, 10)
        skew.ax.set_xlim(-90, 45)

        skew.plot_dry_adiabats(linewidth=0.5)
        skew.plot_moist_adiabats(linewidth=0.5)

        skew.ax.legend()
        skew.ax.set_xlabel("Temperature [$\degree$C]", fontsize=14)
        skew.ax.set_ylabel("Pressure [hPa]", fontsize=14)
        if title:
            skew.ax.set_title(title)

    def _add_profile(self, skew, model, linestyle="-", color=None):
        line = skew.plot(
            model.lev,
            model.Tatm - climlab.constants.tempCtoK,
            label=model.name,
            linewidth=2,
        )[0]
        skew.plot(
            1000,
            model.Ts - climlab.constants.tempCtoK,
            "o",
            markersize=8,
            color=line.get_color(),
        )
        skew.ax.legend()

    def render(self):
        if self.render_mode == "human":
            self._render_frame()
            plt.show()
        elif self.render_mode == "rgb_array":
            fig = self._render_frame()
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape((height, width, 3))
            plt.close(fig)
            return image
