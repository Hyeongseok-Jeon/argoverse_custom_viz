# TODO interesting scene in loss ablation study 6366 5900 2102 32602 31111 36796 31627 19929 27795 26985 10419 6975 31111 20463 30126 8723 30281
# TODO case study : 36796
# TODO loss comparison : 1825  23362 30458 26678 33518(downstream & no representation), 29425 (no reconstruction & no recon no repre), 23405(no reconstruction & downstream)
# TODO loss comparison : 36445 (ds, no recon no repre)
# 2449 15339(no recon loss) 18861 35883 29870 31513 10737 36694 25770 22563 12220
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import numpy as np
import importlib.util
import sys
from torch.utils.data.distributed import DistributedSampler
import progressbar

import tkinter
import tkinter.filedialog
import os
import ui_files.gui
import torch
import argparse
import time
from torch.utils.data import DataLoader
import argoverse.evaluation.eval_forecasting as eval
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.utils.mpl_plotting_utils import draw_lane_polygons, plot_nearby_centerlines
from importlib import import_module

am = ArgoverseMap()
import random
import csv
from mpi4py import MPI
from glob import glob
import importlib.util
from tqdm import tqdm
from PIL import Image

comm = MPI.COMM_WORLD


class MainDialog(QMainWindow, ui_files.gui.Ui_Dialog):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.root_dir = os.getcwd()
        root = tkinter.Tk()
        root.withdraw()
        # self.original_data_dir = tkinter.filedialog.askdirectory(title='Select directory for the original dataset', parent=root, initialdir=os.path.join(self.root_dir))
        self.original_data_dir = '/home/jhs/Desktop/Bidirectional-Multi-view-Coding/bMVC/future_prediction_for_autonomous_vehicle/LaneGCN/dataset/val/data'
        self.afl = ArgoverseForecastingLoader(self.original_data_dir)

        self.loadData.clicked.connect(self.data_load)
        self.next_idx.clicked.connect(self.next)
        self.prev_idx.clicked.connect(self.prev)
        # self.pause.clocked.connect(self.pause)
        # self.stop.clicked.connect(self.stop)

        self.fov = 50
        self.x_offset = 0
        self.y_offset = 0
        self.fov_data.setText(str(self.fov))
        self.x_offset_data.setText(str(self.x_offset))
        self.y_offset_data.setText(str(self.y_offset))
        self.fov_data.returnPressed.connect(self.fov_update)
        self.x_offset_data.returnPressed.connect(self.fov_update)
        self.y_offset_data.returnPressed.connect(self.fov_update)
        self.idx_data.returnPressed.connect(self.idx_update)
        self.data_dir_data.returnPressed.connect(self.index_update)
        self.data_reform()

    def data_load(self):
        root = tkinter.Tk()
        root.withdraw()
        data_file = tkinter.filedialog.askopenfilename(parent=root, initialdir=os.path.join(self.root_dir))

        self.data = np.load(data_file)
        self.row_index = np.where(self.data[:, 0, 0, -1] == 1)[0][0]
        self.dataInfo.setText(data_file + ' is loaded successfully')
        self.data_num.setText(str(self.data.shape[0]))
        self.data_update(self.row_index)

    def data_update(self, row_index):
        self.data_index = self.find_original_index(row_index)
        self.cur_data = self.afl.get(self.original_data_dir + '/' + str(self.data_index) + '.csv')
        self.idx_data.setText(str(row_index))
        self.data_dir_data.setText(str(self.cur_data.current_seq))
        ego_id = self.cur_data.seq_df.TRACK_ID[self.cur_data.seq_df.OBJECT_TYPE == 'AV'].tolist()[0]
        target_id = self.cur_data.seq_df.TRACK_ID[self.cur_data.seq_df.OBJECT_TYPE == 'AGENT'].tolist()[0]

        self.ego_hist = np.concatenate([np.expand_dims(self.cur_data.seq_df.X[self.cur_data.seq_df.TRACK_ID == ego_id][:20].to_numpy(), axis=-1),
                                        np.expand_dims(self.cur_data.seq_df.Y[self.cur_data.seq_df.TRACK_ID == ego_id][:20].to_numpy(), axis=-1)], axis=-1).astype(np.float32)
        self.ego_fut = np.concatenate([np.expand_dims(self.cur_data.seq_df.X[self.cur_data.seq_df.TRACK_ID == ego_id][19:].to_numpy(), axis=-1),
                                       np.expand_dims(self.cur_data.seq_df.Y[self.cur_data.seq_df.TRACK_ID == ego_id][19:].to_numpy(), axis=-1)], axis=-1).astype(np.float32)
        self.target_hist = np.concatenate([np.expand_dims(self.cur_data.seq_df.X[self.cur_data.seq_df.TRACK_ID == target_id][:20].to_numpy(), axis=-1),
                                           np.expand_dims(self.cur_data.seq_df.Y[self.cur_data.seq_df.TRACK_ID == target_id][:20].to_numpy(), axis=-1)], axis=-1).astype(np.float32)
        self.target_fut = np.concatenate([np.expand_dims(self.cur_data.seq_df.X[self.cur_data.seq_df.TRACK_ID == target_id][19:].to_numpy(), axis=-1),
                                          np.expand_dims(self.cur_data.seq_df.Y[self.cur_data.seq_df.TRACK_ID == target_id][19:].to_numpy(), axis=-1)], axis=-1).astype(np.float32)
        self.GT = self.data[row_index, 1, :, :2]
        self.prediction = self.data[row_index, 0, :, :2]
        self.city = self.cur_data.seq_df.CITY_NAME[0]
        self.visualization()

    def find_original_index(self, row_index):
        disp = self.target_traj_converted[:, 20:, :] - self.data[row_index, 1, :, :2]
        disp_list = []
        for j in range(len(self.afl)):
            disp_list.append(np.sum(np.linalg.norm(disp[j], axis=1)))
        I = disp_list.index(min(disp_list))
        disp_list.sort()
        print(disp_list[:3])
        return int(self.data_idx_map[I])

    def fov_update(self):
        self.fov = int(self.fov_data.text())
        self.x_offset = int(self.x_offset_data.text())
        self.y_offset = int(self.y_offset_data.text())
        self.visualization()

    def idx_update(self):
        if len(np.where(self.data[:,0,0,-1] == float(self.idx_data.text()))[0]) == 0:
            self.idx_data.setText('no data')
            self.data_dir_data.setText('no data')
        else:
            self.row_index = np.where(self.data[:,0,0,-1] == float(self.idx_data.text()))[0][0]
            self.data_update(self.row_index)

    def index_update(self):
        target_index = float(self.data_dir_data.text())
        row_index_cand = np.where(self.data[:, 0, 0, -1] == target_index)[0]
        if len(row_index_cand) > 0:
            self.row_index = np.where(self.data[:, 0, 0, -1] == target_index)[0][0]
            self.data_update(self.row_index)
        else:
            self.idx_data.setText('no data')
            self.data_dir_data.setText('no data')

    def next(self):
        while True:
            cur_idx = self.data_index
            cur_idx = cur_idx + 1
            row_index_cand = np.where(self.data[:, 0, 0, -1] == cur_idx)[0]
            self.idx_data.setText(str(row_index_cand))
            if len(row_index_cand) > 0:
                try:
                    self.row_index = np.where(self.data[:, 0, 0, -1] == cur_idx)[0][0]
                    self.data_update(self.row_index)
                    break
                except:
                    pass

    def prev(self):
        while True:
            cur_idx = self.data_index
            cur_idx = cur_idx - 1
            row_index_cand = np.where(self.data[:, 0, 0, -1] == cur_idx)[0]
            self.idx_data.setText(str(row_index_cand))
            if cur_idx == 0:
                self.idx_data.setText('no previous data')
                self.data_dir_data.setText('no previous data')
                self.data_index = 0
                break
            else:
                if len(row_index_cand) > 0:
                    try:
                        self.row_index = np.where(self.data[:, 0, 0, -1] == cur_idx)[0][0]
                        self.data_update(self.row_index)
                        break
                    except:
                        pass

    def visualization(self):
        self.pred_plot.canvas.ax.clear()
        ego_cur_pos = self.ego_hist[-1,:]
        xmin = ego_cur_pos[0] - self.fov/2 + self.x_offset
        xmax = ego_cur_pos[0] + self.fov/2 + self.x_offset
        ymin = ego_cur_pos[1] - self.fov/2 + self.y_offset
        ymax = ego_cur_pos[1] + self.fov/2 + self.y_offset
        city_name = self.city

        local_lane_polygons = am.find_local_lane_polygons([xmin-self.fov/2, xmax+self.fov/2, ymin-self.fov/2, ymax+self.fov/2], city_name)
        draw_lane_polygons(self.pred_plot.canvas.ax, local_lane_polygons, color='darkgray')

        ego_hist_x = self.ego_hist[:,0]
        ego_hist_y = self.ego_hist[:,1]
        ego_fut_x = self.ego_fut[:,0]
        ego_fut_y = self.ego_fut[:,1]
        target_hist_x = self.target_hist[:,0]
        target_hist_y = self.target_hist[:,1]
        target_fut_x = self.target_fut[:,0]
        target_fut_y = self.target_fut[:,1]

        # cur_time = raw_data[ego_index[19]][0].split(',')[0]
        # cur_sur_index = [i for i, x in enumerate(time_stamp) if x == cur_time]
        # sur_x = x[cur_sur_index]
        # sur_y = y[cur_sur_index]
        # self.pred_plot_1.canvas.ax.scatter(sur_x, sur_y, color='silver')
        # self.pred_plot_2.canvas.ax.scatter(sur_x, sur_y, color='silver')
        # self.pred_plot_3.canvas.ax.scatter(sur_x, sur_y, color='silver')
        # self.pred_plot_4.canvas.ax.scatter(sur_x, sur_y, color='silver')
        # self.pred_plot_5.canvas.ax.scatter(sur_x, sur_y, color='silver')
        # self.pred_plot_0.canvas.ax.scatter(sur_x, sur_y, color='silver')
        #
        self.pred_plot.canvas.ax.plot(ego_hist_x, ego_hist_y, '-', color='red')
        self.pred_plot.canvas.ax.scatter(ego_hist_x[-1], ego_hist_y[-1], color='red')
        self.pred_plot.canvas.ax.plot(target_hist_x, target_hist_y, '-', color='blue')
        self.pred_plot.canvas.ax.scatter(target_hist_x[-1], target_hist_y[-1], color='blue')
        self.pred_plot.canvas.ax.plot(ego_fut_x, ego_fut_y, '-', color='red')
        self.pred_plot.canvas.ax.scatter(ego_fut_x[-1], ego_fut_y[-1], color='red')
        self.pred_plot.canvas.ax.plot(target_fut_x, target_fut_y, '-', color='blue')
        self.pred_plot.canvas.ax.scatter(target_fut_x[-1], target_fut_y[-1], color='blue')

        self.pred_plot.canvas.ax.axis('equal')
        self.pred_plot.canvas.ax.set_xlim([xmin.item(), xmax.item()])
        self.pred_plot.canvas.ax.set_ylim([ymin.item(), ymax.item()])
        self.pred_plot.canvas.draw()


        #
        # ego_aug = self.cur_data['ego_aug'][0]['traj'].numpy().copy()
        # ego_aug = np.concatenate([ego_aug, np.zeros_like(ego_aug[:, 0:1, :])], axis=1)
        # marker_size = 50
        # self.state_for_play = np.zeros(shape=(0, 2, 40, 2))
        #
        # for i in range(ego_aug.shape[0]):
        #     ego_aug[i, :, :] = np.concatenate([np.expand_dims(np.asarray([ego_hist_x[-1], ego_hist_y[-1]]), axis=0), ego_aug[i, :20, :]], axis=0)
        #     ego_facecolors = 'none'
        #     sur_facecolors = 'none'
        #     if i == 0:
        #         marker_shape = 's'
        #     elif i == 1:
        #         marker_shape = 'P'
        #     elif i == 2:
        #         marker_shape = '^'
        #     elif i == 3:
        #         marker_shape = '*'
        #     elif i == 4:
        #         marker_shape = 's'
        #     elif i == 5:
        #         marker_shape = 's'
        #     elif i == 6:
        #         marker_shape = 's'
        #     if self.cand_toggles_1[i + 1].isChecked():
        #         aug_x = ego_aug[i, :, 0]
        #         aug_y = ego_aug[i, :, 1]
        #         self.pred_plot_1.canvas.ax.plot(aug_x, aug_y, '--', color='red')
        #         self.pred_plot_1.canvas.ax.scatter(aug_x[-1], aug_y[-1], marker_size, marker=marker_shape, facecolors=ego_facecolors, edgecolors='red')
        #         self.pred_plot_0.canvas.ax.plot(aug_x, aug_y, '--', color='red')
        #         self.pred_plot_0.canvas.ax.scatter(aug_x[-1], aug_y[-1], marker_size, marker=marker_shape, facecolors=ego_facecolors, edgecolors='red')
        #         pred_reg = self.pred_out_1[i + 1]
        #         self.pred_plot_1.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
        #         self.pred_plot_1.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker=marker_shape, facecolors=sur_facecolors, edgecolors='blue')
        #         self.pred_plot_0.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
        #         self.pred_plot_0.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker=marker_shape, facecolors=sur_facecolors, edgecolors='blue')
        #         ego_x_play = np.expand_dims(np.concatenate((ego_hist_x[:-1], aug_x)), axis=-1)
        #         ego_y_play = np.expand_dims(np.concatenate((ego_hist_y[:-1], aug_y)), axis=-1)
        #         ego_traj_play = np.expand_dims(np.concatenate((ego_x_play, ego_y_play), axis=-1), axis=0)
        #         target_x_play = np.expand_dims(np.concatenate((target_hist_x, pred_reg[:, 0].cpu())), axis=-1)
        #         target_y_play = np.expand_dims(np.concatenate((target_hist_y, pred_reg[:, 1].cpu())), axis=-1)
        #         target_traj_play = np.expand_dims(np.concatenate((target_x_play, target_y_play), axis=-1), axis=0)
        #         states = np.expand_dims(np.concatenate((ego_traj_play, target_traj_play), axis=0), axis=0)
        #         self.state_for_play = np.concatenate((self.state_for_play, states), axis=0)
        #     if self.cand_toggles_2[i + 1].isChecked():
        #         aug_x = ego_aug[i, :, 0]
        #         aug_y = ego_aug[i, :, 1]
        #         self.pred_plot_2.canvas.ax.plot(aug_x, aug_y, '--', color='red')
        #         self.pred_plot_2.canvas.ax.scatter(aug_x[-1], aug_y[-1], marker_size, marker=marker_shape, facecolors=ego_facecolors, edgecolors='red')
        #         self.pred_plot_0.canvas.ax.plot(aug_x, aug_y, '--', color='red')
        #         self.pred_plot_0.canvas.ax.scatter(aug_x[-1], aug_y[-1], marker_size, marker=marker_shape, facecolors=ego_facecolors, edgecolors='red')
        #         pred_reg = self.pred_out_2[i + 1]
        #         self.pred_plot_2.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
        #         self.pred_plot_2.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker=marker_shape, facecolors=sur_facecolors, edgecolors='blue')
        #         self.pred_plot_0.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
        #         self.pred_plot_0.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker=marker_shape, facecolors=sur_facecolors, edgecolors='blue')
        #         ego_x_play = np.expand_dims(np.concatenate((ego_hist_x[:-1], aug_x)), axis=-1)
        #         ego_y_play = np.expand_dims(np.concatenate((ego_hist_y[:-1], aug_y)), axis=-1)
        #         ego_traj_play = np.expand_dims(np.concatenate((ego_x_play, ego_y_play), axis=-1), axis=0)
        #         target_x_play = np.expand_dims(np.concatenate((target_hist_x, pred_reg[:, 0].cpu())), axis=-1)
        #         target_y_play = np.expand_dims(np.concatenate((target_hist_y, pred_reg[:, 1].cpu())), axis=-1)
        #         target_traj_play = np.expand_dims(np.concatenate((target_x_play, target_y_play), axis=-1), axis=0)
        #         states = np.expand_dims(np.concatenate((ego_traj_play, target_traj_play), axis=0), axis=0)
        #         self.state_for_play = np.concatenate((self.state_for_play, states), axis=0)
        #     if self.cand_toggles_3[i + 1].isChecked():
        #         aug_x = ego_aug[i, :, 0]
        #         aug_y = ego_aug[i, :, 1]
        #         self.pred_plot_3.canvas.ax.plot(aug_x, aug_y, '--', color='red')
        #         self.pred_plot_3.canvas.ax.scatter(aug_x[-1], aug_y[-1], marker_size, marker=marker_shape, facecolors=ego_facecolors, edgecolors='red')
        #         self.pred_plot_0.canvas.ax.plot(aug_x, aug_y, '--', color='red')
        #         self.pred_plot_0.canvas.ax.scatter(aug_x[-1], aug_y[-1], marker_size, marker=marker_shape, facecolors=ego_facecolors, edgecolors='red')
        #         pred_reg = self.pred_out_3[i + 1]
        #         self.pred_plot_3.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
        #         self.pred_plot_3.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker=marker_shape, facecolors=sur_facecolors, edgecolors='blue')
        #         self.pred_plot_0.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
        #         self.pred_plot_0.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker=marker_shape, facecolors=sur_facecolors, edgecolors='blue')
        #         ego_x_play = np.expand_dims(np.concatenate((ego_hist_x[:-1], aug_x)), axis=-1)
        #         ego_y_play = np.expand_dims(np.concatenate((ego_hist_y[:-1], aug_y)), axis=-1)
        #         ego_traj_play = np.expand_dims(np.concatenate((ego_x_play, ego_y_play), axis=-1), axis=0)
        #         target_x_play = np.expand_dims(np.concatenate((target_hist_x, pred_reg[:, 0].cpu())), axis=-1)
        #         target_y_play = np.expand_dims(np.concatenate((target_hist_y, pred_reg[:, 1].cpu())), axis=-1)
        #         target_traj_play = np.expand_dims(np.concatenate((target_x_play, target_y_play), axis=-1), axis=0)
        #         states = np.expand_dims(np.concatenate((ego_traj_play, target_traj_play), axis=0), axis=0)
        #         self.state_for_play = np.concatenate((self.state_for_play, states), axis=0)
        #     if self.cand_toggles_4[i + 1].isChecked():
        #         aug_x = ego_aug[i, :, 0]
        #         aug_y = ego_aug[i, :, 1]
        #         self.pred_plot_4.canvas.ax.plot(aug_x, aug_y, '--', color='red')
        #         self.pred_plot_4.canvas.ax.scatter(aug_x[-1], aug_y[-1], marker_size, marker=marker_shape, facecolors=ego_facecolors, edgecolors='red')
        #         self.pred_plot_0.canvas.ax.plot(aug_x, aug_y, '--', color='red')
        #         self.pred_plot_0.canvas.ax.scatter(aug_x[-1], aug_y[-1], marker_size, marker=marker_shape, facecolors=ego_facecolors, edgecolors='red')
        #         pred_reg = self.pred_out_4[i + 1]
        #         self.pred_plot_4.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
        #         self.pred_plot_4.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker=marker_shape, facecolors=sur_facecolors, edgecolors='blue')
        #         self.pred_plot_0.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
        #         self.pred_plot_0.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker=marker_shape, facecolors=sur_facecolors, edgecolors='blue')
        #         ego_x_play = np.expand_dims(np.concatenate((ego_hist_x[:-1], aug_x)), axis=-1)
        #         ego_y_play = np.expand_dims(np.concatenate((ego_hist_y[:-1], aug_y)), axis=-1)
        #         ego_traj_play = np.expand_dims(np.concatenate((ego_x_play, ego_y_play), axis=-1), axis=0)
        #         target_x_play = np.expand_dims(np.concatenate((target_hist_x, pred_reg[:, 0].cpu())), axis=-1)
        #         target_y_play = np.expand_dims(np.concatenate((target_hist_y, pred_reg[:, 1].cpu())), axis=-1)
        #         target_traj_play = np.expand_dims(np.concatenate((target_x_play, target_y_play), axis=-1), axis=0)
        #         states = np.expand_dims(np.concatenate((ego_traj_play, target_traj_play), axis=0), axis=0)
        #         self.state_for_play = np.concatenate((self.state_for_play, states), axis=0)
        #     if self.cand_toggles_5[i + 1].isChecked():
        #         aug_x = ego_aug[i, :, 0]
        #         aug_y = ego_aug[i, :, 1]
        #         self.pred_plot_5.canvas.ax.plot(aug_x, aug_y, '--', color='red')
        #         self.pred_plot_5.canvas.ax.scatter(aug_x[-1], aug_y[-1], marker_size, marker=marker_shape, facecolors=ego_facecolors, edgecolors='red')
        #         self.pred_plot_0.canvas.ax.plot(aug_x, aug_y, '--', color='red')
        #         self.pred_plot_0.canvas.ax.scatter(aug_x[-1], aug_y[-1], marker_size, marker=marker_shape, facecolors=ego_facecolors, edgecolors='red')
        #         pred_reg = self.pred_out_5[i + 1]
        #         self.pred_plot_5.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
        #         self.pred_plot_5.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker=marker_shape, facecolors=sur_facecolors, edgecolors='blue')
        #         self.pred_plot_0.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
        #         self.pred_plot_0.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker=marker_shape, facecolors=sur_facecolors, edgecolors='blue')
        #         ego_x_play = np.expand_dims(np.concatenate((ego_hist_x[:-1], aug_x)), axis=-1)
        #         ego_y_play = np.expand_dims(np.concatenate((ego_hist_y[:-1], aug_y)), axis=-1)
        #         ego_traj_play = np.expand_dims(np.concatenate((ego_x_play, ego_y_play), axis=-1), axis=0)
        #         target_x_play = np.expand_dims(np.concatenate((target_hist_x, pred_reg[:, 0].cpu())), axis=-1)
        #         target_y_play = np.expand_dims(np.concatenate((target_hist_y, pred_reg[:, 1].cpu())), axis=-1)
        #         target_traj_play = np.expand_dims(np.concatenate((target_x_play, target_y_play), axis=-1), axis=0)
        #         states = np.expand_dims(np.concatenate((ego_traj_play, target_traj_play), axis=0), axis=0)
        #         self.state_for_play = np.concatenate((self.state_for_play, states), axis=0)
        #
        # if self.ego_path_enable_1.isChecked():
        #     self.pred_plot_1.canvas.ax.plot(ego_fut_x, ego_fut_y, '--', color='red')
        #     self.pred_plot_1.canvas.ax.scatter(ego_fut_x[-1], ego_fut_y[-1], marker_size, marker="o", facecolors='red', edgecolors='red')
        #     self.pred_plot_0.canvas.ax.plot(ego_fut_x, ego_fut_y, '--', color='red')
        #     self.pred_plot_0.canvas.ax.scatter(ego_fut_x[-1], ego_fut_y[-1], marker_size, marker="o", facecolors='red', edgecolors='red')
        #     pred_reg = self.pred_out_1[0]
        #     self.pred_plot_1.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
        #     self.pred_plot_1.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker="o", facecolors='blue', edgecolors='blue')
        #     self.pred_plot_0.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
        #     self.pred_plot_0.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker="o", facecolors='blue', edgecolors='blue')
        #     ego_x_play = np.expand_dims(np.concatenate((ego_hist_x[:-1], ego_fut_x)), axis=-1)
        #     ego_y_play = np.expand_dims(np.concatenate((ego_hist_y[:-1], ego_fut_y)), axis=-1)
        #     ego_traj_play = np.expand_dims(np.concatenate((ego_x_play, ego_y_play), axis=-1), axis=0)
        #     target_x_play = np.expand_dims(np.concatenate((target_hist_x, pred_reg[:, 0].cpu())), axis=-1)
        #     target_y_play = np.expand_dims(np.concatenate((target_hist_y, pred_reg[:, 1].cpu())), axis=-1)
        #     target_traj_play = np.expand_dims(np.concatenate((target_x_play, target_y_play), axis=-1), axis=0)
        #     states = np.expand_dims(np.concatenate((ego_traj_play, target_traj_play), axis=0), axis=0)
        #     self.state_for_play = np.concatenate((self.state_for_play, states), axis=0)
        # if self.ego_path_enable_2.isChecked():
        #     self.pred_plot_2.canvas.ax.plot(ego_fut_x, ego_fut_y, '--', color='red')
        #     self.pred_plot_2.canvas.ax.scatter(ego_fut_x[-1], ego_fut_y[-1], marker_size, marker="o", facecolors='red', edgecolors='red')
        #     self.pred_plot_0.canvas.ax.plot(ego_fut_x, ego_fut_y, '--', color='red')
        #     self.pred_plot_0.canvas.ax.scatter(ego_fut_x[-1], ego_fut_y[-1], marker_size, marker="o", facecolors='red', edgecolors='red')
        #     pred_reg = self.pred_out_2[0]
        #     self.pred_plot_2.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
        #     self.pred_plot_2.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker="o", facecolors='blue', edgecolors='blue')
        #     self.pred_plot_0.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
        #     self.pred_plot_0.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker="o", facecolors='blue', edgecolors='blue')
        #     ego_x_play = np.expand_dims(np.concatenate((ego_hist_x[:-1], ego_fut_x)), axis=-1)
        #     ego_y_play = np.expand_dims(np.concatenate((ego_hist_y[:-1], ego_fut_y)), axis=-1)
        #     ego_traj_play = np.expand_dims(np.concatenate((ego_x_play, ego_y_play), axis=-1), axis=0)
        #     target_x_play = np.expand_dims(np.concatenate((target_hist_x, pred_reg[:, 0].cpu())), axis=-1)
        #     target_y_play = np.expand_dims(np.concatenate((target_hist_y, pred_reg[:, 1].cpu())), axis=-1)
        #     target_traj_play = np.expand_dims(np.concatenate((target_x_play, target_y_play), axis=-1), axis=0)
        #     states = np.expand_dims(np.concatenate((ego_traj_play, target_traj_play), axis=0), axis=0)
        #     self.state_for_play = np.concatenate((self.state_for_play, states), axis=0)
        # if self.ego_path_enable_3.isChecked():
        #     self.pred_plot_3.canvas.ax.plot(ego_fut_x, ego_fut_y, '--', color='red')
        #     self.pred_plot_3.canvas.ax.scatter(ego_fut_x[-1], ego_fut_y[-1], marker_size, marker="o", facecolors='red', edgecolors='red')
        #     self.pred_plot_0.canvas.ax.plot(ego_fut_x, ego_fut_y, '--', color='red')
        #     self.pred_plot_0.canvas.ax.scatter(ego_fut_x[-1], ego_fut_y[-1], marker_size, marker="o", facecolors='red', edgecolors='red')
        #     pred_reg = self.pred_out_3[0]
        #     self.pred_plot_3.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
        #     self.pred_plot_3.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker="o", facecolors='blue', edgecolors='blue')
        #     self.pred_plot_0.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
        #     self.pred_plot_0.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker="o", facecolors='blue', edgecolors='blue')
        #     ego_x_play = np.expand_dims(np.concatenate((ego_hist_x[:-1], ego_fut_x)), axis=-1)
        #     ego_y_play = np.expand_dims(np.concatenate((ego_hist_y[:-1], ego_fut_y)), axis=-1)
        #     ego_traj_play = np.expand_dims(np.concatenate((ego_x_play, ego_y_play), axis=-1), axis=0)
        #     target_x_play = np.expand_dims(np.concatenate((target_hist_x, pred_reg[:, 0].cpu())), axis=-1)
        #     target_y_play = np.expand_dims(np.concatenate((target_hist_y, pred_reg[:, 1].cpu())), axis=-1)
        #     target_traj_play = np.expand_dims(np.concatenate((target_x_play, target_y_play), axis=-1), axis=0)
        #     states = np.expand_dims(np.concatenate((ego_traj_play, target_traj_play), axis=0), axis=0)
        #     self.state_for_play = np.concatenate((self.state_for_play, states), axis=0)
        # if self.ego_path_enable_4.isChecked():
        #     self.pred_plot_4.canvas.ax.plot(ego_fut_x, ego_fut_y, '--', color='red')
        #     self.pred_plot_4.canvas.ax.scatter(ego_fut_x[-1], ego_fut_y[-1], marker_size, marker="o", facecolors='red', edgecolors='red')
        #     self.pred_plot_0.canvas.ax.plot(ego_fut_x, ego_fut_y, '--', color='red')
        #     self.pred_plot_0.canvas.ax.scatter(ego_fut_x[-1], ego_fut_y[-1], marker_size, marker="o", facecolors='red', edgecolors='red')
        #     pred_reg = self.pred_out_4[0]
        #     self.pred_plot_4.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
        #     self.pred_plot_4.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker="o", facecolors='blue', edgecolors='blue')
        #     self.pred_plot_0.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
        #     self.pred_plot_0.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker="o", facecolors='blue', edgecolors='blue')
        #     ego_x_play = np.expand_dims(np.concatenate((ego_hist_x[:-1], ego_fut_x)), axis=-1)
        #     ego_y_play = np.expand_dims(np.concatenate((ego_hist_y[:-1], ego_fut_y)), axis=-1)
        #     ego_traj_play = np.expand_dims(np.concatenate((ego_x_play, ego_y_play), axis=-1), axis=0)
        #     target_x_play = np.expand_dims(np.concatenate((target_hist_x, pred_reg[:, 0].cpu())), axis=-1)
        #     target_y_play = np.expand_dims(np.concatenate((target_hist_y, pred_reg[:, 1].cpu())), axis=-1)
        #     target_traj_play = np.expand_dims(np.concatenate((target_x_play, target_y_play), axis=-1), axis=0)
        #     states = np.expand_dims(np.concatenate((ego_traj_play, target_traj_play), axis=0), axis=0)
        #     self.state_for_play = np.concatenate((self.state_for_play, states), axis=0)
        # if self.ego_path_enable_5.isChecked():
        #     self.pred_plot_5.canvas.ax.plot(ego_fut_x, ego_fut_y, '--', color='red')
        #     self.pred_plot_5.canvas.ax.scatter(ego_fut_x[-1], ego_fut_y[-1], marker_size, marker="o", facecolors='red', edgecolors='red')
        #     self.pred_plot_0.canvas.ax.plot(ego_fut_x, ego_fut_y, '--', color='red')
        #     self.pred_plot_0.canvas.ax.scatter(ego_fut_x[-1], ego_fut_y[-1], marker_size, marker="o", facecolors='red', edgecolors='red')
        #     pred_reg = self.pred_out_5[0]
        #     self.pred_plot_5.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
        #     self.pred_plot_5.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker="o", facecolors='blue', edgecolors='blue')
        #     self.pred_plot_0.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
        #     self.pred_plot_0.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker="o", facecolors='blue', edgecolors='blue')
        #     ego_x_play = np.expand_dims(np.concatenate((ego_hist_x[:-1], ego_fut_x)), axis=-1)
        #     ego_y_play = np.expand_dims(np.concatenate((ego_hist_y[:-1], ego_fut_y)), axis=-1)
        #     ego_traj_play = np.expand_dims(np.concatenate((ego_x_play, ego_y_play), axis=-1), axis=0)
        #     target_x_play = np.expand_dims(np.concatenate((target_hist_x, pred_reg[:, 0].cpu())), axis=-1)
        #     target_y_play = np.expand_dims(np.concatenate((target_hist_y, pred_reg[:, 1].cpu())), axis=-1)
        #     target_traj_play = np.expand_dims(np.concatenate((target_x_play, target_y_play), axis=-1), axis=0)
        #     states = np.expand_dims(np.concatenate((ego_traj_play, target_traj_play), axis=0), axis=0)
        #     self.state_for_play = np.concatenate((self.state_for_play, states), axis=0)
        #
        # self.pred_plot_0.canvas.ax.plot(target_fut_x, target_fut_y, ':', color='black')
        # self.pred_plot_0.canvas.ax.scatter(target_fut_x[-1], target_fut_y[-1], color='black')
        # if self.show_predict_1.isChecked():
        #     self.pred_plot_1.canvas.ax.plot(target_fut_x, target_fut_y, ':', color='black')
        #     self.pred_plot_1.canvas.ax.scatter(target_fut_x[-1], target_fut_y[-1], color='black')
        # if self.show_predict_2.isChecked():
        #     self.pred_plot_2.canvas.ax.plot(target_fut_x, target_fut_y, ':', color='black')
        #     self.pred_plot_2.canvas.ax.scatter(target_fut_x[-1], target_fut_y[-1], color='black')
        # if self.show_predict_3.isChecked():
        #     self.pred_plot_3.canvas.ax.plot(target_fut_x, target_fut_y, ':', color='black')
        #     self.pred_plot_3.canvas.ax.scatter(target_fut_x[-1], target_fut_y[-1], color='black')
        # if self.show_predict_4.isChecked():
        #     self.pred_plot_4.canvas.ax.plot(target_fut_x, target_fut_y, ':', color='black')
        #     self.pred_plot_4.canvas.ax.scatter(target_fut_x[-1], target_fut_y[-1], color='black')
        # if self.show_predict_5.isChecked():
        #     self.pred_plot_5.canvas.ax.plot(target_fut_x, target_fut_y, ':', color='black')
        #     self.pred_plot_5.canvas.ax.scatter(target_fut_x[-1], target_fut_y[-1], color='black')
        #
        # self.pred_plot_1.canvas.ax.set_xlim([xmin_1.item(), xmax_1.item()])
        # self.pred_plot_1.canvas.ax.set_ylim([ymin_1.item(), ymax_1.item()])
        # # self.pred_plot_1.canvas.ax.axis('equal')
        # self.pred_plot_1.canvas.draw()
        #
        # self.pred_plot_2.canvas.ax.set_xlim([xmin_2.item(), xmax_2.item()])
        # self.pred_plot_2.canvas.ax.set_ylim([ymin_2.item(), ymax_2.item()])
        # # self.pred_plot_2.canvas.ax.axis('equal')
        # self.pred_plot_2.canvas.draw()
        #
        # self.pred_plot_3.canvas.ax.set_xlim([xmin_3.item(), xmax_3.item()])
        # self.pred_plot_3.canvas.ax.set_ylim([ymin_3.item(), ymax_3.item()])
        # # self.pred_plot_3.canvas.ax.axis('equal')
        # self.pred_plot_3.canvas.draw()
        #
        # self.pred_plot_4.canvas.ax.set_xlim([xmin_4.item(), xmax_4.item()])
        # self.pred_plot_4.canvas.ax.set_ylim([ymin_4.item(), ymax_4.item()])
        # # self.pred_plot_4.canvas.ax.axis('equal')
        # self.pred_plot_4.canvas.draw()
        #
        # self.pred_plot_5.canvas.ax.set_xlim([xmin_5.item(), xmax_5.item()])
        # self.pred_plot_5.canvas.ax.set_ylim([ymin_5.item(), ymax_5.item()])
        # # self.pred_plot_5.canvas.ax.axis('equal')
        # self.pred_plot_5.canvas.draw()
        #
        # self.pred_plot_0.canvas.ax.set_xlim([xmin_0.item(), xmax_0.item()])
        # self.pred_plot_0.canvas.ax.set_ylim([ymin_0.item(), ymax_0.item()])
        # # self.pred_plot_5.canvas.ax.axis('equal')
        # self.pred_plot_0.canvas.draw()

    def get_eval_data_1(self):
        pred_err = torch.norm(self.pred_gt[0] - self.pred_out_1[0], dim=1)
        ade_pred = torch.mean(pred_err)
        fde_pred = pred_err[-1]

        recon_err = torch.norm(self.recon_gt - self.recon_out_1, dim=2)
        ade_recon = torch.mean(recon_err)
        fde_recon = torch.mean(recon_err[:, -1])
        return ade_pred, fde_pred, ade_recon, fde_recon

    def data_reform(self):
        bar = progressbar.ProgressBar(maxval=100, \
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        data_list = np.empty([len(self.afl), 50, 2])
        self.data_idx_map = []
        for i in range(len(self.afl)):
            bar.update(int(100*i/len(self.afl)))
            data_tmp = self.afl[i]
            target_id = data_tmp.seq_df.TRACK_ID[data_tmp.seq_df.OBJECT_TYPE == 'AGENT'].tolist()[0]
            target_traj = np.concatenate([np.expand_dims(data_tmp.seq_df.X[data_tmp.seq_df.TRACK_ID == target_id].to_numpy(), axis=-1),
                                         np.expand_dims(data_tmp.seq_df.Y[data_tmp.seq_df.TRACK_ID == target_id].to_numpy(), axis=-1)], axis=-1).astype(np.float32)
            data_list[i,:,:] = target_traj
            self.data_idx_map.append(data_tmp.current_seq.stem)
        bar.finish()
        target_traj_translation = preprocess(data_list)
        self.target_traj_converted = preprocess_dir(target_traj_translation)



def preprocess_dir(data):
    k = len(data)
    rotation_beta = np.empty([k])
    rot_data = np.empty([k, 50, 2])
    for i in range(k):
        dir = data[i, 20, :] - data[i, 19, :]
        rotation_beta[i] = np.arctan2(dir[1], dir[0])
        rotation_beta[i] = np.pi / 2. - rotation_beta[i]
        c, s = np.cos(rotation_beta[i]), np.sin(rotation_beta[i])
        R = np.array(((c, -s), (s, c)))
        for j in range(50):
            rot_data[i, j, :] = np.matmul(R, data[i, j, :])
    return rot_data

def preprocess(data):
    k = len(data)
    trans_data = np.empty([k, 50, 2])
    for i in range(k):
        median = data[i, 19, :].copy()
        for j in range(50):
            trans_data[i, j, :] = data[i, j, :] - median
    return trans_data

def sync(data):
    data_list = comm.allgather(data)
    data = dict()
    for key in data_list[0]:
        if isinstance(data_list[0][key], list):
            data[key] = []
        else:
            data[key] = 0
        for i in range(len(data_list)):
            data[key] += data_list[i][key]
    return data


def pred_metrics(preds, gt_preds, has_preds):
    # assert has_preds.all()
    preds = np.asarray(preds, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)

    """batch_size x num_mods x num_preds"""
    err = np.sqrt(((preds - np.expand_dims(gt_preds, 1)) ** 2).sum(3))

    ade1 = err[:, 0].mean()
    fde1 = err[:, 0, -1].mean()

    min_idcs = err[:, :, -1].argmin(1)
    row_idcs = np.arange(len(min_idcs)).astype(np.int64)
    err = err[row_idcs, min_idcs]
    ade = err.mean()
    fde = err[:, -1].mean()
    return ade1, fde1, ade, fde, min_idcs


def val(config, downstream_net, loss, post_process, epoch):
    epoch = 9999
    # config, Dataset, collate_fn, downstream_net, loss, post_process, opt = model.get_model(encoder_net, enc_config)
    dist_filtered_data = dict()
    dist_filtered_data['preprocess_val'] = currdir + '/LaneGCN/dataset/preprocess/val_crs_dist6_angle90_dist_filtering_down_sample_downsize.p'
    dist_filtered_data['preprocess'] = True
    hvd.init()
    dataset = Dataset(config["val_split"], dist_filtered_data, train=False)
    val_sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        sampler=val_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    downstream_net.eval()

    start_time = time.time()
    metrics = dict()
    for i, data in enumerate(val_loader):
        if i % 100 == 0:
            print(i)
        data = dict(data)
        with torch.no_grad():
            out, actors, actor_idcs, ego_fut_aug_idcs = downstream_net(data)
            loss_out, loss_orig = loss(out, actors, actor_idcs, ego_fut_aug_idcs, data)
            post_out = post_process(out, ego_fut_aug_idcs, data)
            post_process.append(metrics, loss_out, loss_orig, post_out)

    dt = time.time() - start_time
    post_process.display(metrics, dt, epoch)


def val_id(config, downstream_net, loss, post_process, epoch):
    epoch = 9999
    # config, Dataset, collate_fn, downstream_net, loss, post_process, opt = model.get_model(encoder_net, enc_config)
    dist_filtered_data = dict()
    dist_filtered_data['preprocess_val'] = currdir + '/LaneGCN/dataset/preprocess/val_crs_dist6_angle90_dist_filtering_downsize.p'
    dist_filtered_data['preprocess'] = True
    hvd.init()
    dataset = Dataset(config["val_split"], dist_filtered_data, train=False)
    val_sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        sampler=val_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    downstream_net.eval()

    start_time = time.time()
    metrics = dict()
    for i, data in enumerate(val_loader):
        if i % 100 == 0:
            print(i)

        data = dict(data)
        with torch.no_grad():
            out, ids, gts, idcs = downstream_net(data)
            ego_fut_aug_idcs, actor_idcs_mod, actor_ctrs_mod = idcs
            pred_out = torch.cat([out['reg'][i[0]][1:, 0, :, :] for i in ego_fut_aug_idcs])
            reconstruction_out = torch.cat([x.unsqueeze(dim=0) for x in out['reconstruction']])
            ids_hist, ids_fut = ids
            reconstruction_gt, pred_gt = gts
            loss_out, loss_orig = loss(pred_out, pred_gt, reconstruction_out, reconstruction_gt, ids_hist, ids_fut, ego_fut_aug_idcs, actor_idcs_mod)
            post_out = post_process(out, ego_fut_aug_idcs, data)
            post_process.append(metrics, loss_out, loss_orig, post_out)

    dt = time.time() - start_time
    post_process.display(metrics, dt, epoch)


app = QApplication(sys.argv)
main_dialog = MainDialog()
main_dialog.show()
app.exec_()

# root_dir = '/home/jhs/Desktop/SRFNet/LaneGCN/dataset/val/data/'
#
# from argoverse.map_representation.map_api import ArgoverseMap
# am = ArgoverseMap()
# from argoverse.utils.mpl_plotting_utils import draw_lane_polygons
#
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
#
# from argoverse.visualization.visualize_sequences import viz_sequence
# seq_path = f"{root_dir}/2645.csv"
# viz_sequence(afl.get(seq_path).seq_df, show=True)
# xmin = 500
# xmax = 700
# ymin = 500
# ymax = 700
# city_name = 'MIA'
# local_das = am.find_local_driveable_areas([xmin, xmax, ymin, ymax], city_name)
#
#
# local_lane_polygons = am.find_local_lane_polygons([xmin, xmax, ymin, ymax], city_name)
# local_das = am.find_local_driveable_areas([xmin, xmax, ymin, ymax], city_name)
#
# domv = DatasetOnMapVisualizer(dataset_dir, experiment_prefix, use_existing_files=use_existing_files, log_id=argoverse_data.current_log)
