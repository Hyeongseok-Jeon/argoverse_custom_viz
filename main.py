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
        self.data_index = int(self.data[row_index, 0, 0, -1])
        self.cur_data = self.afl.get(self.original_data_dir + '/' + str(self.data_index) + '.csv')
        self.idx_data.setText(str(row_index))
        self.data_dir_data.setText(str(self.cur_data.current_seq))
        ego_id = self.cur_data.seq_df.TRACK_ID[self.cur_data.seq_df.OBJECT_TYPE == 'AV'].tolist()[0]
        target_id = self.cur_data.seq_df.TRACK_ID[self.cur_data.seq_df.OBJECT_TYPE == 'AGENT'].tolist()[0]

        self.ego_hist = np.concatenate([np.expand_dims(self.cur_data.seq_df.X[self.cur_data.seq_df.TRACK_ID == ego_id][:20].to_numpy(), axis=-1),
                                        np.expand_dims(self.cur_data.seq_df.Y[self.cur_data.seq_df.TRACK_ID == ego_id][:20].to_numpy(), axis=-1)], axis=-1).astype(np.float32)
        self.ego_fut = np.concatenate([np.expand_dims(self.cur_data.seq_df.X[self.cur_data.seq_df.TRACK_ID == ego_id][21:].to_numpy(), axis=-1),
                                       np.expand_dims(self.cur_data.seq_df.Y[self.cur_data.seq_df.TRACK_ID == ego_id][21:].to_numpy(), axis=-1)], axis=-1).astype(np.float32)
        self.target_hist = np.concatenate([np.expand_dims(self.cur_data.seq_df.X[self.cur_data.seq_df.TRACK_ID == target_id][:20].to_numpy(), axis=-1),
                                           np.expand_dims(self.cur_data.seq_df.Y[self.cur_data.seq_df.TRACK_ID == target_id][:20].to_numpy(), axis=-1)], axis=-1).astype(np.float32)
        self.target_fut = np.concatenate([np.expand_dims(self.cur_data.seq_df.X[self.cur_data.seq_df.TRACK_ID == target_id][21:].to_numpy(), axis=-1),
                                          np.expand_dims(self.cur_data.seq_df.Y[self.cur_data.seq_df.TRACK_ID == target_id][21:].to_numpy(), axis=-1)], axis=-1).astype(np.float32)
        self.GT = self.data[row_index, 1, :, :2]
        self.prediction = self.data[row_index, 0, :, :2]
        self.city = self.cur_data.seq_df.CITY_NAME[0]

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
        target_index = self.data_dir_data.text()
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


    def update_data(self):
        self.map_data_1.setText(self.cur_data['city'][0])
        self.map_data_2.setText(self.cur_data['city'][0])
        self.map_data_3.setText(self.cur_data['city'][0])
        self.map_data_4.setText(self.cur_data['city'][0])
        self.map_data_5.setText(self.cur_data['city'][0])
        self.map_data_0.setText(self.cur_data['city'][0])
        self.num_of_vehicles_data_1.setText(str(self.cur_data['gt_preds'][0].shape[0]))
        self.num_of_vehicles_data_2.setText(str(self.cur_data['gt_preds'][0].shape[0]))
        self.num_of_vehicles_data_3.setText(str(self.cur_data['gt_preds'][0].shape[0]))
        self.num_of_vehicles_data_4.setText(str(self.cur_data['gt_preds'][0].shape[0]))
        self.num_of_vehicles_data_5.setText(str(self.cur_data['gt_preds'][0].shape[0]))
        self.num_of_vehicles_data_0.setText(str(self.cur_data['gt_preds'][0].shape[0]))
        self.idx_data_1.setText(str(self.cur_data['idx'][0]))
        self.idx_data_2.setText(str(self.cur_data['idx'][0]))
        self.idx_data_3.setText(str(self.cur_data['idx'][0]))
        self.idx_data_4.setText(str(self.cur_data['idx'][0]))
        self.idx_data_5.setText(str(self.cur_data['idx'][0]))
        self.idx_data_0.setText(str(self.cur_data['idx'][0]))

        ego_aug = self.cur_data['ego_aug'][0]['traj']
        self.pred_gt = self.cur_data['gt_preds'][0][1:2]
        self.recon_gt = torch.cat([self.cur_data['gt_preds'][0][0:1, :, :], ego_aug])
        self.show_predict_1.setChecked(True)
        self.show_predict_2.setChecked(True)
        self.show_predict_3.setChecked(True)
        self.show_predict_4.setChecked(True)
        self.show_predict_5.setChecked(True)
        with torch.no_grad():
            out1 = self.net_1(self.cur_data)
            out2 = self.net_2(self.cur_data)
            out3 = self.net_3(self.cur_data)
            out4 = self.net_4(self.cur_data)
            out5 = self.net_5(self.cur_data)
        out5 = self.out_mod(out5)
        self.pred_out_1 = torch.cat([out1[0]['reg'][i][1:2, 0, :, :].cpu() for i in range(len(out1[0]['reg']))])
        self.pred_out_2 = torch.cat([out2[0]['reg'][i][1:2, 0, :, :].cpu() for i in range(len(out2[0]['reg']))])
        self.pred_out_3 = torch.cat([out3[0]['reg'][i][1:2, 0, :, :].cpu() for i in range(len(out3[0]['reg']))])
        self.pred_out_4 = torch.cat([out4[0]['reg'][i][1:2, 0, :, :].cpu() for i in range(len(out4[0]['reg']))])
        self.pred_out_5 = torch.cat([out5[0]['reg'][i][1:2, 0, :, :].cpu() for i in range(len(out5[0]['reg']))])
        self.recon_out_1 = torch.cat([out1[0]['reconstruction'][i].unsqueeze(0).cpu() for i in range(len(out1[0]['reconstruction']))])
        self.recon_out_2 = torch.cat([out2[0]['reconstruction'][i].unsqueeze(0).cpu() for i in range(len(out2[0]['reconstruction']))])
        self.recon_out_3 = torch.cat([out3[0]['reconstruction'][i].unsqueeze(0).cpu() for i in range(len(out3[0]['reconstruction']))])
        self.recon_out_4 = torch.cat([out4[0]['reconstruction'][i].unsqueeze(0).cpu() for i in range(len(out4[0]['reconstruction']))])
        self.recon_out_5 = torch.cat([out5[0]['reconstruction'][i].unsqueeze(0).cpu() for i in range(len(out5[0]['reconstruction']))])

        for i in range(9):
            if i < self.pred_out_1.shape[0]:
                self.cand_toggles_1[i].setEnabled(True)
                self.cand_toggles_1[i].setChecked(True)
                self.cand_toggles_2[i].setEnabled(True)
                self.cand_toggles_2[i].setChecked(True)
                self.cand_toggles_3[i].setEnabled(True)
                self.cand_toggles_3[i].setChecked(True)
                self.cand_toggles_4[i].setEnabled(True)
                self.cand_toggles_4[i].setChecked(True)
                self.cand_toggles_5[i].setEnabled(True)
                self.cand_toggles_5[i].setChecked(True)
            else:
                self.cand_toggles_1[i].setEnabled(False)
                self.cand_toggles_1[i].setChecked(False)
                self.cand_toggles_2[i].setEnabled(False)
                self.cand_toggles_2[i].setChecked(False)
                self.cand_toggles_3[i].setEnabled(False)
                self.cand_toggles_3[i].setChecked(False)
                self.cand_toggles_4[i].setEnabled(False)
                self.cand_toggles_4[i].setChecked(False)
                self.cand_toggles_5[i].setEnabled(False)
                self.cand_toggles_5[i].setChecked(False)

        ade_pred_1, fde_pred_1, ade_recon_1, fde_recon_1 = self.get_eval_data_1()
        ade_pred_2, fde_pred_2, ade_recon_2, fde_recon_2 = self.get_eval_data_2()
        ade_pred_3, fde_pred_3, ade_recon_3, fde_recon_3 = self.get_eval_data_3()
        ade_pred_4, fde_pred_4, ade_recon_4, fde_recon_4 = self.get_eval_data_4()
        ade_pred_5, fde_pred_5, ade_recon_5, fde_recon_5 = self.get_eval_data_5()

        self.ade_pred_1.setText(str(ade_pred_1.item())[:5])
        self.fde_pred_1.setText(str(fde_pred_1.item())[:5])
        self.ade_recon_1.setText(str(ade_recon_1.item())[:5])
        self.fde_recon_1.setText(str(fde_recon_1.item())[:5])
        self.ade_pred_2.setText(str(ade_pred_2.item())[:5])
        self.fde_pred_2.setText(str(fde_pred_2.item())[:5])
        self.ade_recon_2.setText(str(ade_recon_2.item())[:5])
        self.fde_recon_2.setText(str(fde_recon_2.item())[:5])
        self.ade_pred_3.setText(str(ade_pred_3.item())[:5])
        self.fde_pred_3.setText(str(fde_pred_3.item())[:5])
        self.ade_recon_3.setText(str(ade_recon_3.item())[:5])
        self.fde_recon_3.setText(str(fde_recon_3.item())[:5])
        self.ade_pred_4.setText(str(ade_pred_4.item())[:5])
        self.fde_pred_4.setText(str(fde_pred_4.item())[:5])
        self.ade_recon_4.setText(str(ade_recon_4.item())[:5])
        self.fde_recon_4.setText(str(fde_recon_4.item())[:5])
        self.ade_pred_5.setText(str(ade_pred_5.item())[:5])
        self.fde_pred_5.setText(str(fde_pred_5.item())[:5])
        self.ade_recon_5.setText(str(ade_recon_5.item())[:5])
        self.fde_recon_5.setText(str(fde_recon_5.item())[:5])
        # self.update_table()
        self.visualization()

    def out_mod(self, out):
        ele1 = out[0]
        ele2 = out[1]
        ele3 = out[2]
        ele4 = out[3]
        # self.pred_out_5 = torch.cat([out5[0]['reg'][i][1:2, 0, :, :].cpu() for i in range(len(out5[0]['reg']))])

        target = ele1['reg'][0][1:2, 0, 0, :]
        for i in range(len(ele1['reg'])):
            displacement = target - ele1['reg'][i][1:2, 0, 0, :]
            ele1['reg'][i][1:2, 0, :, :] = ele1['reg'][i][1:2, 0, :, :] + displacement
        out_mod = (ele1, ele2, ele3, ele4)
        return out_mod

    def update_table(self):
        self.tableWidget.clear()
        for i in range(4):
            if i < len(self.pred_out):
                pred_out = self.pred_out[i]
                best_idx = np.argmax(pred_out['cls'][0][0].cpu().detach().numpy())
                pred_reg = pred_out['reg'][0][0, best_idx, :, :]
                x = pred_reg[:, 0].cpu()
                y = pred_reg[:, 1].cpu()
                for j in range(30):
                    self.tableWidget.setItem(j, 2 * i, QTableWidgetItem(str(x[j].item())))
                    self.tableWidget.setItem(j, 2 * i + 1, QTableWidgetItem(str(y[j].item())))

    def visualization(self):
        self.pred_plot_1.canvas.ax.clear()
        self.pred_plot_2.canvas.ax.clear()
        self.pred_plot_3.canvas.ax.clear()
        self.pred_plot_4.canvas.ax.clear()
        self.pred_plot_5.canvas.ax.clear()
        self.pred_plot_0.canvas.ax.clear()
        ego_cur_pos = self.cur_data['gt_preds'][0][0, 0, :]
        xmin_1 = ego_cur_pos[0] - self.fov_1 + self.x_offset_1
        xmax_1 = ego_cur_pos[0] + self.fov_1 + self.x_offset_1
        ymin_1 = ego_cur_pos[1] - self.fov_1 + self.y_offset_1
        ymax_1 = ego_cur_pos[1] + self.fov_1 + self.y_offset_1
        xmin_2 = ego_cur_pos[0] - self.fov_2 + self.x_offset_2
        xmax_2 = ego_cur_pos[0] + self.fov_2 + self.x_offset_2
        ymin_2 = ego_cur_pos[1] - self.fov_2 + self.y_offset_2
        ymax_2 = ego_cur_pos[1] + self.fov_2 + self.y_offset_2
        xmin_3 = ego_cur_pos[0] - self.fov_3 + self.x_offset_3
        xmax_3 = ego_cur_pos[0] + self.fov_3 + self.x_offset_3
        ymin_3 = ego_cur_pos[1] - self.fov_3 + self.y_offset_3
        ymax_3 = ego_cur_pos[1] + self.fov_3 + self.y_offset_3
        xmin_4 = ego_cur_pos[0] - self.fov_4 + self.x_offset_4
        xmax_4 = ego_cur_pos[0] + self.fov_4 + self.x_offset_4
        ymin_4 = ego_cur_pos[1] - self.fov_4 + self.y_offset_4
        ymax_4 = ego_cur_pos[1] + self.fov_4 + self.y_offset_4
        xmin_5 = ego_cur_pos[0] - self.fov_5 + self.x_offset_5
        xmax_5 = ego_cur_pos[0] + self.fov_5 + self.x_offset_5
        ymin_5 = ego_cur_pos[1] - self.fov_5 + self.y_offset_5
        ymax_5 = ego_cur_pos[1] + self.fov_5 + self.y_offset_5
        xmin_0 = ego_cur_pos[0] - self.fov_0 + self.x_offset_0
        xmax_0 = ego_cur_pos[0] + self.fov_0 + self.x_offset_0
        ymin_0 = ego_cur_pos[1] - self.fov_0 + self.y_offset_0
        ymax_0 = ego_cur_pos[1] + self.fov_0 + self.y_offset_0
        city_name = self.cur_data['city'][0]
        local_lane_polygons_1 = am.find_local_lane_polygons([xmin_1, xmax_1, ymin_1, ymax_1], city_name)
        local_lane_polygons_2 = am.find_local_lane_polygons([xmin_2, xmax_2, ymin_2, ymax_2], city_name)
        local_lane_polygons_3 = am.find_local_lane_polygons([xmin_3, xmax_3, ymin_3, ymax_3], city_name)
        local_lane_polygons_4 = am.find_local_lane_polygons([xmin_4, xmax_4, ymin_4, ymax_4], city_name)
        local_lane_polygons_5 = am.find_local_lane_polygons([xmin_5, xmax_5, ymin_5, ymax_5], city_name)
        local_lane_polygons_0 = am.find_local_lane_polygons([xmin_0, xmax_0, ymin_0, ymax_0], city_name)
        draw_lane_polygons(self.pred_plot_1.canvas.ax, local_lane_polygons_1, color='darkgray')
        draw_lane_polygons(self.pred_plot_2.canvas.ax, local_lane_polygons_2, color='darkgray')
        draw_lane_polygons(self.pred_plot_3.canvas.ax, local_lane_polygons_3, color='darkgray')
        draw_lane_polygons(self.pred_plot_4.canvas.ax, local_lane_polygons_4, color='darkgray')
        draw_lane_polygons(self.pred_plot_5.canvas.ax, local_lane_polygons_5, color='darkgray')
        draw_lane_polygons(self.pred_plot_0.canvas.ax, local_lane_polygons_0, color='darkgray')

        raw_data = []
        with open(self.data_dir + self.cur_data['file_name'][0], newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                raw_data.append(row)
        raw_data = raw_data[1:]
        x = np.asarray([float(raw_data[i][0].split(',')[3]) for i in range(len(raw_data))])
        y = np.asarray([float(raw_data[i][0].split(',')[4]) for i in range(len(raw_data))])
        veh_class = [raw_data[i][0].split(',')[2] for i in range(len(raw_data))]
        time_stamp = [raw_data[i][0].split(',')[0] for i in range(len(raw_data))]
        ego_index = [i for i, x in enumerate(veh_class) if x == 'AV']
        ego_x = x[ego_index]
        ego_y = y[ego_index]
        ego_hist_x = ego_x[:20]
        ego_hist_y = ego_y[:20]
        ego_fut_x = ego_x[19:40]
        ego_fut_y = ego_y[19:40]
        target_index = [i for i, x in enumerate(veh_class) if x == 'AGENT']
        target_x = x[target_index]
        target_y = y[target_index]
        target_hist_x = target_x[:20]
        target_hist_y = target_y[:20]
        target_fut_x = target_x[19:40]
        target_fut_y = target_y[19:40]
        cur_time = raw_data[ego_index[19]][0].split(',')[0]
        cur_sur_index = [i for i, x in enumerate(time_stamp) if x == cur_time]
        sur_x = x[cur_sur_index]
        sur_y = y[cur_sur_index]
        self.pred_plot_1.canvas.ax.scatter(sur_x, sur_y, color='silver')
        self.pred_plot_2.canvas.ax.scatter(sur_x, sur_y, color='silver')
        self.pred_plot_3.canvas.ax.scatter(sur_x, sur_y, color='silver')
        self.pred_plot_4.canvas.ax.scatter(sur_x, sur_y, color='silver')
        self.pred_plot_5.canvas.ax.scatter(sur_x, sur_y, color='silver')
        self.pred_plot_0.canvas.ax.scatter(sur_x, sur_y, color='silver')

        self.pred_plot_1.canvas.ax.plot(ego_hist_x, ego_hist_y, '-', color='red')
        self.pred_plot_1.canvas.ax.scatter(ego_hist_x[-1], ego_hist_y[-1], color='red')
        self.pred_plot_1.canvas.ax.plot(target_hist_x, target_hist_y, '-', color='blue')
        self.pred_plot_1.canvas.ax.scatter(target_hist_x[-1], target_hist_y[-1], color='blue')
        self.pred_plot_2.canvas.ax.plot(ego_hist_x, ego_hist_y, '-', color='red')
        self.pred_plot_2.canvas.ax.scatter(ego_hist_x[-1], ego_hist_y[-1], color='red')
        self.pred_plot_2.canvas.ax.plot(target_hist_x, target_hist_y, '-', color='blue')
        self.pred_plot_2.canvas.ax.scatter(target_hist_x[-1], target_hist_y[-1], color='blue')
        self.pred_plot_3.canvas.ax.plot(ego_hist_x, ego_hist_y, '-', color='red')
        self.pred_plot_3.canvas.ax.scatter(ego_hist_x[-1], ego_hist_y[-1], color='red')
        self.pred_plot_3.canvas.ax.plot(target_hist_x, target_hist_y, '-', color='blue')
        self.pred_plot_3.canvas.ax.scatter(target_hist_x[-1], target_hist_y[-1], color='blue')
        self.pred_plot_4.canvas.ax.plot(ego_hist_x, ego_hist_y, '-', color='red')
        self.pred_plot_4.canvas.ax.scatter(ego_hist_x[-1], ego_hist_y[-1], color='red')
        self.pred_plot_4.canvas.ax.plot(target_hist_x, target_hist_y, '-', color='blue')
        self.pred_plot_4.canvas.ax.scatter(target_hist_x[-1], target_hist_y[-1], color='blue')
        self.pred_plot_5.canvas.ax.plot(ego_hist_x, ego_hist_y, '-', color='red')
        self.pred_plot_5.canvas.ax.scatter(ego_hist_x[-1], ego_hist_y[-1], color='red')
        self.pred_plot_5.canvas.ax.plot(target_hist_x, target_hist_y, '-', color='blue')
        self.pred_plot_5.canvas.ax.scatter(target_hist_x[-1], target_hist_y[-1], color='blue')
        self.pred_plot_0.canvas.ax.plot(ego_hist_x, ego_hist_y, '-', color='red')
        self.pred_plot_0.canvas.ax.scatter(ego_hist_x[-1], ego_hist_y[-1], color='red')
        self.pred_plot_0.canvas.ax.plot(target_hist_x, target_hist_y, '-', color='blue')
        self.pred_plot_0.canvas.ax.scatter(target_hist_x[-1], target_hist_y[-1], color='blue')

        ego_aug = self.cur_data['ego_aug'][0]['traj'].numpy().copy()
        ego_aug = np.concatenate([ego_aug, np.zeros_like(ego_aug[:, 0:1, :])], axis=1)
        marker_size = 50
        self.state_for_play = np.zeros(shape=(0, 2, 40, 2))

        for i in range(ego_aug.shape[0]):
            ego_aug[i, :, :] = np.concatenate([np.expand_dims(np.asarray([ego_hist_x[-1], ego_hist_y[-1]]), axis=0), ego_aug[i, :20, :]], axis=0)
            ego_facecolors = 'none'
            sur_facecolors = 'none'
            if i == 0:
                marker_shape = 's'
            elif i == 1:
                marker_shape = 'P'
            elif i == 2:
                marker_shape = '^'
            elif i == 3:
                marker_shape = '*'
            elif i == 4:
                marker_shape = 's'
            elif i == 5:
                marker_shape = 's'
            elif i == 6:
                marker_shape = 's'
            if self.cand_toggles_1[i + 1].isChecked():
                aug_x = ego_aug[i, :, 0]
                aug_y = ego_aug[i, :, 1]
                self.pred_plot_1.canvas.ax.plot(aug_x, aug_y, '--', color='red')
                self.pred_plot_1.canvas.ax.scatter(aug_x[-1], aug_y[-1], marker_size, marker=marker_shape, facecolors=ego_facecolors, edgecolors='red')
                self.pred_plot_0.canvas.ax.plot(aug_x, aug_y, '--', color='red')
                self.pred_plot_0.canvas.ax.scatter(aug_x[-1], aug_y[-1], marker_size, marker=marker_shape, facecolors=ego_facecolors, edgecolors='red')
                pred_reg = self.pred_out_1[i + 1]
                self.pred_plot_1.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
                self.pred_plot_1.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker=marker_shape, facecolors=sur_facecolors, edgecolors='blue')
                self.pred_plot_0.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
                self.pred_plot_0.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker=marker_shape, facecolors=sur_facecolors, edgecolors='blue')
                ego_x_play = np.expand_dims(np.concatenate((ego_hist_x[:-1], aug_x)), axis=-1)
                ego_y_play = np.expand_dims(np.concatenate((ego_hist_y[:-1], aug_y)), axis=-1)
                ego_traj_play = np.expand_dims(np.concatenate((ego_x_play, ego_y_play), axis=-1), axis=0)
                target_x_play = np.expand_dims(np.concatenate((target_hist_x, pred_reg[:, 0].cpu())), axis=-1)
                target_y_play = np.expand_dims(np.concatenate((target_hist_y, pred_reg[:, 1].cpu())), axis=-1)
                target_traj_play = np.expand_dims(np.concatenate((target_x_play, target_y_play), axis=-1), axis=0)
                states = np.expand_dims(np.concatenate((ego_traj_play, target_traj_play), axis=0), axis=0)
                self.state_for_play = np.concatenate((self.state_for_play, states), axis=0)
            if self.cand_toggles_2[i + 1].isChecked():
                aug_x = ego_aug[i, :, 0]
                aug_y = ego_aug[i, :, 1]
                self.pred_plot_2.canvas.ax.plot(aug_x, aug_y, '--', color='red')
                self.pred_plot_2.canvas.ax.scatter(aug_x[-1], aug_y[-1], marker_size, marker=marker_shape, facecolors=ego_facecolors, edgecolors='red')
                self.pred_plot_0.canvas.ax.plot(aug_x, aug_y, '--', color='red')
                self.pred_plot_0.canvas.ax.scatter(aug_x[-1], aug_y[-1], marker_size, marker=marker_shape, facecolors=ego_facecolors, edgecolors='red')
                pred_reg = self.pred_out_2[i + 1]
                self.pred_plot_2.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
                self.pred_plot_2.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker=marker_shape, facecolors=sur_facecolors, edgecolors='blue')
                self.pred_plot_0.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
                self.pred_plot_0.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker=marker_shape, facecolors=sur_facecolors, edgecolors='blue')
                ego_x_play = np.expand_dims(np.concatenate((ego_hist_x[:-1], aug_x)), axis=-1)
                ego_y_play = np.expand_dims(np.concatenate((ego_hist_y[:-1], aug_y)), axis=-1)
                ego_traj_play = np.expand_dims(np.concatenate((ego_x_play, ego_y_play), axis=-1), axis=0)
                target_x_play = np.expand_dims(np.concatenate((target_hist_x, pred_reg[:, 0].cpu())), axis=-1)
                target_y_play = np.expand_dims(np.concatenate((target_hist_y, pred_reg[:, 1].cpu())), axis=-1)
                target_traj_play = np.expand_dims(np.concatenate((target_x_play, target_y_play), axis=-1), axis=0)
                states = np.expand_dims(np.concatenate((ego_traj_play, target_traj_play), axis=0), axis=0)
                self.state_for_play = np.concatenate((self.state_for_play, states), axis=0)
            if self.cand_toggles_3[i + 1].isChecked():
                aug_x = ego_aug[i, :, 0]
                aug_y = ego_aug[i, :, 1]
                self.pred_plot_3.canvas.ax.plot(aug_x, aug_y, '--', color='red')
                self.pred_plot_3.canvas.ax.scatter(aug_x[-1], aug_y[-1], marker_size, marker=marker_shape, facecolors=ego_facecolors, edgecolors='red')
                self.pred_plot_0.canvas.ax.plot(aug_x, aug_y, '--', color='red')
                self.pred_plot_0.canvas.ax.scatter(aug_x[-1], aug_y[-1], marker_size, marker=marker_shape, facecolors=ego_facecolors, edgecolors='red')
                pred_reg = self.pred_out_3[i + 1]
                self.pred_plot_3.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
                self.pred_plot_3.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker=marker_shape, facecolors=sur_facecolors, edgecolors='blue')
                self.pred_plot_0.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
                self.pred_plot_0.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker=marker_shape, facecolors=sur_facecolors, edgecolors='blue')
                ego_x_play = np.expand_dims(np.concatenate((ego_hist_x[:-1], aug_x)), axis=-1)
                ego_y_play = np.expand_dims(np.concatenate((ego_hist_y[:-1], aug_y)), axis=-1)
                ego_traj_play = np.expand_dims(np.concatenate((ego_x_play, ego_y_play), axis=-1), axis=0)
                target_x_play = np.expand_dims(np.concatenate((target_hist_x, pred_reg[:, 0].cpu())), axis=-1)
                target_y_play = np.expand_dims(np.concatenate((target_hist_y, pred_reg[:, 1].cpu())), axis=-1)
                target_traj_play = np.expand_dims(np.concatenate((target_x_play, target_y_play), axis=-1), axis=0)
                states = np.expand_dims(np.concatenate((ego_traj_play, target_traj_play), axis=0), axis=0)
                self.state_for_play = np.concatenate((self.state_for_play, states), axis=0)
            if self.cand_toggles_4[i + 1].isChecked():
                aug_x = ego_aug[i, :, 0]
                aug_y = ego_aug[i, :, 1]
                self.pred_plot_4.canvas.ax.plot(aug_x, aug_y, '--', color='red')
                self.pred_plot_4.canvas.ax.scatter(aug_x[-1], aug_y[-1], marker_size, marker=marker_shape, facecolors=ego_facecolors, edgecolors='red')
                self.pred_plot_0.canvas.ax.plot(aug_x, aug_y, '--', color='red')
                self.pred_plot_0.canvas.ax.scatter(aug_x[-1], aug_y[-1], marker_size, marker=marker_shape, facecolors=ego_facecolors, edgecolors='red')
                pred_reg = self.pred_out_4[i + 1]
                self.pred_plot_4.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
                self.pred_plot_4.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker=marker_shape, facecolors=sur_facecolors, edgecolors='blue')
                self.pred_plot_0.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
                self.pred_plot_0.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker=marker_shape, facecolors=sur_facecolors, edgecolors='blue')
                ego_x_play = np.expand_dims(np.concatenate((ego_hist_x[:-1], aug_x)), axis=-1)
                ego_y_play = np.expand_dims(np.concatenate((ego_hist_y[:-1], aug_y)), axis=-1)
                ego_traj_play = np.expand_dims(np.concatenate((ego_x_play, ego_y_play), axis=-1), axis=0)
                target_x_play = np.expand_dims(np.concatenate((target_hist_x, pred_reg[:, 0].cpu())), axis=-1)
                target_y_play = np.expand_dims(np.concatenate((target_hist_y, pred_reg[:, 1].cpu())), axis=-1)
                target_traj_play = np.expand_dims(np.concatenate((target_x_play, target_y_play), axis=-1), axis=0)
                states = np.expand_dims(np.concatenate((ego_traj_play, target_traj_play), axis=0), axis=0)
                self.state_for_play = np.concatenate((self.state_for_play, states), axis=0)
            if self.cand_toggles_5[i + 1].isChecked():
                aug_x = ego_aug[i, :, 0]
                aug_y = ego_aug[i, :, 1]
                self.pred_plot_5.canvas.ax.plot(aug_x, aug_y, '--', color='red')
                self.pred_plot_5.canvas.ax.scatter(aug_x[-1], aug_y[-1], marker_size, marker=marker_shape, facecolors=ego_facecolors, edgecolors='red')
                self.pred_plot_0.canvas.ax.plot(aug_x, aug_y, '--', color='red')
                self.pred_plot_0.canvas.ax.scatter(aug_x[-1], aug_y[-1], marker_size, marker=marker_shape, facecolors=ego_facecolors, edgecolors='red')
                pred_reg = self.pred_out_5[i + 1]
                self.pred_plot_5.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
                self.pred_plot_5.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker=marker_shape, facecolors=sur_facecolors, edgecolors='blue')
                self.pred_plot_0.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
                self.pred_plot_0.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker=marker_shape, facecolors=sur_facecolors, edgecolors='blue')
                ego_x_play = np.expand_dims(np.concatenate((ego_hist_x[:-1], aug_x)), axis=-1)
                ego_y_play = np.expand_dims(np.concatenate((ego_hist_y[:-1], aug_y)), axis=-1)
                ego_traj_play = np.expand_dims(np.concatenate((ego_x_play, ego_y_play), axis=-1), axis=0)
                target_x_play = np.expand_dims(np.concatenate((target_hist_x, pred_reg[:, 0].cpu())), axis=-1)
                target_y_play = np.expand_dims(np.concatenate((target_hist_y, pred_reg[:, 1].cpu())), axis=-1)
                target_traj_play = np.expand_dims(np.concatenate((target_x_play, target_y_play), axis=-1), axis=0)
                states = np.expand_dims(np.concatenate((ego_traj_play, target_traj_play), axis=0), axis=0)
                self.state_for_play = np.concatenate((self.state_for_play, states), axis=0)

        if self.ego_path_enable_1.isChecked():
            self.pred_plot_1.canvas.ax.plot(ego_fut_x, ego_fut_y, '--', color='red')
            self.pred_plot_1.canvas.ax.scatter(ego_fut_x[-1], ego_fut_y[-1], marker_size, marker="o", facecolors='red', edgecolors='red')
            self.pred_plot_0.canvas.ax.plot(ego_fut_x, ego_fut_y, '--', color='red')
            self.pred_plot_0.canvas.ax.scatter(ego_fut_x[-1], ego_fut_y[-1], marker_size, marker="o", facecolors='red', edgecolors='red')
            pred_reg = self.pred_out_1[0]
            self.pred_plot_1.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
            self.pred_plot_1.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker="o", facecolors='blue', edgecolors='blue')
            self.pred_plot_0.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
            self.pred_plot_0.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker="o", facecolors='blue', edgecolors='blue')
            ego_x_play = np.expand_dims(np.concatenate((ego_hist_x[:-1], ego_fut_x)), axis=-1)
            ego_y_play = np.expand_dims(np.concatenate((ego_hist_y[:-1], ego_fut_y)), axis=-1)
            ego_traj_play = np.expand_dims(np.concatenate((ego_x_play, ego_y_play), axis=-1), axis=0)
            target_x_play = np.expand_dims(np.concatenate((target_hist_x, pred_reg[:, 0].cpu())), axis=-1)
            target_y_play = np.expand_dims(np.concatenate((target_hist_y, pred_reg[:, 1].cpu())), axis=-1)
            target_traj_play = np.expand_dims(np.concatenate((target_x_play, target_y_play), axis=-1), axis=0)
            states = np.expand_dims(np.concatenate((ego_traj_play, target_traj_play), axis=0), axis=0)
            self.state_for_play = np.concatenate((self.state_for_play, states), axis=0)
        if self.ego_path_enable_2.isChecked():
            self.pred_plot_2.canvas.ax.plot(ego_fut_x, ego_fut_y, '--', color='red')
            self.pred_plot_2.canvas.ax.scatter(ego_fut_x[-1], ego_fut_y[-1], marker_size, marker="o", facecolors='red', edgecolors='red')
            self.pred_plot_0.canvas.ax.plot(ego_fut_x, ego_fut_y, '--', color='red')
            self.pred_plot_0.canvas.ax.scatter(ego_fut_x[-1], ego_fut_y[-1], marker_size, marker="o", facecolors='red', edgecolors='red')
            pred_reg = self.pred_out_2[0]
            self.pred_plot_2.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
            self.pred_plot_2.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker="o", facecolors='blue', edgecolors='blue')
            self.pred_plot_0.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
            self.pred_plot_0.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker="o", facecolors='blue', edgecolors='blue')
            ego_x_play = np.expand_dims(np.concatenate((ego_hist_x[:-1], ego_fut_x)), axis=-1)
            ego_y_play = np.expand_dims(np.concatenate((ego_hist_y[:-1], ego_fut_y)), axis=-1)
            ego_traj_play = np.expand_dims(np.concatenate((ego_x_play, ego_y_play), axis=-1), axis=0)
            target_x_play = np.expand_dims(np.concatenate((target_hist_x, pred_reg[:, 0].cpu())), axis=-1)
            target_y_play = np.expand_dims(np.concatenate((target_hist_y, pred_reg[:, 1].cpu())), axis=-1)
            target_traj_play = np.expand_dims(np.concatenate((target_x_play, target_y_play), axis=-1), axis=0)
            states = np.expand_dims(np.concatenate((ego_traj_play, target_traj_play), axis=0), axis=0)
            self.state_for_play = np.concatenate((self.state_for_play, states), axis=0)
        if self.ego_path_enable_3.isChecked():
            self.pred_plot_3.canvas.ax.plot(ego_fut_x, ego_fut_y, '--', color='red')
            self.pred_plot_3.canvas.ax.scatter(ego_fut_x[-1], ego_fut_y[-1], marker_size, marker="o", facecolors='red', edgecolors='red')
            self.pred_plot_0.canvas.ax.plot(ego_fut_x, ego_fut_y, '--', color='red')
            self.pred_plot_0.canvas.ax.scatter(ego_fut_x[-1], ego_fut_y[-1], marker_size, marker="o", facecolors='red', edgecolors='red')
            pred_reg = self.pred_out_3[0]
            self.pred_plot_3.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
            self.pred_plot_3.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker="o", facecolors='blue', edgecolors='blue')
            self.pred_plot_0.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
            self.pred_plot_0.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker="o", facecolors='blue', edgecolors='blue')
            ego_x_play = np.expand_dims(np.concatenate((ego_hist_x[:-1], ego_fut_x)), axis=-1)
            ego_y_play = np.expand_dims(np.concatenate((ego_hist_y[:-1], ego_fut_y)), axis=-1)
            ego_traj_play = np.expand_dims(np.concatenate((ego_x_play, ego_y_play), axis=-1), axis=0)
            target_x_play = np.expand_dims(np.concatenate((target_hist_x, pred_reg[:, 0].cpu())), axis=-1)
            target_y_play = np.expand_dims(np.concatenate((target_hist_y, pred_reg[:, 1].cpu())), axis=-1)
            target_traj_play = np.expand_dims(np.concatenate((target_x_play, target_y_play), axis=-1), axis=0)
            states = np.expand_dims(np.concatenate((ego_traj_play, target_traj_play), axis=0), axis=0)
            self.state_for_play = np.concatenate((self.state_for_play, states), axis=0)
        if self.ego_path_enable_4.isChecked():
            self.pred_plot_4.canvas.ax.plot(ego_fut_x, ego_fut_y, '--', color='red')
            self.pred_plot_4.canvas.ax.scatter(ego_fut_x[-1], ego_fut_y[-1], marker_size, marker="o", facecolors='red', edgecolors='red')
            self.pred_plot_0.canvas.ax.plot(ego_fut_x, ego_fut_y, '--', color='red')
            self.pred_plot_0.canvas.ax.scatter(ego_fut_x[-1], ego_fut_y[-1], marker_size, marker="o", facecolors='red', edgecolors='red')
            pred_reg = self.pred_out_4[0]
            self.pred_plot_4.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
            self.pred_plot_4.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker="o", facecolors='blue', edgecolors='blue')
            self.pred_plot_0.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
            self.pred_plot_0.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker="o", facecolors='blue', edgecolors='blue')
            ego_x_play = np.expand_dims(np.concatenate((ego_hist_x[:-1], ego_fut_x)), axis=-1)
            ego_y_play = np.expand_dims(np.concatenate((ego_hist_y[:-1], ego_fut_y)), axis=-1)
            ego_traj_play = np.expand_dims(np.concatenate((ego_x_play, ego_y_play), axis=-1), axis=0)
            target_x_play = np.expand_dims(np.concatenate((target_hist_x, pred_reg[:, 0].cpu())), axis=-1)
            target_y_play = np.expand_dims(np.concatenate((target_hist_y, pred_reg[:, 1].cpu())), axis=-1)
            target_traj_play = np.expand_dims(np.concatenate((target_x_play, target_y_play), axis=-1), axis=0)
            states = np.expand_dims(np.concatenate((ego_traj_play, target_traj_play), axis=0), axis=0)
            self.state_for_play = np.concatenate((self.state_for_play, states), axis=0)
        if self.ego_path_enable_5.isChecked():
            self.pred_plot_5.canvas.ax.plot(ego_fut_x, ego_fut_y, '--', color='red')
            self.pred_plot_5.canvas.ax.scatter(ego_fut_x[-1], ego_fut_y[-1], marker_size, marker="o", facecolors='red', edgecolors='red')
            self.pred_plot_0.canvas.ax.plot(ego_fut_x, ego_fut_y, '--', color='red')
            self.pred_plot_0.canvas.ax.scatter(ego_fut_x[-1], ego_fut_y[-1], marker_size, marker="o", facecolors='red', edgecolors='red')
            pred_reg = self.pred_out_5[0]
            self.pred_plot_5.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
            self.pred_plot_5.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker="o", facecolors='blue', edgecolors='blue')
            self.pred_plot_0.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
            self.pred_plot_0.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), marker_size, marker="o", facecolors='blue', edgecolors='blue')
            ego_x_play = np.expand_dims(np.concatenate((ego_hist_x[:-1], ego_fut_x)), axis=-1)
            ego_y_play = np.expand_dims(np.concatenate((ego_hist_y[:-1], ego_fut_y)), axis=-1)
            ego_traj_play = np.expand_dims(np.concatenate((ego_x_play, ego_y_play), axis=-1), axis=0)
            target_x_play = np.expand_dims(np.concatenate((target_hist_x, pred_reg[:, 0].cpu())), axis=-1)
            target_y_play = np.expand_dims(np.concatenate((target_hist_y, pred_reg[:, 1].cpu())), axis=-1)
            target_traj_play = np.expand_dims(np.concatenate((target_x_play, target_y_play), axis=-1), axis=0)
            states = np.expand_dims(np.concatenate((ego_traj_play, target_traj_play), axis=0), axis=0)
            self.state_for_play = np.concatenate((self.state_for_play, states), axis=0)

        self.pred_plot_0.canvas.ax.plot(target_fut_x, target_fut_y, ':', color='black')
        self.pred_plot_0.canvas.ax.scatter(target_fut_x[-1], target_fut_y[-1], color='black')
        if self.show_predict_1.isChecked():
            self.pred_plot_1.canvas.ax.plot(target_fut_x, target_fut_y, ':', color='black')
            self.pred_plot_1.canvas.ax.scatter(target_fut_x[-1], target_fut_y[-1], color='black')
        if self.show_predict_2.isChecked():
            self.pred_plot_2.canvas.ax.plot(target_fut_x, target_fut_y, ':', color='black')
            self.pred_plot_2.canvas.ax.scatter(target_fut_x[-1], target_fut_y[-1], color='black')
        if self.show_predict_3.isChecked():
            self.pred_plot_3.canvas.ax.plot(target_fut_x, target_fut_y, ':', color='black')
            self.pred_plot_3.canvas.ax.scatter(target_fut_x[-1], target_fut_y[-1], color='black')
        if self.show_predict_4.isChecked():
            self.pred_plot_4.canvas.ax.plot(target_fut_x, target_fut_y, ':', color='black')
            self.pred_plot_4.canvas.ax.scatter(target_fut_x[-1], target_fut_y[-1], color='black')
        if self.show_predict_5.isChecked():
            self.pred_plot_5.canvas.ax.plot(target_fut_x, target_fut_y, ':', color='black')
            self.pred_plot_5.canvas.ax.scatter(target_fut_x[-1], target_fut_y[-1], color='black')

        self.pred_plot_1.canvas.ax.set_xlim([xmin_1.item(), xmax_1.item()])
        self.pred_plot_1.canvas.ax.set_ylim([ymin_1.item(), ymax_1.item()])
        # self.pred_plot_1.canvas.ax.axis('equal')
        self.pred_plot_1.canvas.draw()

        self.pred_plot_2.canvas.ax.set_xlim([xmin_2.item(), xmax_2.item()])
        self.pred_plot_2.canvas.ax.set_ylim([ymin_2.item(), ymax_2.item()])
        # self.pred_plot_2.canvas.ax.axis('equal')
        self.pred_plot_2.canvas.draw()

        self.pred_plot_3.canvas.ax.set_xlim([xmin_3.item(), xmax_3.item()])
        self.pred_plot_3.canvas.ax.set_ylim([ymin_3.item(), ymax_3.item()])
        # self.pred_plot_3.canvas.ax.axis('equal')
        self.pred_plot_3.canvas.draw()

        self.pred_plot_4.canvas.ax.set_xlim([xmin_4.item(), xmax_4.item()])
        self.pred_plot_4.canvas.ax.set_ylim([ymin_4.item(), ymax_4.item()])
        # self.pred_plot_4.canvas.ax.axis('equal')
        self.pred_plot_4.canvas.draw()

        self.pred_plot_5.canvas.ax.set_xlim([xmin_5.item(), xmax_5.item()])
        self.pred_plot_5.canvas.ax.set_ylim([ymin_5.item(), ymax_5.item()])
        # self.pred_plot_5.canvas.ax.axis('equal')
        self.pred_plot_5.canvas.draw()

        self.pred_plot_0.canvas.ax.set_xlim([xmin_0.item(), xmax_0.item()])
        self.pred_plot_0.canvas.ax.set_ylim([ymin_0.item(), ymax_0.item()])
        # self.pred_plot_5.canvas.ax.axis('equal')
        self.pred_plot_0.canvas.draw()

    def scenario_play(self):
        data = self.state_for_play
        scene_id = self.idx_data_1.text()
        for i in range(data.shape[0]):
            trajs = data[i]
            ego_cur_pos = self.cur_data['gt_preds'][0][0, 0, :]
            xmin_0 = ego_cur_pos[0] - self.fov_0 + self.x_offset_0
            xmax_0 = ego_cur_pos[0] + self.fov_0 + self.x_offset_0
            ymin_0 = ego_cur_pos[1] - self.fov_0 + self.y_offset_0
            ymax_0 = ego_cur_pos[1] + self.fov_0 + self.y_offset_0
            city_name = self.cur_data['city'][0]
            local_lane_polygons_0 = am.find_local_lane_polygons([xmin_0, xmax_0, ymin_0, ymax_0], city_name)
            for t in range(40):
                print(t)
                self.pred_plot_0.canvas.ax.clear()
                draw_lane_polygons(self.pred_plot_0.canvas.ax, local_lane_polygons_0, color='darkgray')
                self.pred_plot_0.canvas.ax.plot(trajs[0, :t + 1, 0], trajs[0, :t + 1, 1], '-', color='red')
                self.pred_plot_0.canvas.ax.plot(trajs[1, :t + 1, 0], trajs[1, :t + 1, 1], '-', color='blue')
                self.pred_plot_0.canvas.ax.scatter(trajs[0, t, 0], trajs[0, t, 1], 50, marker="o", facecolors='none', edgecolors='red')
                self.pred_plot_0.canvas.ax.scatter(trajs[1, t, 0], trajs[1, t, 1], 50, marker="o", facecolors='none', edgecolors='blue')
                self.pred_plot_0.canvas.ax.set_xlim([xmin_0.item(), xmax_0.item()])
                self.pred_plot_0.canvas.ax.set_ylim([ymin_0.item(), ymax_0.item()])
                self.pred_plot_0.canvas.draw()
                buf = self.pred_plot_0.canvas.buffer_rgba()
                X = np.asarray(buf)
                im = Image.fromarray(X)
                name = 'plot/' + scene_id + '_' + str(i) + '_' + str(t) + '.png'
                im.save(name)

    def get_eval_data_1(self):
        pred_err = torch.norm(self.pred_gt[0] - self.pred_out_1[0], dim=1)
        ade_pred = torch.mean(pred_err)
        fde_pred = pred_err[-1]

        recon_err = torch.norm(self.recon_gt - self.recon_out_1, dim=2)
        ade_recon = torch.mean(recon_err)
        fde_recon = torch.mean(recon_err[:, -1])
        return ade_pred, fde_pred, ade_recon, fde_recon


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
