from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import numpy as np
import importlib.util
import sys
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
from argoverse.utils.mpl_plotting_utils import draw_lane_polygons, plot_nearby_centerlines
from importlib import import_module

# am = ArgoverseMap()
import random
import csv
import glob



class MainDialog(QMainWindow, ui_files.gui.Ui_Dialog):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.root_dir = os.getcwd()
        self.loadData.clicked.connect(self.data_load)
        self.next_idx.clicked.connect(self.next)
        self.ego_path_enable.clicked.connect(self.visualization)
        self.ego_path_aug_1.clicked.connect(self.visualization)
        self.ego_path_aug_2.clicked.connect(self.visualization)
        self.ego_path_aug_3.clicked.connect(self.visualization)
        self.ego_path_aug_4.clicked.connect(self.visualization)
        self.ego_path_aug_5.clicked.connect(self.visualization)
        self.ego_path_aug_6.clicked.connect(self.visualization)
        self.ego_path_aug_7.clicked.connect(self.visualization)
        self.ego_path_aug_8.clicked.connect(self.visualization)
        self.show_predict.clicked.connect(self.visualization)

        self.idx = 0
        self.fov = 25
        self.data_dir = '/home/jhs/Desktop/SRFNet/LaneGCN/dataset/val/data/'
        self.cand_toggles = [self.ego_path_enable,
                             self.ego_path_aug_1,
                             self.ego_path_aug_2,
                             self.ego_path_aug_3,
                             self.ego_path_aug_4,
                             self.ego_path_aug_5,
                             self.ego_path_aug_6,
                             self.ego_path_aug_7,
                             self.ego_path_aug_8]

    def next(self):
        self.cur_data = next(iter(self.data_loader))
        target_traj = self.cur_data['gt_preds'][0][1]
        while torch.norm(target_traj[0, :] - target_traj[-1, :]) < 2:
            self.cur_data = next(iter(self.data_loader))
        self.update_data()

    def update_data(self):
        self.map_data.setText(self.cur_data['city'][0])
        self.num_of_vehicles_data.setText(str(self.cur_data['feats'][0].shape[0]))
        self.SceneInfo_data.setText('None')
        self.idx_data.setText(str(self.cur_data['idx'][0]))
        ego_aug = self.cur_data['ego_aug'][0]['traj']
        self.ego_path_enable.setEnabled(True)
        self.ego_path_enable.setChecked(True)
        self.pred_out = []
        self.show_predict.setChecked(True)
        with torch.no_grad():
            self.pred_out.append(self.net(self.cur_data))
        for i in range(8):
            if i < ego_aug.shape[0]:
                self.cand_toggles[i + 1].setEnabled(True)
                self.cand_toggles[i + 1].setChecked(True)
                with torch.no_grad():
                    data_tmp = self.cur_data.copy()
                    data_tmp['action'][0][0:1, 0, :, :] = data_tmp['action'][0][0:1, i + 1, :, :]
                    self.pred_out.append(self.net(data_tmp))
            else:
                self.cand_toggles[i + 1].setEnabled(False)
                self.cand_toggles[i + 1].setChecked(False)

        ade1, fde1, ade6, fde6 = self.get_eval_data(self.pred_out[0])
        if '1_lanegcn' in self.model_id or '2_lanegcn_simple_head' in self.model_id:
            for i in range(len(self.pred_out)):
                self.pred_out[i]['reg'][0] = self.pred_out[i]['reg'][0][1:2]

        self.ade1_data.setText(str(ade1)[:5])
        self.fde1_data.setText(str(fde1)[:5])
        self.ade6_data.setText(str(ade6)[:5])
        self.fde6_data.setText(str(fde6)[:5])
        self.update_table()
        self.visualization()

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
        self.pred_plot.canvas.ax.clear()
        ego_cur_pos = self.cur_data['gt_preds'][0][0, 0, :]
        xmin = ego_cur_pos[0] - self.fov
        xmax = ego_cur_pos[0] + self.fov
        ymin = ego_cur_pos[1] - self.fov
        ymax = ego_cur_pos[1] + self.fov
        city_name = self.cur_data['city'][0]
        local_lane_polygons = am.find_local_lane_polygons([xmin, xmax, ymin, ymax], city_name)
        draw_lane_polygons(self.pred_plot.canvas.ax, local_lane_polygons, color='k')

        raw_data = []
        with open(self.data_dir + self.cur_data['file_name'][0], newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                raw_data.append(row)
        raw_data = raw_data[1:]
        x = np.asarray([float(raw_data[i][0].split(',')[3]) for i in range(len(raw_data))])
        y = np.asarray([float(raw_data[i][0].split(',')[4]) for i in range(len(raw_data))])
        veh_class = [raw_data[i][0].split(',')[2] for i in range(len(raw_data))]
        ego_index = [i for i, x in enumerate(veh_class) if x == 'AV']
        ego_x = x[ego_index]
        ego_y = y[ego_index]
        ego_hist_x = ego_x[:20]
        ego_hist_y = ego_y[:20]
        ego_fut_x = ego_x[19:]
        ego_fut_y = ego_y[19:]
        target_index = [i for i, x in enumerate(veh_class) if x == 'AGENT']
        target_x = x[target_index]
        target_y = y[target_index]
        target_hist_x = target_x[:20]
        target_hist_y = target_y[:20]
        target_fut_x = target_x[19:]
        target_fut_y = target_y[19:]
        self.pred_plot.canvas.ax.plot(ego_hist_x, ego_hist_y, '-', color='red')
        self.pred_plot.canvas.ax.plot(target_hist_x, target_hist_y, '-', color='blue')
        self.pred_plot.canvas.ax.scatter(target_hist_x[-1], target_hist_y[-1], color='blue')

        ego_aug = self.cur_data['ego_aug'][0]['traj'].numpy().copy()
        ego_aug = np.concatenate([ego_aug, np.zeros_like(ego_aug[:, 0:1, :])], axis=1)
        for i in range(ego_aug.shape[0]):
            ego_aug[i, :, :] = np.concatenate([np.expand_dims(np.asarray([ego_hist_x[-1], ego_hist_y[-1]]), axis=0), ego_aug[i, :30, :]], axis=0)
            if self.cand_toggles[i + 1].isChecked():
                aug_x = ego_aug[i, :, 0]
                aug_y = ego_aug[i, :, 1]
                self.pred_plot.canvas.ax.plot(aug_x, aug_y, '--', color='red')
                self.pred_plot.canvas.ax.scatter(aug_x[-1], aug_y[-1], facecolors='none', edgecolors='red')

                pred_out = self.pred_out[i + 1]
                best_idx = np.argmax(pred_out['cls'][0][0].cpu().detach().numpy())
                pred_reg = pred_out['reg'][0][0, best_idx, :, :]
                self.pred_plot.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
                self.pred_plot.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), facecolors='none', edgecolors='blue')

        if self.ego_path_enable.isChecked():
            self.pred_plot.canvas.ax.plot(ego_fut_x, ego_fut_y, '--', color='red')
            self.pred_plot.canvas.ax.scatter(ego_x[-1], ego_y[-1], facecolors='none', edgecolors='red')
            pred_out = self.pred_out[0]
            best_idx = np.argmax(pred_out['cls'][0][0].cpu().detach().numpy())
            pred_reg = pred_out['reg'][0][0, best_idx, :, :]
            self.pred_plot.canvas.ax.plot(pred_reg[:, 0].cpu(), pred_reg[:, 1].cpu(), '--', color='blue')
            self.pred_plot.canvas.ax.scatter(pred_reg[-1, 0].cpu(), pred_reg[-1, 1].cpu(), facecolors='none', edgecolors='blue')

        if self.show_predict.isChecked():
            self.pred_plot.canvas.ax.plot(target_fut_x, target_fut_y, ':', color='black')
            self.pred_plot.canvas.ax.scatter(target_fut_x[-1], target_fut_y[-1], color='black')
        self.pred_plot.canvas.draw()
        self.pred_plot.canvas.ax.axis('equal')

    def get_eval_data(self, pred_out):
        metrics = dict()
        loss_out = self.loss(pred_out, self.cur_data)
        post_out = self.post_process(pred_out, self.cur_data)
        self.post_process.append(metrics, loss_out, post_out)
        dt = 0
        metrics = sync(metrics)
        cls = metrics["cls_loss"] / (metrics["num_cls"] + 1e-10)
        reg = metrics["reg_loss"] / (metrics["num_reg"] + 1e-10)
        loss = cls + reg

        preds = np.concatenate(metrics["preds"], 0)
        gt_preds = np.concatenate(metrics["gt_preds"], 0)
        has_preds = np.concatenate(metrics["has_preds"], 0)
        ade1, fde1, ade, fde, min_idcs = pred_metrics(preds, gt_preds, has_preds)

        return ade1, fde1, ade, fde


    def data_load(self):
        root = tkinter.Tk()
        root.withdraw()

        currdir = os.getcwd()
        data_file_dir = tkinter.filedialog.askdirectory(parent=root, initialdir=os.path.join(currdir))
        
        self.data_dir = data_file_dir
        self.file_list = glob.glob(self.data_dir+'/*.csv')
        print(self.file_list)
        self.cur_data_idx = 0
        self.dataInfo.setText(data_file_dir + ' is loaded successfully')
        self.data_num.setText(str(len(self.file_list)))


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
