# -*- coding:UTF-8 -*-

import os
import numpy as np

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font


class SaveExcel(object):
    def __init__(self, test_list, root_path, excel_name='reg_output'):
        self.test_list = test_list
        self.save_path = root_path
        excel_path = os.path.join(root_path, '{}.xlsx'.format(excel_name))
        self.excel_path = excel_path
        self.creat_excel()

    def creat_excel(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = 'sheet1'
        font = Font(bold=True)
        cell0 = sheet.cell(row=1, column=1)
        cell0.value = 'epoch'
        cell0.font = font
        alignment = Alignment(horizontal="center", vertical="center")
        for i, item in enumerate(self.test_list, 0):
            cell1 = sheet.cell(row=1, column=i * 5 + 2)
            cell2 = sheet.cell(row=1, column=i * 5 + 3)
            cell3 = sheet.cell(row=1, column=i * 5 + 4)
            cell4 = sheet.cell(row=1, column=i * 5 + 5)
            cell5 = sheet.cell(row=1, column=i * 5 + 6)

            cell1.value = '{:02d} RR'.format(item)
            cell2.value = '{:02d} TM'.format(item)
            cell3.value = '{:02d} TS'.format(item)
            cell4.value = '{:02d} RM'.format(item)
            cell5.value = '{:02d} RS'.format(item)
            cell1.font = font
            cell2.font = font
            cell3.font = font
            cell4.font = font
            cell5.font = font
            cell1.alignment = alignment
            cell2.alignment = alignment
            cell3.alignment = alignment
            cell4.alignment = alignment
            cell5.alignment = alignment
        workbook.save(filename=self.excel_path)

    def update(self, eval_dir: str, read_file_name: str = 'reg_output'):
        col_len = len(self.test_list)
        workbook = load_workbook(filename=self.excel_path)
        sheet = workbook.active
        sheet.column_dimensions.width = 30
        sheet.column_dimensions['A'].width = 9
        alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        mean_RR = []
        mean_TM = []
        mean_TS = []
        mean_RM = []
        mean_RS = []
        for col, item in enumerate(self.test_list, 1):
            txt_path = os.path.join(eval_dir, 'regformer_{:02d}'.format(item), '{}.txt'.format(read_file_name))
            if not os.path.exists(txt_path):
                continue
            epoch_list = []
            RR_list = []
            TM_list = []
            TS_list = []
            RM_list = []
            RS_list = []

            with open(txt_path, 'r') as f:
                data = f.readlines()
                for row in range(int(len(data) / 6)):
                    index1 = data[row * 6].index(':') + 2
                    index2 = data[row * 6 + 1].index(':') + 2
                    index3 = data[row * 6 + 2].index(':') + 2
                    index4 = data[row * 6 + 3].index(':') + 2
                    index5 = data[row * 6 + 4].index(':') + 2
                    index6 = data[row * 6 + 5].index(':') + 2
                    ep = data[row * 6][index1:].strip()
                    RR = data[row * 6 + 1][index2:].strip()
                    TM = data[row * 6 + 2][index3:].strip()
                    TS = data[row * 6 + 3][index4:].strip()
                    RM = data[row * 6 + 4][index5:].strip()
                    RS = data[row * 6 + 5][index6:].strip()

                    cell0 = sheet.cell(row=row + 2, column=1)
                    cell1 = sheet.cell(row=row + 2, column=(col-1) * 5 + 2)
                    cell2 = sheet.cell(row=row + 2, column=(col-1) * 5 + 3)
                    cell3 = sheet.cell(row=row + 2, column=(col-1) * 5 + 4)
                    cell4 = sheet.cell(row=row + 2, column=(col-1) * 5 + 5)
                    cell5 = sheet.cell(row=row + 2, column=(col-1) * 5 + 6)

                    cell0.value = int(ep)
                    cell0.alignment = alignment
                    cell1.value = float(RR)
                    cell1.alignment = alignment
                    cell2.value = float(TM)
                    cell2.alignment = alignment
                    cell3.value = float(TS)
                    cell3.alignment = alignment
                    cell4.value = float(RM)
                    cell4.alignment = alignment
                    cell5.value = float(RS)
                    cell5.alignment = alignment

                    epoch_list.append(int(ep))
                    RR_list.append(float(RR))
                    TM_list.append(float(TM))
                    TS_list.append(float(TS))
                    RM_list.append(float(RM))
                    RS_list.append(float(RS))

            mean_RR.append(RR_list)
            mean_TM.append(TM_list)
            mean_TS.append(TS_list)
            mean_RM.append(RM_list)
            mean_RS.append(RS_list)


            cell = sheet.cell(row=1, column=col_len * 5 + 5 + col)
            cell.value = '{:02d}'.format(item)
            cell.alignment = alignment

            min_RR = max(RR_list)
            min_index = RR_list.index(min_RR)
            cell_min = sheet.cell(row=2, column=col_len * 5 + 5 + col)
            cell_min.value = '{:d}: {:.4f}'.format(epoch_list[min_index], min_RR)
            cell_min.alignment = alignment

            min_TM = min(TM_list)
            min_index = TM_list.index(min_TM)
            cell_min = sheet.cell(row=3, column=col_len * 5 + 5 + col)
            cell_min.value = '{:d}: {:.4f}'.format(epoch_list[min_index], min_TM)
            cell_min.alignment = alignment

            min_TS = min(TS_list)
            min_index = TS_list.index(min_TS)
            cell_min = sheet.cell(row=4, column=col_len * 5 + 5 + col)
            cell_min.value = '{:d}: {:.4f}'.format(epoch_list[min_index], min_TS)
            cell_min.alignment = alignment

            min_RM = min(RM_list)
            min_index = RM_list.index(min_RM)
            cell_min = sheet.cell(row=5, column=col_len * 5 + 5 + col)
            cell_min.value = '{:d}: {:.4f}'.format(epoch_list[min_index], min_RM)
            cell_min.alignment = alignment

            min_RS = min(RS_list)
            min_index = RS_list.index(min_RS)
            cell_min = sheet.cell(row=6, column=col_len * 5 + 5 + col)
            cell_min.value = '{:d}: {:.4f}'.format(epoch_list[min_index], min_RS)
            cell_min.alignment = alignment

        mean_array = np.array(mean_RR)
        mean = np.mean(mean_array, axis=0)
        min_mean = max(mean)
        min_index = np.where(mean == min_mean)[-1][-1]
        cell = sheet.cell(row=1, column=col_len * 5 + 5)
        cell.value = 'mean_min'
        cell.alignment = alignment
        cell_min = sheet.cell(row=2, column=col_len * 5 + 5)
        cell_min.value = '{:d}: {:.4f}'.format(epoch_list[min_index], min_mean)
        cell_min.alignment = alignment

        mean_array = np.array(mean_TM)
        mean = np.mean(mean_array, axis=0)
        min_mean = min(mean)
        min_index = np.where(mean == min_mean)[-1][-1]
        cell_min = sheet.cell(row=3, column=col_len * 5 + 5)
        cell_min.value = '{:d}: {:.4f}'.format(epoch_list[min_index], min_mean)
        cell_min.alignment = alignment

        mean_array = np.array(mean_TS)
        mean = np.mean(mean_array, axis=0)
        min_mean = min(mean)
        min_index = np.where(mean == min_mean)[-1][-1]
        cell_min = sheet.cell(row=4, column=col_len * 5 + 5)
        cell_min.value = '{:d}: {:.4f}'.format(epoch_list[min_index], min_mean)
        cell_min.alignment = alignment

        mean_array = np.array(mean_RM)
        mean = np.mean(mean_array, axis=0)
        min_mean = min(mean)
        min_index = np.where(mean == min_mean)[-1][-1]
        cell_min = sheet.cell(row=5, column=col_len * 5 + 5)
        cell_min.value = '{:d}: {:.4f}'.format(epoch_list[min_index], min_mean)
        cell_min.alignment = alignment

        mean_array = np.array(mean_RS)
        mean = np.mean(mean_array, axis=0)
        min_mean = min(mean)
        min_index = np.where(mean == min_mean)[-1][-1]
        cell_min = sheet.cell(row=6, column=col_len * 5 + 5)
        cell_min.value = '{:d}: {:.4f}'.format(epoch_list[min_index], min_mean)
        cell_min.alignment = alignment

        workbook.save(filename=self.excel_path)
