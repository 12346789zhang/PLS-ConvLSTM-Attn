"""

2025年9月27日
模型： LS+AR
数据：EOP + EAM + EAM6 pred

"""

import math
import os
import warnings
from datetime import datetime, timedelta

import matplotlib
import numpy as np
import pandas as pd  # 导入pandas模块，用于数据处理和分析
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller


warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "SimHei"  # 定义使其正常显示中文字体黑体
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示表示负号

output_folder = 'F:\\zll\\LOD\\fig'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
plot_counter = 1

# 确保目标文件夹存在
log_folder1 = r"F:\\zll\\LOD"
log_folder2 = r"F:\\zll\\LOD"
os.makedirs(log_folder1, exist_ok=True)
os.makedirs(log_folder2, exist_ok=True)

for g in range(0, 500, 1):

    def is_integer(s):
       
        try:
            int(s)
            return True
        except ValueError:
            return False

    def read_specific_column_rows(file_path, column_index, start_date, end_date):
      
        selected_data = []
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
                for line in lines:
                    columns = line.split()
                    # 检查第一列是否能转换为整数，若不能则跳过该行
                    if len(columns) >= column_index and is_integer(columns[0]):
                        year = int(columns[0])
                        month = int(columns[1])
                        day = int(columns[2])
                        current_date = datetime(year, month, day)
                        if start_date <= current_date <= end_date:
                            selected_data.append(float(columns[column_index - 1]))
        except FileNotFoundError:
            raise FileNotFoundError(f"文件 {file_path} 不存在")
        except Exception as e:
            raise RuntimeError(f"读取文件 {file_path} 时出错: {e}")
        return selected_data

    # 根据 g 动态调整日期范围
    def get_dynamic_date_range(g):
        """根据 g 动态调整日期范围"""
        base_start_date = datetime(2015, 8, 31)
        base_end_date = datetime(2021, 8, 31)

        # 动态调整日期范围
        start_date = base_start_date + timedelta(days=g)
        end_date = base_end_date + timedelta(days=g)

        # 添加一个向后 365 天的日期范围
        extended_end_date = end_date + timedelta(days=90)

        return start_date, end_date, extended_end_date

    # 示例调用

    start_date, end_date, extended_end_date = get_dynamic_date_range(g)


    UT1_UTC = read_specific_column_rows("F:\\zll\\LOD\\EOP14.txt", 7, start_date, end_date)
    LOD = read_specific_column_rows("F:\\zll\\LOD\\EOP14.txt", 8, start_date, end_date)
    MJD = read_specific_column_rows("F:\\zll\\LOD\\EOP14.txt", 4, start_date, end_date)

    UT4_UTC = read_specific_column_rows("F:\\zll\\LOD\\EOP14.txt", 7, start_date, extended_end_date)
    lod2 = read_specific_column_rows("F:\\zll\\LOD\\EOP14.txt", 8, start_date, extended_end_date)
    MJD90 = read_specific_column_rows("F:\\zll\\LOD\\EOP14.txt", 4, start_date, extended_end_date)

    DUT_values = read_specific_column_rows("F:\\zll\\LOD\\潮汐值.txt", 5, start_date, end_date)
    DLOD_values = read_specific_column_rows("F:\\zll\\LOD\\潮汐值.txt", 6, start_date, end_date)

    DUT_values1 = read_specific_column_rows("F:\\zll\\LOD\\潮汐值.txt", 5, start_date, extended_end_date)
    DLOD_values1 = read_specific_column_rows("F:\\zll\\LOD\\潮汐值.txt", 6, start_date, extended_end_date)
    ##################################跳秒################################################################

    #LODKCX = np.array(LOD) - np.array(DLOD_values)

    UT1_UTCX = np.array(UT1_UTC) - np.array(DUT_values)
    # -----------------------------扣除跳秒-----------------------------

    def get_leap_second(mjd):
        leap_sec_epochs = [
            57754, 57204, 56109, 54832, 53736, 51179, 50630, 50083, 49534, 49169,
            48804, 48257, 47892, 47161, 46247, 45516, 45151, 44786, 44239, 43874,
            43509, 43144, 42778, 42413, 42048, 41683, 41499, 41317,
        ]
        sec = [37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12,
               11,
               10]
        for i in range(len(leap_sec_epochs)):
            if mjd >= leap_sec_epochs[i]:
                return sec[i]
        return 10  # 默认返回0


    leap_seconds = [get_leap_second(mjd) for mjd in MJD]
    leap_seconds90 = [get_leap_second(mjd) for mjd in MJD90]

    UT1_TAI = UT1_UTCX - leap_seconds


    # -----------------------------刘维尔方程 一阶向前/后差分-----------------------------
    # diff = -np.diff(UT1CX)

    def extrap(y, n):
        
        # 在数组前后分别填充 n 个边界值
        return np.pad(y, (n, n), mode='edge')


    def diff_backward(y):
       
        # 输入校验
        if not isinstance(y, (list, np.ndarray)):
            raise ValueError("输入 y 必须是列表或 np.ndarray 类型")
        y = np.array(y)  # 转换为 np.ndarray

        # 扩展数组边界
        y_ext = extrap(y, 1)

        # 计算向后差分：y[i] - y[i-1]
        diff_result = y - y_ext[:-2]

        return diff_result


    # 计算向后差分
    LODKCX = -diff_backward(UT1_TAI)

    GAM = (7.292115e-5 / (2 * math.pi)) * LODKCX


    # ----------------------------有效角动量数据-----------------------------------------------------

    def calculate_weighted_average(data, step, weights):
        """
        计算加权平均值
        """
        if len(data) % step != 0:
            # 调整数据长度为步长的整数倍
            new_length = len(data) - (len(data) % step)
            data = data[:new_length]
        grouped_data = [data[i: i + step] for i in range(0, len(data), step)]
        weighted_avg = [np.average(group, weights=weights) for group in grouped_data]
        return weighted_avg


    def process_term(file_path, column_index, start_date, end_date, step, weights):
        """
        处理单个项（AAM/OAM/HAM/SLAM 的质量项或运动项）
        """
        data = read_specific_column_rows(file_path, column_index, start_date, end_date)
        if not data:
            print(f"警告: 文件 {file_path} 中未找到符合日期范围的数据")
            return []
        weighted_avg = calculate_weighted_average(data, step, weights)
        return weighted_avg


    # 常量定义
    STEP = 8
    WEIGHTS = [1, 1, 3, 4, 5, 4, 3, 1]

    # 处理 AAM 质量项和运动项
    aam_mass = process_term("F:\\zll\\LOD\\AAM_85-24.txt", 8, start_date, end_date, STEP,
                            WEIGHTS)
    aam_motion = process_term("F:\\zll\\LOD\\AAM_85-24.txt", 11, start_date, end_date, STEP,
                              WEIGHTS)

    # 处理 OAM 质量项和运动项
    oam_mass = process_term("F:\\zll\\LOD\\OAM_85-24.txt", 8, start_date, end_date, STEP,
                            WEIGHTS)
    oam_motion = process_term("F:\\zll\\LOD\\OAM_85-24.txt", 11, start_date, end_date, STEP,
                              WEIGHTS)
    # 处理 HAM 质量项和运动项，不进行加权平均
    ham_mass = read_specific_column_rows("F:\\zll\\LOD\\HAM_85-24.txt", 8,
                                         start_date, end_date)
    ham_motion = read_specific_column_rows("F:\\zll\\LOD\\HAM_85-24.txt", 11,
                                           start_date, end_date)

    # 处理 SLAM 质量项，不进行加权平均
    slam_mass = read_specific_column_rows("F:\\zll\\LOD\\SLAM_85-24.txt", 8,
                                          start_date, end_date)

    print(slam_mass)

    # 检查数据长度是否一致
    if not ham_mass:
        ham_mass = [0] * len(aam_mass)  # 如果 ham_mass 为空，设置为与 aam_mass 长度相同的零数组
    if not ham_motion:
        ham_motion = [0] * len(aam_mass)  # 如果 ham_motion 为空，设置为与 aam_mass 长度相同的零数组
    if not slam_mass:
        slam_mass = [0] * len(aam_mass)  # 如果 slam_mass 为空，设置为与 aam_mass 长度相同的零数组

    # print(len(aam_mass), len(aam_motion), len(oam_mass), len(oam_motion), len(ham_mass), len(ham_motion),len(slam_mass))

    # 计算 EAM
    EAM = np.array(aam_mass) + np.array(aam_motion) + np.array(oam_mass) + np.array(oam_motion) + np.array(
        ham_mass) + np.array(ham_motion) + np.array(slam_mass)

    GAM_EAM = np.array(GAM) - np.array(EAM)


    # -----------------------------------EAM 6 days--------------------------------------------

    # 定义一个函数来读取并提取文件中的第三列和第五列数据，并将它们相加
    def read_and_add_columns(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                third_column = []
                fifth_column = []
                for line in file:
                    columns = line.strip().split()  # 假设数据是以空格分隔的
                    if len(columns) >= 4:  # 确保至少有第三列
                        try:
                            third_column.append(float(columns[3]))  # 第四列索引为 3
                            if len(columns) >= 7:  # 确保至少有第五列
                                fifth_column.append(float(columns[6]))  # 第七列索引为 6
                        except ValueError:
                            print(f"文件 {file_path} 中的数据格式不正确: {line.strip()}")
                return third_column, fifth_column
        except FileNotFoundError:
            print(f"文件 {file_path} 未找到。")
            return [], []  # 返回空列表表示文件未找到
        except Exception as e:
            print(f"读取文件 {file_path} 时发生错误: {e}")
            return [], []


    # 生成指定日期范围内的文件路径并处理文件
    def process_files(folder_path, file_suffix, start_date, end_date, weighted_avg=False):
        current_date = start_date
        res_third = []
        res_fifth = []
        step = 8
        weights = np.array([1, 1, 3, 4, 5, 4, 3, 1])
        missing_files = []  # 用于存储不存在的文件
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            file_name = f"{date_str}{file_suffix}"
            file_path = os.path.join(folder_path, file_name)

            # 读取并提取第三列和第五列数据
            third_column, fifth_column = read_and_add_columns(file_path)

            if not third_column and not fifth_column:
                missing_files.append(file_path)  # 记录不存在的文件

            if third_column is not None and fifth_column is not None:
                if weighted_avg:
                    # 计算第三列的加权平均
                    for i in range(0, len(third_column), step):
                        chunk = third_column[i: i + step]
                        if len(chunk) == step:
                            res11 = np.average(chunk, weights=weights)
                            res_third.append(res11)

                    # 计算第五列的加权平均
                    for i in range(0, len(fifth_column), step):
                        chunk = fifth_column[i: i + step]
                        if len(chunk) == step:
                            res11 = np.average(chunk, weights=weights)
                            res_fifth.append(res11)
                else:
                    res_third.extend(third_column)
                    res_fifth.extend(fifth_column)

            # 移动到下一个日期
            current_date += timedelta(days=1)

        if missing_files:
            print("以下文件不存在:")
            for file in missing_files:
                print(file)

        # 获取每6个数据的块
        chunk_size = 6
        res1_third = [
            res_third[j: j + chunk_size]
            for j in range(0, len(res_third), chunk_size)
        ]
        res1_fifth = [
            res_fifth[j: j + chunk_size]
            for j in range(0, len(res_fifth), chunk_size)
        ]

        # 计算需要填充 0 的块数
        total_days = (end_date - start_date).days + 1
        total_chunks = (total_days * 24 // 8) // 6  # 假设一天 24 小时，每 8 个数据计算一次，每 6 个结果一组
        while len(res1_third) < total_chunks:
            res1_third.append([0] * chunk_size)
        while len(res1_fifth) < total_chunks:
            res1_fifth.append([0] * chunk_size)

        return res1_third, res1_fifth


    # 定义所需的变量
    start_date = datetime(2021, 9, 1)
    end_date = datetime(2023, 5, 1)
    # 文件夹路径2
    folder_path1 = r"F:\张璐璐\日常变化\EAM\AAM"
    folder_path2 = r"F:\张璐璐\日常变化\EAM\OAM"
    folder_path3 = r"F:\张璐璐\日常变化\EAM\HAM"
    folder_path4 = r"F:\张璐璐\日常变化\EAM\SLAM"

    # 处理 AAM 文件
    res1A_third, res1A_fifth = process_files(folder_path1, ".AAMxyz_prediction.txt", start_date, end_date,
                                             weighted_avg=True)
    # print(np.array(res1A_third))
    # print(np.array(res1A_fifth))
    res1A = res1A_third[g]
    res1AA = res1A_fifth[g]

    # 处理 OAM 文件
    res1O_third, res1O_fifth = process_files(folder_path2, ".OAMxyz_prediction.txt", start_date, end_date,
                                             weighted_avg=True)
    # print("OAM 第三列完整数据（包含用 0 填充部分）:")
    # print(np.array(res1O_third))
    # print("OAM 第五列完整数据（包含用 0 填充部分）:")
    # print(np.array(res1O_fifth))
    res1O = res1O_third[g]
    res1OO = res1O_fifth[g]

    # 处理 HAM 文件
    res1H_third, res1H_fifth = process_files(folder_path3, ".HAMxyz_prediction.txt", start_date, end_date,
                                             weighted_avg=False)
    # print("HAM 第三列完整数据（包含用 0 填充部分）:")
    # print(np.array(res1H_third))
    # print("HAM 第五列完整数据（包含用 0 填充部分）:")
    # print(np.array(res1H_fifth))
    res1H = res1H_third[g]
    res1HH = res1H_fifth[g]

    # 处理 SLAM 文件
    res1SL_third, _ = process_files(folder_path4, ".SLAMxyz_prediction.txt", start_date, end_date, weighted_avg=False)
    # print("SLAM 第三列完整数据（包含用 0 填充部分）:")
    # print(np.array(res1SL_third))
    res1SL = res1SL_third[g]

    # 计算 EAM1
    eam6 = (
            np.array(res1A)
            + np.array(res1AA)
            + np.array(res1O)
            + np.array(res1OO)
            + np.array(res1H)
            + np.array(res1HH)
            + np.array(res1SL)
    )
    EAM6 = np.concatenate((EAM,eam6))
    # ------------------------------------最小二乘拟合----------------------------------------

    def lsq_func_lod(y, mjd):
        p0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        res_lsq = least_squares(_func_lod_res, p0, args=(mjd, y))
        xfit = func_lod(res_lsq.x, mjd)
        return res_lsq.x, xfit


    def _func_lod_res(p, t, y):
        return func_lod(p, t) - y


    def func_lod(arg, t):
        return (
                arg[0]
                + arg[1] * t
                + arg[2] * np.sin(2 * np.pi * t / 9.13)
                + arg[3] * np.cos(2 * np.pi * t / 9.13)
                + arg[4] * np.sin(2 * np.pi * t / 13.7)
                + arg[5] * np.cos(2 * np.pi * t / 13.7)
                + arg[6] * np.sin(2 * np.pi * t / (365 / 3))
                + arg[7] * np.cos(2 * np.pi * t / (365 / 3))
                + arg[8] * np.sin(2 * np.pi * t / (365 / 2))
                + arg[9] * np.cos(2 * np.pi * t / (365 / 2))
                + arg[10] * np.sin(2 * np.pi * t / (365 / 1))
                + arg[11] * np.cos(2 * np.pi * t / (365 / 1))
                + arg[12] * np.sin(2 * np.pi * t / 23.9)
                + arg[13] * np.cos(2 * np.pi * t / 23.9)
                + arg[14] * np.sin(2 * np.pi * t / 91.3)
                + arg[15] * np.cos(2 * np.pi * t / 91.3)
                + arg[16] * np.sin(2 * np.pi * t / (365 * 9.3))
                + arg[17] * np.cos(2 * np.pi * t / (365 * 9.3))
                + arg[18] * np.sin(2 * np.pi * t / (365 * 3))
                + arg[19] * np.cos(2 * np.pi * t / (365 * 3))
        )


    t = np.array(range(0, len(GAM_EAM), 1))
    y = np.array(GAM_EAM)

    # 进行最小二乘拟合
    params, y_fit = lsq_func_lod(y, t)

    # 生成新的时间范围，包括原始时间和额外的90天
    t_ext90 = np.arange(t[0], t[-1] + 91, 1)

    # 使用拟合参数进行外推
    gam_eam90 = func_lod(params, t_ext90)

    ccx = y - y_fit

    # ---------------------------------------------------------------------------------------
    # -------------------------------AR 外推6天----------------------------------
    # 检验平稳性
    result = adfuller(ccx)
    # 确定最佳的AR模型的期数
    aic_values = []
    bic_values = []
    max_lags = 80  # 最大滞后期数

    for lag in range(1, max_lags + 1):
        model = AutoReg(ccx, lags=lag)
        model_fit = model.fit()
        aic_values.append(model_fit.aic)
        bic_values.append(model_fit.bic)

    # 创建一个DataFrame以便于分析
    results = pd.DataFrame(
        {"Lag": range(1, max_lags + 1), "AIC": aic_values, "BIC": bic_values}
    )

    # 找到最优的滞后期
    best_aic_lag = results.loc[results["AIC"].idxmin()]["Lag"]
    best_bic_lag = results.loc[results["BIC"].idxmin()]["Lag"]

    # 拟合AR模型
    model = AutoReg(ccx, lags=80)
    model_fit = model.fit()

    # 打印模型摘要
    print(model_fit.summary())

    # 进行90天预测
    predictions90 = model_fit.predict(
        start=len(ccx), end=len(ccx) + 90, dynamic=False
    )

    ccx90 = np.concatenate((ccx, predictions90))


    # --------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------

    def lsq_func_lod(y, mjd):
        p0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        res_lsq = least_squares(_func_lod_res, p0, args=(mjd, y))
        xfit = func_lod(res_lsq.x, mjd)
        return res_lsq.x, xfit


    def _func_lod_res(p, t, y):
        return func_lod(p, t) - y


    def func_lod(arg, t):
        return (
                arg[0]
                + arg[1] * t
                + arg[2] * np.sin(2 * np.pi * t / 9.13)
                + arg[3] * np.cos(2 * np.pi * t / 9.13)
                + arg[4] * np.sin(2 * np.pi * t / 13.7)
                + arg[5] * np.cos(2 * np.pi * t / 13.7)
                + arg[6] * np.sin(2 * np.pi * t / (365 / 3))
                + arg[7] * np.cos(2 * np.pi * t / (365 / 3))
                + arg[8] * np.sin(2 * np.pi * t / (365 / 2))
                + arg[9] * np.cos(2 * np.pi * t / (365 / 2))
                + arg[10] * np.sin(2 * np.pi * t / (365 / 1))
                + arg[11] * np.cos(2 * np.pi * t / (365 / 1))
                + arg[12] * np.sin(2 * np.pi * t / 23.9)
                + arg[13] * np.cos(2 * np.pi * t / 23.9)
                + arg[14] * np.sin(2 * np.pi * t / 91.3)
                + arg[15] * np.cos(2 * np.pi * t / 91.3)

        )


    t = np.array(range(0, len(EAM6), 1))
    y = np.array(EAM6)

    # 进行最小二乘拟合
    params, y_fit2 = lsq_func_lod(y, t)

    # 生成新的时间范围，包括原始时间和额外的90天
    t_ext = np.arange(t[0], t[-1] + 85, 1)

    # 使用拟合参数进行外推
    eam90 = func_lod(params, t_ext)

    residuals2 = y - y_fit2

    # -----------------------------------------------------------------------------------------------------
    # 检验平稳性
    result = adfuller(residuals2)
    # 确定最佳的AR模型的期数
    aic_values = []
    bic_values = []
    max_lags = 80  # 最大滞后期数

    for lag in range(1, max_lags + 1):
        model = AutoReg(residuals2, lags=lag)
        model_fit = model.fit()
        aic_values.append(model_fit.aic)
        bic_values.append(model_fit.bic)

    # 创建一个DataFrame以便于分析
    results = pd.DataFrame(
        {"Lag": range(1, max_lags + 1), "AIC": aic_values, "BIC": bic_values}
    )

    # 找到最优的滞后期
    best_aic_lag = results.loc[results["AIC"].idxmin()]["Lag"]
    best_bic_lag = results.loc[results["BIC"].idxmin()]["Lag"]

    # 拟合AR模型
    model = AutoReg(residuals2, lags=80)
    model_fit = model.fit()

    # 打印模型摘要
    print(model_fit.summary())

    # 进行预测
    predictions6 = model_fit.predict(
        start=len(residuals2), end=len(residuals2) + 83, dynamic=False
    )
    ccxe90 = np.concatenate((residuals2, predictions6))
    EAM90 = eam90 + ccxe90

    GAM90 = gam_eam90 + EAM90

    # ------------------反向刘维尔方程（GAM PRED转换到LODR PRED）(积分转换成 UT1 PRED)------------------

    GAMfull90 = GAM90 * (2 * np.pi) / 7.292115e-5

    # -----------------------------加上潮汐---------------------------------
    # --------------------------------------------------------------------------------------

    LODZCX = GAMfull90 + DLOD_values1

    # ---------------------------------积分------------------------------------
    initial_value = UT1_TAI[0]


    def integrate_with_initial(diff_data, initial_value):
      
        return -np.cumsum(diff_data) + initial_value  # 累加差分结果并添加初始值


    # 对向后差分结果进行积分
    UT1CXX = integrate_with_initial(GAMfull90, initial_value) + DUT_values1 + leap_seconds90

    # ---------------------------平绝对误差MAE------------------------------------

    # ---------------------------每一次循环的数据写入新的txt文件中-----------------------------------------------------------------

    # ---------------------------平均绝对误差MAE------------------------------------

    # 示例数据
    y_true_lod = np.array(lod2[-90:])  # 真实值
    y_pred_lod = np.array(LODZCX[-90:])  # 预测值
    mae_lod1 = np.abs([y_true_lod[i] - y_pred_lod[i] for i in range(len(y_pred_lod))]) * 10 ** 3

    # 示例数据
    y_true_ut = np.array(UT4_UTC[-90:])  # 真实值
    y_pred_ut = np.array(UT1CXX[-90:])  # 预测值
    mae_ut2 = np.abs([y_true_ut[i] - y_pred_ut[i] for i in range(len(y_pred_ut))]) * 10 ** 3

    plt.figure(1)
    plt.subplot(3, 3, 1)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.plot(mae_lod1[0:6])

    plt.subplot(3, 3, 2)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.plot(mae_lod1[0:90])

    plt.subplot(3, 3, 4)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.plot(mae_ut2[0:6])

    # plt.legend()

    plt.subplot(3, 3, 5)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.plot(mae_ut2[0:90])

    plt.subplot(3, 3, 7)
    plt.plot(lod2)
    plt.plot(LODZCX)

    plt.subplot(3, 3, 8)
    plt.plot(UT4_UTC)
    plt.plot(UT1CXX)

    plot_filename = os.path.join(output_folder, f'{plot_counter}.png')
    plt.savefig(plot_filename)
    plt.close()
    #
    # 每次循环后增加计数器
    plot_counter += 1

    # ---------------------------每一次循环的数据写入新的txt文件中-----------------------------------------------------------------

    # 构建文件名，例如 "D:\日志变化\output_0.txt", "D:\日志变化\output_1.txt", ...
    filename = os.path.join(log_folder2, f"output_{g}.txt")

    # 将结果写入文件
    with open(filename, "w", encoding="utf-8") as file:
        for value in y_pred_lod:
            file.write(f"{value}\n")

    # 构建文件名，例如 "D:\日志变化\output_0.txt", "D:\日志变化\output_1.txt", ...
    filename = os.path.join(log_folder1, f"output_{g}.txt")

    # 将结果写入文件
    with open(filename, "w", encoding="utf-8") as file:
        for value in y_pred_ut:
            file.write(f"{value}\n")

    print(f"第{g}次预测")







