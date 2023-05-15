import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from rich import print
from rich.progress import track
import matplotlib
from sklearn.metrics import silhouette_samples, silhouette_score
import scikitplot as skplt
plt.style.use(['science','no-latex', 'grid'])
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5
# plt.rcParams['xtick.major.size'] = 5
# plt.rcParams['ytick.major.size'] = 5
# plt.rcParams['xtick.minor.size'] = 2
# plt.rcParams['ytick.minor.size'] = 2

# 设置保存图片的格式和dpi
matplotlib.rcParams['savefig.dpi'] = 600
matplotlib.rcParams['savefig.format'] = 'svg'

MIN_SAMPLES = 300

# 调用scikitplot绘图
def plot_silhouette_of_various_clusters(eps_li: list, min_samples: int=MIN_SAMPLES) -> pd.DataFrame:
    """绘制DBSCAN在不同eps下的轮廓系数图, 并保存不同eps下的轮廓系数值和聚类簇数

    Args:
        eps_li (list): eps取值list
        min_samples (int, optional):每簇的最小样本量. Defaults to MIN_SAMPLES.

    Returns:
        pd.DataFrame: 包含不同eps下的轮廓系数值和聚类簇数结果的DataFrame
    """
    silhouette_score_list = []
    for i in eps_li:
        temp_y_predict = DBSCAN(eps=i, min_samples=min_samples, n_jobs=-1).fit_predict(X)
        silhouette_score_ = silhouette_score(X, temp_y_predict)
        add_dict = {"eps": i, "silhouette_score": silhouette_score_, "cluster_num": len([value for value in np.unique(temp_y_predict) if value >= 0])}
        print(add_dict)
        silhouette_score_list.append(add_dict)
        skplt.metrics.plot_silhouette(X, temp_y_predict, figsize=(8,8),title=None, text_fontsize="small")
        # plt.xlim(-0.1, 0.5)
        plt.savefig('./assets/DBSCAN聚类eps为{}时各样本的轮廓系数值.svg'.format(str(i)), bbox_inches='tight')
        # plt.show()
    return pd.DataFrame(silhouette_score_list)


if __name__ == "__main__":
    origin_df = pd.read_excel("./datasets/副本shujulianxi.xlsx", sheet_name=0)
    origin_df['拍摄时间'] = pd.to_datetime(origin_df['拍摄时间']).dt.strftime("%Y-%m-%d")
    origin_df.drop_duplicates(subset=['用户ID', '拍摄时间'], keep="first", inplace=True)
    print("Data Load Success!!!")
    X = origin_df[['经度', '纬度']].astype('float16').iloc[:100000]
    silhouette_score_df = plot_silhouette_of_various_clusters(eps_li=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
    silhouette_score_df.to_csv("./assets/silhouette_score_df.csv", index=False, encoding="utf-8-sig")
    # 数据量太大，只能单个跑
    # silhouette_score_df = plot_silhouette_of_various_clusters(eps_li=[0.15])
    # part_df = pd.read_csv(r"./assets/silhouette_score_df.csv")
    # complete_df = pd.concat([part_df, silhouette_score_df], axis=0, ignore_index=True)
    # complete_df.to_csv("./assets/silhouette_score_df.csv", index=False, encoding="utf-8-sig")
