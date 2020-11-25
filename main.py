# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
import tensorflow as tf

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def read_and_merge_data():
    steamData = pd.read_csv('steam.csv')
    steamSpyData = pd.read_csv('steamspy_tag_data.csv')
    # print('steamData size: ', steamData.shape)
    # print('steamSpyData size: ', steamSpyData.shape)
    # mergedDataframe = steamData.merge(steamSpyData, how='appid')
    mergedDataframe = pd.merge(steamData, steamSpyData, on='appid')
    # print('mergedDataframe size: ', mergedDataframe.shape)
    return mergedDataframe


# def remap_column(column_name):
#     x = dataframe[column_name].value_counts()
#     item_type_mapping = {}
#     item_list = x.index
#     for i in range(0, len(item_list)):
#         item_type_mapping[item_list[i]] = i
#
#     res = new('Result', {
#         'frame': dataframe[column_name].map(lambda x: item_type_mapping[x]),
#         'dictionary': item_type_mapping
#     })
#     return res


def new_remap_column(column_name, frame):
    z = frame[column_name].value_counts()
    dict1 = z.to_dict()  # converts to dictionary
    frame[column_name] = frame[column_name].map(dict1)

    res = new('Result', {
        'frame': frame[column_name],
        'dictionary': dict1
    })
    return res


def new(name, data):
    return type(name, (object,), data)


def remap_values():
    string_columns = list(dataframe.select_dtypes(include=['object']).columns)
    print('string_columns', string_columns)
    # string_columns['name', 'release_date', 'developer', 'publisher', 'platforms', 'categories', 'genres', 'owners']

    dataframe['release_date'] = pd.to_datetime(dataframe['release_date'])
    dataframe['release_date'] = (datetime.datetime.now() - dataframe['release_date']).dt.days

    result = new_remap_column('developer', dataframe)
    dataframe['developer'] = result.frame
    developer_dict = result.dictionary
    # #
    result = new_remap_column('publisher', dataframe)
    dataframe['publisher'] = result.frame
    publisher_dict = result.dictionary
    #
    result = new_remap_column('platforms', dataframe)
    dataframe['platforms'] = result.frame
    platforms_dict = result.dictionary
    #
    result = new_remap_column('categories', dataframe)
    dataframe['categories'] = result.frame
    categories_dict = result.dictionary
    #
    result = new_remap_column('genres', dataframe)
    dataframe['genres'] = result.frame
    genres_dict = result.dictionary
    #
    result = new_remap_column('owners', dataframe)
    dataframe['owners'] = result.frame
    owners_dict = result.dictionary

    res = new('Result', {
        'dataframe': dataframe,
        'developer_dict': developer_dict,
        'publisher_dict': publisher_dict,
        'platforms_dict': platforms_dict,
        'categories_dict': categories_dict,
        'genres_dict': genres_dict,
        'owners_dict': owners_dict,
    })
    return res


def analyse_dataframe(dframe):
    print('HEAD', dframe.head())
    print('INFO', dframe.info())
    print('DESCRIBE', dframe.describe())
    dframe = dframe[['release_date', 'english', 'developer', 'publisher', 'platforms',
     'required_age', 'categories', 'genres', 'positive_ratings', 'negative_ratings',
     'average_playtime', 'owners', 'price', '2.5d', '2d', '3d', 'action_rpg',
     'action_adventure', 'adventure', 'alternate_history', 'anime', 'arcade',
     'assassin', 'batman', 'battle_royale', 'blood', 'survival', 'western',
     'board_game', 'card_game', 'cartoon', 'cinematic', 'co_op', 'co_op_campaign',
     'competitive', 'comedy', 'crime', 'detective', 'difficult', 'education',
     'emotional', 'fantasy', 'historical', 'indie', 'kickstarter',
     'lara_croft', 'multiplayer', 'realistic', 'sexual_content', 'simulation',
     'singleplayer', 'sports', 'world_war_i', 'world_war_ii', 'e_sports']].copy()
    print('HEAD after', dframe.head())
    print('INFO after', dframe.info())
    print('DESCRIBE after', dframe.describe())
    # del dataframe['appid']
    # del dataframe['name']
    # del dataframe['achievements']  # mna to ani nebavi zbierat na steame plus nie kazda hra to ma
    return dframe


def back_remap(frame, data):
    tmp_dataframe = frame.copy()
    tmp_dataframe.developer = tmp_dataframe.developer.astype(int)
    tmp_dataframe.publisher = tmp_dataframe.publisher.astype(int)
    tmp_dataframe.platforms = tmp_dataframe.platforms.astype(int)
    tmp_dataframe.categories = tmp_dataframe.categories.astype(int)
    tmp_dataframe.genres = tmp_dataframe.genres.astype(int)
    tmp_dataframe.owners = tmp_dataframe.owners.astype(int)

    tmp_dataframe['developer'] = tmp_dataframe['developer'].map({v: k for k, v in data.developer_dict.items()})
    tmp_dataframe['publisher'] = tmp_dataframe['publisher'].map({v: k for k, v in data.publisher_dict.items()})
    tmp_dataframe['platforms'] = tmp_dataframe['platforms'].map({v: k for k, v in data.platforms_dict.items()})
    tmp_dataframe['categories'] = tmp_dataframe['categories'].map({v: k for k, v in data.categories_dict.items()})
    tmp_dataframe['genres'] = tmp_dataframe['genres'].map({v: k for k, v in data.genres_dict.items()})
    tmp_dataframe['owners'] = tmp_dataframe['owners'].map({v: k for k, v in data.owners_dict.items()})
    return tmp_dataframe


def figure_graph(clusters, start, end):
    fig2_columns = []
    fig2_columns.extend(clusters.iloc[:, start:end].columns.values)
    fig2 = make_subplots(rows=6, cols=5,
                         subplot_titles=fig2_columns
                         )
    fig2_clmn = list(clusters.iloc[:, start:end])
    row = 1
    col = 1
    for i in fig2_clmn:
        fig2.add_trace(
            go.Bar(x=clusters[i].index, y=clusters[i].values),
            row=row, col=col
        )
        col += 1
        if col == 6:
            col = 1
            row += 1

    return fig2


def viz_clusters(frame, id_name):
    counts = frame[id_name].value_counts()
    clusters = frame.groupby([id_name]).mean()

    fig1_columns = ['clusters_count']
    fig1_columns.extend(clusters.iloc[:, 0:29].columns.values)
    fig1 = make_subplots(rows=6, cols=5,
                         subplot_titles=fig1_columns
                         )
    fig1.add_trace(
        go.Bar(x=counts.index, y=counts.values),
        row=1, col=1
    )
    fig1_clmn = list(clusters.iloc[:, 0:29])
    row = 1
    col = 2
    for i in fig1_clmn:
        fig1.add_trace(
            go.Bar(x=clusters[i].index, y=clusters[i].values),
            row=row, col=col
        )
        col += 1
        if col == 6:
            col = 1
            row += 1

    fig2 = figure_graph(clusters, 29, 57)
    # fig3 = figure_graph(clusters, 99, 149)
    # fig4 = figure_graph(clusters, 149, 199)
    # fig5 = figure_graph(clusters, 199, 249)
    # fig6 = figure_graph(clusters, 249, 299)
    # fig7 = figure_graph(clusters, 299, 349)
    # fig8 = figure_graph(clusters, 349, 380)

    fig1.show()
    fig2.show()
    # fig3.show()
    # fig4.show()
    # fig5.show()
    # fig6.show()
    # fig7.show()
    # fig8.show()

    #add figures for  -> release_date, developer, publisher, platforms, categories, genres, owners


def k_means_solution():
    print('=====STARTING k means CLUSTERING=====')
    km = MiniBatchKMeans(5, init='k-means++', random_state=0)
    km_model = km.fit(frame_dicts.dataframe)
    kmeans_labels = km_model.labels_
    frame_dicts.dataframe = scaler.inverse_transform(frame_dicts.dataframe)

    frame_dicts.dataframe = pd.DataFrame(frame_dicts.dataframe, index=index, columns=columns)
    kmeans_dataframe = frame_dicts.dataframe.copy()

    pca = PCA(n_components=3)
    components = pca.fit_transform(kmeans_dataframe)

    kmeans_dataframe['cluster_id'] = kmeans_labels
    kmeans_dataframe['name'] = names
    # kmeans_dataframe = back_remap(kmeans_dataframe, frame_dicts)

    # data visualization
    total_var = pca.explained_variance_ratio_.sum() * 100

    fig3D = px.scatter_3d(
        components, x=0, y=1, z=2, color=kmeans_dataframe['cluster_id'], hover_name=kmeans_dataframe['name'],
        title=f'Total Explained Variance: {total_var:.2f}%',
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
    )
    fig3D.show()

    viz_clusters(kmeans_dataframe, 'cluster_id')
    print('=====ENDING k means CLUSTERING=====')
    return kmeans_dataframe


def dbscan_solution():
    print('=====STARTING dbscan CLUSTERING=====')
    dbscan = DBSCAN(eps=5, min_samples=3)
    dbscan_lables = dbscan.fit_predict(frame_dicts.dataframe)

    frame_dicts.dataframe = scaler.inverse_transform(frame_dicts.dataframe)

    frame_dicts.dataframe = pd.DataFrame(frame_dicts.dataframe, index=index, columns=columns)
    dbscan_dataframe = frame_dicts.dataframe.copy()

    tsne = TSNE(n_components=3)
    components = tsne.fit_transform(dbscan_dataframe)

    dbscan_dataframe['dbscan_id'] = dbscan_lables
    dbscan_dataframe['name'] = names
    # dbscan_dataframe = back_remap(dbscan_dataframe, frame_dicts)

    # data visualization
    fig3D = px.scatter_3d(
        components, x=0, y=1, z=2, color=dbscan_dataframe['dbscan_id'], hover_name=dbscan_dataframe['name'],
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
    )
    fig3D.show()

    viz_clusters(dbscan_dataframe, 'dbscan_id')
    print('=====ENDING dbscan CLUSTERING=====')
    return dbscan_dataframe


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataframe = read_and_merge_data()
    start_dataframe = dataframe.copy()

    index = dataframe.index
    names = dataframe.name
    dataframe = analyse_dataframe(dataframe)

    columns = dataframe.columns

    #remap string values to int
    frame_dicts = remap_values()

    #normalize data
    scaler = StandardScaler()
    frame_dicts.dataframe = scaler.fit_transform(frame_dicts.dataframe)

    # k means code
    kmeans_dataframe = k_means_solution()

    # dbscan code
    dbscan_dataframe = dbscan_solution()


