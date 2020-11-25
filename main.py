# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
import tensorflow as tf

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def read_and_merge_data():
    steamData = pd.read_csv('steam.csv')
    steamSpyData = pd.read_csv('steamspy_tag_data.csv')
    # print('steamData size: ', steamData.shape)
    # print('steamSpyData size: ', steamSpyData.shape)
    # mergedDataframe = steamData.merge(steamSpyData, how='appid')
    mergedDataframe = pd.merge(steamData, steamSpyData, on='appid')
    # print('mergedDataframe size: ', mergedDataframe.shape)
    return mergedDataframe


def remap_column(column_name):
    x = dataframe[column_name].value_counts()
    item_type_mapping = {}
    item_list = x.index
    for i in range(0, len(item_list)):
        item_type_mapping[item_list[i]] = i

    res = new('Result', {
        'frame': dataframe[column_name].map(lambda x: item_type_mapping[x]),
        'dictionary': item_type_mapping
    })
    return res


def new(name, data):
    return type(name, (object,), data)


def remap_values():
    string_columns = list(dataframe.select_dtypes(include=['object']).columns)
    print('string_columns', string_columns)
    # string_columns['name', 'release_date', 'developer', 'publisher', 'platforms', 'categories', 'genres', 'owners']

    result = remap_column('name')
    dataframe['name'] = result.frame
    name_dict = result.dictionary

    result = remap_column('release_date')
    dataframe['release_date'] = result.frame
    release_date_dict = result.dictionary

    result = remap_column('developer')
    dataframe['developer'] = result.frame
    developer_dict = result.dictionary

    result = remap_column('publisher')
    dataframe['publisher'] = result.frame
    publisher_dict = result.dictionary

    result = remap_column('platforms')
    dataframe['platforms'] = result.frame
    platforms_dict = result.dictionary

    result = remap_column('categories')
    dataframe['categories'] = result.frame
    categories_dict = result.dictionary

    result = remap_column('genres')
    dataframe['genres'] = result.frame
    genres_dict = result.dictionary

    result = remap_column('owners')
    dataframe['owners'] = result.frame
    owners_dict = result.dictionary

    res = new('Result', {
        'dataframe': dataframe,
        'name_dict': name_dict,
        'release_date_dict': release_date_dict,
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
    # print('DTYPES', dataframe.dtypes())
    # print(dataframe['developer'].unique())
    # print(dataframe['developer'].value_counts())


def back_remap(data):
    data.dataframe = pd.DataFrame(data.dataframe, index=index, columns=columns)

    tmp_dataframe = data.dataframe.copy()
    tmp_dataframe.name = tmp_dataframe.name.astype(int)
    tmp_dataframe.release_date = tmp_dataframe.release_date.astype(int)
    tmp_dataframe.developer = tmp_dataframe.developer.astype(int)
    tmp_dataframe.publisher = tmp_dataframe.publisher.astype(int)
    tmp_dataframe.platforms = tmp_dataframe.platforms.astype(int)
    tmp_dataframe.categories = tmp_dataframe.categories.astype(int)
    tmp_dataframe.genres = tmp_dataframe.genres.astype(int)
    tmp_dataframe.owners = tmp_dataframe.owners.astype(int)


    tmp_dataframe['name'] = tmp_dataframe['name'].map({v: k for k, v in data.name_dict.items()})
    tmp_dataframe['release_date'] = tmp_dataframe['release_date'].map({v: k for k, v in data.release_date_dict.items()})
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
    fig2 = make_subplots(rows=10, cols=5,
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


def viz_kmeans_cluster(frame):
    counts = frame['cluster_id'].value_counts()
    clusters = frame.groupby(['cluster_id']).mean()

    fig1_columns = ['clusters_count']
    fig1_columns.extend(clusters.iloc[:, 0:49].columns.values)
    fig1 = make_subplots(rows=10, cols=5,
                         subplot_titles=fig1_columns
                         )
    fig1.add_trace(
        go.Bar(x=counts.index, y=counts.values),
        row=1, col=1
    )
    fig1_clmn = list(clusters.iloc[:, 0:49])
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

    fig2 = figure_graph(clusters, 49, 99)
    fig3 = figure_graph(clusters, 99, 149)
    fig4 = figure_graph(clusters, 149, 199)
    fig5 = figure_graph(clusters, 199, 249)
    fig6 = figure_graph(clusters, 249, 299)
    fig7 = figure_graph(clusters, 299, 349)
    fig8 = figure_graph(clusters, 349, 380)

    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()
    fig5.show()
    fig6.show()
    fig7.show()
    fig8.show()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataframe = read_and_merge_data()

    start_dataframe = dataframe.copy()
    columns = dataframe.columns
    index = dataframe.index

    analyse_dataframe(dataframe)

    #remap string values to int
    frame_dicts = remap_values()
    print('NEW FRAME DICTS OBJ', frame_dicts.dataframe.describe())

    # normalize data
    scaler = StandardScaler()
    frame_dicts.dataframe = scaler.fit_transform(frame_dicts.dataframe)

    #k-means clustering
    print('=====STARTING CLUSTERING=====')
    km = MiniBatchKMeans(8, init='k-means++', random_state=0)
    km_model = km.fit(frame_dicts.dataframe)
    kmeans_labels = km.labels_
    frame_dicts.dataframe = scaler.inverse_transform(frame_dicts.dataframe)

    #return back remaped dataframe
    kmeans_dataframe = back_remap(frame_dicts)
    kmeans_dataframe['cluster_id'] = kmeans_labels

    viz_kmeans_cluster(kmeans_dataframe)