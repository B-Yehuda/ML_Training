from concurrent.futures import ProcessPoolExecutor
import networkx as nx
import numpy as np
import pandas as pd
import warnings
import logging
from src.projects.twitch_community_detection.settings.config_settings import TwitchFCommunityDetectionConfig
from src.ml_projects_utils.writing_output_utils import write_output, copy_from_aws_to_redshift_table
from src.ml_projects_utils.sql_utils import read_sql_file
from src.vault import get_vault_secrets
from src.load_and_process_data import connect_redshift, load_data, load_local_credentials

from dataclasses import dataclass
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')


@dataclass
class CommunitiesSelectorConf:
    deployment_type: str
    feature: str
    feature_size: int
    df_communities_features: pd.DataFrame


@dataclass
class ChannelsSelectorConf:
    deployment_type: str
    feature: str
    selected_communities: list
    df_channel_community_feature: pd.DataFrame


@dataclass
class ExtractNeighborsConf:
    g: nx.classes.graph.Graph
    seed_channels: list
    seed: str
    deployment_type: str
    feature: str
    n_neighbors_per_channel: int


def filter_highly_connected_channels(df_edges, percentile_threshold) -> pd.DataFrame:
    # define threshold
    interactions_threshold = np.percentile(df_edges["cnt"], [percentile_threshold])[0]

    # count all channels
    all_channels = set(df_edges['ch1']).union(set(df_edges['ch2']))

    # count big channels (i.e. where cnt > defined threshold)
    big_channels = set(df_edges['ch1'][df_edges["cnt"] > interactions_threshold]).union(
        set(df_edges['ch2'][df_edges["cnt"] > interactions_threshold]))

    # print stat
    print(f"\nData stat:")
    print(f"- Number of edges (interactions) is: \033[1m{len(df_edges)}\033[0m")
    print(f"- Number of nodes (channels) is: \033[1m{len(all_channels)}\033[0m")

    print(
        f"- Number of channels with (#) shared chatters "
        f"above percentile threshold of {round(percentile_threshold, 2)}% "
        f"is: \033[1m{len(big_channels)}\033[0m "
        f"(i.e. \033[1m{100 * len(big_channels) / len(all_channels)}\033[0m% of total channels)")

    # remove big channels
    df_edges = df_edges[~df_edges['ch1'].isin(big_channels)]
    df_edges = df_edges[~df_edges['ch2'].isin(big_channels)]

    # count remain channels
    remain_channels = set(df_edges['ch1']).union(set(df_edges['ch2']))

    # print stat
    print(f"\nData stat (after filtering highly connected channels):")
    print(f"- Number of remain edges (interactions) is: \033[1m{len(df_edges)}\033[0m")
    print(f"- Number of remain nodes (channels) is: \033[1m{len(remain_channels)}\033[0m")

    return df_edges


def extract_usernames(df_edges, cur) -> pd.DataFrame:
    channels = set(df_edges.ch1).union(set(df_edges.ch2))
    channels = ', '.join([f"'{c}'" for c in channels])
    cur.execute(f""" 
        SELECT provider_id AS channel_id
              , username
         FROM bi_db.creators
         WHERE provider = 'twitch' AND 
               provider_id in ({channels})
    """)
    df_usernames = pd.DataFrame(cur.fetchall())
    df_usernames.columns = [desc[0] for desc in cur.description]

    df_usernames = df_usernames.set_index('channel_id')
    df_usernames.head()

    return df_usernames


def create_network(df_edges):
    # define the graph
    g = nx.from_pandas_edgelist(df_edges[['ch1', 'ch2', 'interaction']],
                                source='ch1',
                                target='ch2',
                                edge_attr=['interaction'])

    # print stat
    print(f"\nNetwork stat:")
    print(f"- {g}")

    return g


def community_detection(g):
    # print stat
    print(f"\nLouvain Communities Detection algorithm stat:")
    print(f"- Training model - started at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")

    start = time.time()
    communities = nx.algorithms.community.louvain_communities(g, weight='interaction', resolution=1)
    end = time.time()

    # print stat
    print(f"- Training model - finished at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")
    print(f"- Total training time: \033[1m{round(end - start, 2)} seconds\033[0m")
    print(f"- Number of communities detected: \033[1m{len(communities)}\033[0m\n")

    return communities


def assign_community_to_channel(communities) -> dict:
    channel_community_dict = dict()
    for i, c in enumerate(communities):
        for channel in c:
            channel_community_dict[channel] = i

    return channel_community_dict


def add_usernames_to_network(g, df_usernames):
    # channels and usernames
    channel_ids = [n for n in g.nodes()]
    usernames = df_usernames['username'].reindex(channel_ids).to_dict()

    # add each node data - username (values = dict of usernames sorted by nodes)
    nx.set_node_attributes(g, values=usernames, name='username')


def add_communities_data_to_network(g, channel_community_dict):
    # add each node data - community data
    nx.set_node_attributes(g, values=channel_community_dict, name='community')

    # # add each edge data - weight data (weight = UDC considering graph's weight ('interaction') and community)
    # for node1, node2, e in g.edges(data=True):
    #     if g.nodes[node2]['community'] == g.nodes[node1]['community']:
    #         e['weight'] = e['interaction'] * 2
    #     else:
    #         e['weight'] = e['interaction'] / 2


def communities_selector(conf: CommunitiesSelectorConf) -> dict:
    selected_communities = []

    for community in conf.df_communities_features['community'].unique():
        community_mask = conf.df_communities_features['community'] == community
        community_size_mask = conf.df_communities_features['community_size'] >= 10
        # deployments_mask = df_communities_features['deployments'] >= 100
        deployment_type_mask = conf.df_communities_features['deployment_type'] == conf.deployment_type
        feature_mask = conf.df_communities_features['feature'] == conf.feature
        feature_size_mask = conf.df_communities_features['feature_size'] >= conf.feature_size

        if not conf.df_communities_features[community_mask & community_size_mask &
                                            deployment_type_mask &
                                            feature_mask & feature_size_mask].empty:
            selected_communities.append(community)

    selected_communities = [*set(selected_communities)]

    if selected_communities:
        return {'deployment_type': conf.deployment_type,
                'feature': conf.feature,
                'selected_communities': selected_communities}
    else:
        return {}


def channels_selector(conf: ChannelsSelectorConf) -> dict:
    seed_channels = []

    for community in conf.selected_communities:
        channel_community_mask = conf.df_channel_community_feature['community'] == community
        deployment_type_mask = conf.df_channel_community_feature['deployment_type'] == conf.deployment_type
        feature_mask = conf.df_channel_community_feature['feature'] == conf.feature
        # extract seed channels
        seed_channels.extend(
            conf.df_channel_community_feature[channel_community_mask & deployment_type_mask & feature_mask][
                'channel'].values)

    seed_channels = [*set(seed_channels)]

    if seed_channels:
        return {'deployment_type': conf.deployment_type,
                'feature': conf.feature,
                'seed_channels': seed_channels}
    else:
        return {}


def extract_neighbors(conf: ExtractNeighborsConf) -> pd.DataFrame:
    # extract node's neighbors (using both network graph and community detection)
    neighbors = []

    for neighbor in nx.neighbors(conf.g, conf.seed):
        if conf.g.nodes[neighbor]['community'] == conf.g.nodes[conf.seed]['community'] and \
                neighbor not in conf.seed_channels:
            edge = conf.g.edges[conf.seed, neighbor]
            weight = edge['interaction']
            neighbors.append({'channel': conf.seed,
                              'neighbor': neighbor,
                              'weight': weight,
                              'deployment_type': conf.deployment_type,
                              'feature': conf.feature
                              })
    if neighbors:
        return pd.DataFrame(neighbors).sort_values('weight', ascending=False).head(
            conf.n_neighbors_per_channel).reset_index(drop=True)


def twitch_community_detection_pipeline(config_objects,
                                        schema_interim,
                                        is_filter_highly_connected_channels,
                                        is_add_usernames):
    for config in config_objects.values():
        config = config[0]

    # Retrieve dats from config file
    twitch_community_detection_config = TwitchFCommunityDetectionConfig(config)

    print("--------------------------------------------------------------------------------------------------------")
    print(
        f"Pipeline {twitch_community_detection_config.pipeline_name} using {twitch_community_detection_config.pipeline} - started at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")
    print("--------------------------------------------------------------------------------------------------------")

    # retrieve redshift credentials
    if twitch_community_detection_config.location_for_writing_communities_data in {"GCS", "AWS"}:
        redshift_credentials = get_vault_secrets()
    elif twitch_community_detection_config.location_for_writing_communities_data in {"LOCAL_CSV", "LOCAL_JSON",
                                                                                     "REDSHIFT",
                                                                                     "GCS_VIA_LOCAL", "AWS_VIA_LOCAL"}:
        redshift_credentials = load_local_credentials(file_name="redshift_config.json")
    else:
        raise ValueError("No location_for_writing_model was specified in the config file")

    # connect redshift
    engine, conn, cur = connect_redshift(credentials=redshift_credentials)

    # load data from redshift
    df_edges = load_data(cur=cur, config=config)
    df_edges.to_csv("df_edges.csv", index=False)

    # filter highly connected channels
    if is_filter_highly_connected_channels:
        df_edges = filter_highly_connected_channels(df_edges=df_edges, percentile_threshold=99.9)

    # create network
    g = create_network(df_edges=df_edges)

    # run community detection algorithm
    communities = community_detection(g)

    # create channel_community_dict contains {ch: community}
    channel_community_dict = assign_community_to_channel(communities=communities)

    # add usernames data to the network
    if is_add_usernames:
        df_usernames = extract_usernames(df_edges=df_edges, cur=cur)
        add_usernames_to_network(g=g, df_usernames=df_usernames)

    # add communities data to the network
    add_communities_data_to_network(g=g, channel_community_dict=channel_community_dict)

    # save results in df
    df_channel_community = pd.DataFrame(channel_community_dict.items(), columns=['channel', 'community'])

    # write df_channel_community into AWS bucket
    object_name = "channel_community"
    aws_file_path = write_output(
        location_for_writing_output=twitch_community_detection_config.location_for_writing_communities_data,
        engine=engine,
        conn=conn,
        cur=cur,
        schema_name=schema_interim,
        df=df_channel_community,
        object_name=object_name,
        database_name=None,
        skip_if_exists=False,
        drop_if_exists=True,
        create_table_from_sql_or_df="df",
        buckets=[twitch_community_detection_config.predictions_to_aws_bucket],
        config=config
    )

    # write df_channel_community into Redshift table
    copy_from_aws_to_redshift_table(conn=conn,
                                    cur=cur,
                                    schema=schema_interim,
                                    table_name=object_name,
                                    columns_and_types_dict={'channel': 'VARCHAR(255)', 'community': 'INTEGER'},
                                    skip_if_exists=False,
                                    drop_if_exists=True,
                                    bucket_name=twitch_community_detection_config.predictions_to_aws_bucket,
                                    aws_file_path=aws_file_path)

    # aggregate df_channel_community table
    print(f"\nAggregating \033[1m{schema_interim}.{object_name}\033[0m table:")

    df_channel_community_feature = pd.read_sql(sql=read_sql_file(name="df_channel_community_feature"), con=conn)
    df_channel_community_feature.to_csv("df_channel_community_feature.csv", index=False)
    print(f"- Created \033[1mdf_channel_community_feature\033[0m")

    df_communities_features = pd.read_sql(sql=read_sql_file(name="df_communities_features"), con=conn)
    df_communities_features.to_csv("df_communities_features.csv", index=False)
    print(f"- Created \033[1mdf_communities_features\033[0m")

    is_parallel_search = False
    # neighbors parallel search
    if is_parallel_search:
        print(f"\nNeighbors parallel search - started at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m\n")

        with ProcessPoolExecutor() as executor:
            # select communities
            selected_communities_dicts = [sel_c_dict for sel_c_dict in
                                          executor.map(communities_selector,
                                                       [CommunitiesSelectorConf(
                                                           deployment_type=features_data["deployment_type"],
                                                           feature=features_data["feature"],
                                                           feature_size=int(features_data["feature_size"]),
                                                           df_communities_features=df_communities_features)
                                                           for features_data in
                                                           twitch_community_detection_config.features_data_dicts]
                                                       )]
            selected_communities_dicts = [item for item in selected_communities_dicts if item]

            # abort if no communities were found
            if not selected_communities_dicts:
                print(f"No communities were found for \033[1mall features\033[0m under the given conditions, aborting.")
                return

            # select seed channels
            seed_channels_dicts = [seed_c_dict for seed_c_dict in
                                   executor.map(channels_selector,
                                                [ChannelsSelectorConf(
                                                    deployment_type=sel_c_dict["deployment_type"],
                                                    feature=sel_c_dict["feature"],
                                                    selected_communities=sel_c_dict["selected_communities"],
                                                    df_channel_community_feature=df_channel_community_feature)
                                                    for sel_c_dict in selected_communities_dicts]
                                                )]
            seed_channels_dicts = [item for item in seed_channels_dicts if item]

            # abort empty lists
            if not seed_channels_dicts:
                print(
                    f"No seed channels were found for \033[1mall features\033[0m under the given conditions, aborting.")
                return

            # search neighbors of seed channels
            df_neighbors = pd.concat(list(
                executor.map(extract_neighbors,
                             [ExtractNeighborsConf(
                                 g=g,
                                 seed_channels=seed_c_dict["seed_channels"],
                                 seed=seed,
                                 deployment_type=seed_c_dict["deployment_type"],
                                 feature=seed_c_dict["feature"],
                                 n_neighbors_per_channel=twitch_community_detection_config.n_neighbors_per_channel * 2)
                                 for seed_c_dict in seed_channels_dicts for seed in seed_c_dict["seed_channels"]]
                             )))

            if not df_neighbors.empty:
                # sort df columns
                df_neighbors = df_neighbors[['channel', 'neighbor', 'weight', 'deployment_type', 'feature']]

                # for each neighbor - choose the highest interaction of interactions with channels (neighbor)
                df_neighbors = df_neighbors.sort_values('weight', ascending=False).drop_duplicates(
                    'neighbor').sort_index()

                # add neighbor_rnk column
                df_neighbors['neighbor_rnk'] = df_neighbors.groupby(['deployment_type', 'feature', 'channel'])[
                    'weight'].rank(
                    method="first", ascending=False).astype(int)

                # select n_neighbors_per_channel
                df_neighbors = df_neighbors[
                    df_neighbors['neighbor_rnk'].between(1, twitch_community_detection_config.n_neighbors_per_channel)]

                df_neighbors.to_csv(f"df_neighbors_parallel_search.csv", index=False)

                print(
                    f"\nNeighbors parallel search - finished at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")

            else:
                print(f"\nNo neighbors were found, aborting.")
                return

    is_concurrent_search = True
    # neighbors concurrent search
    if is_concurrent_search:
        print(f"\nNeighbors concurrent search - started at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m\n")

        # initialize empty df to store neighbors
        df_neighbors = pd.DataFrame()

        # fetch number of neighbors to return for each channel from config file
        n_neighbors_per_channel = int(config["Data_Processing"].get("n_neighbors_per_channel"))

        for features_data in twitch_community_detection_config.features_data_dicts:
            deployment_type, feature, feature_size = features_data.values()

            # select communities
            selected_communities_dicts = \
                communities_selector(CommunitiesSelectorConf(deployment_type=deployment_type,
                                                             feature=feature,
                                                             feature_size=int(feature_size),
                                                             df_communities_features=df_communities_features))

            # skip if no communities were found
            if not selected_communities_dicts:
                print(
                    f"No communities were found for \033[1m{deployment_type}-{feature}\033[0m under the given conditions, skipping.")
                continue

            # select seed channels
            seed_channels_dicts = \
                channels_selector(ChannelsSelectorConf(deployment_type=deployment_type,
                                                       feature=feature,
                                                       selected_communities=selected_communities_dicts[
                                                           'selected_communities'],
                                                       df_channel_community_feature=df_channel_community_feature))

            if not seed_channels_dicts:
                raise ValueError("\n No seed channels were found even though selected_communities exists")

            # search neighbors of seed channels
            for seed in seed_channels_dicts['seed_channels']:
                neighbors = extract_neighbors(ExtractNeighborsConf(g=g,
                                                                   seed_channels=seed_channels_dicts['seed_channels'],
                                                                   seed=seed,
                                                                   deployment_type=deployment_type,
                                                                   feature=feature,
                                                                   n_neighbors_per_channel=n_neighbors_per_channel * 2))
                df_neighbors = pd.concat((df_neighbors, neighbors), axis=0)

        if not df_neighbors.empty:
            # sort df columns
            df_neighbors = df_neighbors[['channel', 'neighbor', 'weight', 'deployment_type', 'feature']]

            # for each neighbor - choose the highest interaction of interactions with channels (neighbor)
            df_neighbors = df_neighbors.sort_values('weight', ascending=False).drop_duplicates('neighbor').sort_index()

            # add neighbor_rnk column
            df_neighbors['neighbor_rnk'] = df_neighbors.groupby(['deployment_type', 'feature', 'channel'])['weight'] \
                .rank(method="first", ascending=False).astype(int)

            # select n_neighbors_per_channel neighbors
            df_neighbors = df_neighbors[df_neighbors['neighbor_rnk'].between(1, n_neighbors_per_channel)]

            df_neighbors.to_csv(f"df_neighbors_concurrent_search.csv", index=False)

            print(
                f"\nNeighbors concurrent search - finished at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")

            # write df_neighbors into AWS bucket
            object_name = "neighbors"
            aws_file_path = write_output(location_for_writing_output=twitch_community_detection_config.location_for_writing_communities_data,
                                         engine=engine,
                                         conn=conn,
                                         cur=cur,
                                         schema_name=schema_interim,
                                         df=df_neighbors,
                                         object_name=object_name,
                                         database_name=None,
                                         skip_if_exists=False,
                                         drop_if_exists=True,
                                         create_table_from_sql_or_df="df",
                                         buckets=[twitch_community_detection_config.predictions_to_aws_bucket],
                                         config=config
                                         )

            # write df_neighbors into Redshift table
            copy_from_aws_to_redshift_table(conn=conn,
                                            cur=cur,
                                            schema=schema_interim,
                                            table_name=object_name,
                                            columns_and_types_dict={'channel': 'VARCHAR(255)',
                                                                    'neighbor': 'VARCHAR(255)',
                                                                    'weight': 'FLOAT',
                                                                    'deployment_type': 'VARCHAR(255)',
                                                                    'feature': 'VARCHAR(255)',
                                                                    'neighbor_rnk': 'INTEGER'},
                                            skip_if_exists=False,
                                            drop_if_exists=True,
                                            bucket_name=twitch_community_detection_config.predictions_to_aws_bucket,
                                            aws_file_path=aws_file_path)

        else:
            print(f"\nNo neighbors were found, aborting.")
            return

    is_sub_communities_search = False
    if is_sub_communities_search:
        # select communities to search for sub communities
        selected_communities = []
        for community in df_communities_features['community']:
            community_mask = df_communities_features['community'] == community
            tier_1_mask = df_communities_features['tier'] == 'tier_1'
            tier_2_mask = df_communities_features['tier'] == 'tier_2'
            tier_3_mask = df_communities_features['tier'] == 'tier_3'
            if df_communities_features[community_mask]['community_size'].tolist()[0] >= 5 and \
                    df_communities_features[community_mask]['deployers_size'].tolist()[0] >= 100 and \
                    df_communities_features[community_mask & tier_1_mask]['tier_pct_of_deployers'].tolist()[0] < 0.1 and \
                    df_communities_features[community_mask & tier_3_mask]['tier_pct_of_deployers'].tolist()[0] > 0.5:
                selected_communities.append(community)
        selected_communities = [*set(selected_communities)]

        # store data of each selected community
        selected_communities_data = []
        for i, community in enumerate(selected_communities):
            selected_communities_data.append({})
            mask = df_channel_community['community'] == community
            # store community
            selected_communities_data[i]['community'] = \
                community
            # store channels
            selected_communities_data[i]['channels'] = \
                df_channel_community[mask]['channel'].values
            # store edges
            selected_communities_data[i]['edges'] = \
                df_edges[
                    df_edges['ch1'].isin(selected_communities_data[i]['channels']) &
                    df_edges['ch2'].isin(selected_communities_data[i]['channels'])
                    ]
            # TODO: Should I add .copy() or a reference to df_edges is enough?

        with ProcessPoolExecutor() as executor:
            # create subnetworks
            sub_g = list(executor.map(create_network,
                                      [community_data['edges'] for community_data in selected_communities_data]
                                      ))

            # run sub-community detection algorithm
            sub_communities = list(executor.map(community_detection,
                                                [g for g in sub_g]))

            # create channel_sub_community_dict contains {community: {ch: sub-community}}
            channel_sub_community = list(assign_community_to_channel(sub_c) for sub_c in sub_communities)
            channel_sub_community_dict = dict(zip(selected_communities, channel_sub_community))

            # save results
            df_channel_sub_community = pd.DataFrame([(innerKey, outerKey, f"{outerKey}_{values}")
                                                     for outerKey, innerDict in channel_sub_community_dict.items()
                                                     for innerKey, values in innerDict.items()],
                                                    columns=['channel', 'community', 'sub_community']
                                                    )
            df_channel_sub_community.to_csv("df_channel_sub_community.csv", index=False)

            # # extract neighbors
            # for channel in g.nodes(data=True):
            #     neighbors = extract_neighbors(g, channel)
