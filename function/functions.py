import json

import pandas as pd
import numpy as np
import polars as pl

from function.config import *


class MyFunctions:
    @staticmethod
    def preprocessing_pl(x):
        """preprocessiong raw data before feature enginnering."""
        columns = [
            pl.col("page").cast(pl.Float32),
            (
                (pl.col("elapsed_time").diff(1))
                 .fill_null(0)
                 .clip(0, 1e9)
                 .over(["session_id", "level_group"])
                 .alias("elapsed_time_diff")
            ),
            (
                (pl.col("screen_coor_x").diff(1))
                 .abs()
                 .over(["session_id", "level_group"])
                .alias("location_x_diff")
            ),
            (
                (pl.col("screen_coor_y").diff(1))
                 .abs()
                 .over(["session_id", "level_group"])
                .alias("location_y_diff")
            ),
            pl.col("fqid").fill_null("fqid_None"),
            pl.col("text_fqid").fill_null("text_fqid_None")
        ]

        x = (
              x.drop(["fullscreen", "hq", "music"])
              .with_columns(columns))
        return x

    @staticmethod
    def feature_engineer_pl(x, grp, use_extra, feature_suffix):
        """create aggregate features for polars."""
        def except_handling(s):
            try:
                return s.max() - s.min()
            except:
                return 0

        aggs = [
            pl.col("index").count().alias(f"session_number_{feature_suffix}"),
            pl.col("text_fqid").filter(pl.col("text_fqid") == 'text_fqid_None').count().alias(f'null_count'),
            pl.col('fqid').filter(pl.col('fqid').str.starts_with('to')).count().alias('page_change_count'),
            pl.col('index').filter(pl.col('text').str.contains('?', literal=True)).count().alias('question_count'),
            pl.col('index').filter((pl.col("event_name") == 'observation_click') & (pl.col('text_fqid').str.contains('block'))).count().alias('block_count'),
            pl.col('index').filter((pl.col("event_name") == 'person_click') & (pl.col('text_fqid').str.contains('recap'))).count().alias('recap_count'),
            pl.col('index').filter((pl.col("event_name") == 'person_click') & (pl.col('text_fqid').str.contains('lost'))).count().alias('lost_count'),

            *[pl.col(c).drop_nulls().n_unique().alias(f"{c}_unique_{feature_suffix}") for c in LIST_CATS],
            *[pl.col('index').filter(pl.col('text').str.contains(c, literal=True)).count().alias(f'word_{c}') for c in DIALOGS],
            *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c, literal=True))).mean().alias(f'word_mean_{c}') for c in DIALOGS],
            *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c, literal=True))).std().alias(f'word_std_{c}') for c in DIALOGS],
            *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c, literal=True))).max().alias(f'word_max_{c}') for c in DIALOGS],
            *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c, literal=True))).sum().alias(f'word_sum_{c}') for c in DIALOGS],
            *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c, literal=True))).median().alias(f'word_median_{c}') for c in DIALOGS],

            *[pl.col(c).mean().alias(f"{c}_mean_{feature_suffix}") for c in LIST_NUMS],
            *[pl.col(c).median().alias(f"{c}_median_{feature_suffix}") for c in LIST_NUMS],
            *[pl.col(c).std().alias(f"{c}_std_{feature_suffix}") for c in LIST_NUMS],
            *[pl.col(c).min().alias(f"{c}_min_{feature_suffix}") for c in LIST_NUMS],
            *[pl.col(c).max().alias(f"{c}_max_{feature_suffix}") for c in LIST_NUMS],
            *[pl.col(c).sum().alias(f"{c}_sum_{feature_suffix}") for c in LIST_NUMS],

            *[pl.col("event_name").filter(pl.col("event_name") == c).count().alias(f"{c}_event_name_counts{feature_suffix}")for c in LIST_EVENT_NAME_FEATURE],
            *[pl.col("elapsed_time_diff").filter(pl.col("event_name")==c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c in LIST_EVENT_NAME_FEATURE],
            *[pl.col("elapsed_time_diff").filter(pl.col("event_name")==c).median().alias(f"{c}_ET_median_{feature_suffix}") for c in LIST_EVENT_NAME_FEATURE],
            *[pl.col("elapsed_time_diff").filter(pl.col("event_name")==c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in LIST_EVENT_NAME_FEATURE],
            *[pl.col("elapsed_time_diff").filter(pl.col("event_name")==c).max().alias(f"{c}_ET_max_{feature_suffix}") for c in LIST_EVENT_NAME_FEATURE],

            *[pl.col("name").filter(pl.col("name") == c).count().alias(f"{c}_name_counts{feature_suffix}")for c in LIST_NAME_FEATURE],   
            *[pl.col("elapsed_time_diff").filter(pl.col("name")==c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c in LIST_NAME_FEATURE],
            *[pl.col("elapsed_time_diff").filter(pl.col("name")==c).median().alias(f"{c}_ET_median_{feature_suffix}") for c in LIST_NAME_FEATURE],
            *[pl.col("elapsed_time_diff").filter(pl.col("name")==c).max().alias(f"{c}_ET_max_{feature_suffix}") for c in LIST_NAME_FEATURE],
            *[pl.col("elapsed_time_diff").filter(pl.col("name")==c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in LIST_NAME_FEATURE],
            *[pl.col("elapsed_time_diff").filter(pl.col("name")==c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for c in LIST_NAME_FEATURE],

            *[pl.col("room_fqid").filter(pl.col("room_fqid") == c).count().alias(f"{c}_room_fqid_counts{feature_suffix}")for c in LIST_ROOM],
            *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in LIST_ROOM],
            *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c in LIST_ROOM],
            *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).median().alias(f"{c}_ET_median_{feature_suffix}") for c in LIST_ROOM],
            *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for c in LIST_ROOM],
            *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for c in LIST_ROOM],

            *[pl.col("fqid").filter(pl.col("fqid") == c).count().alias(f"{c}_fqid_counts{feature_suffix}")for c in LIST_FQID],
            *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in LIST_FQID],
            *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c in LIST_FQID],
            *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).median().alias(f"{c}_ET_median_{feature_suffix}") for c in LIST_FQID],
            *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for c in LIST_FQID],
            *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for c in LIST_FQID],

            *[pl.col("text_fqid").filter(pl.col("text_fqid") == c).count().alias(f"{c}_text_fqid_counts{feature_suffix}") for c in LIST_TEXT],
            *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in LIST_TEXT],
            *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c in LIST_TEXT],
            *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).median().alias(f"{c}_ET_median_{feature_suffix}") for c in LIST_TEXT],
            *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for c in LIST_TEXT],
            *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for c in LIST_TEXT],

            *[pl.col("location_x_diff").filter(pl.col("event_name") == c).mean().alias(f"{c}_ET_mean_x{feature_suffix}") for c in LIST_EVENT_NAME_FEATURE],
            *[pl.col("location_x_diff").filter(pl.col("event_name") == c).median().alias(f"{c}_ET_median_x{feature_suffix}") for c in LIST_EVENT_NAME_FEATURE],
            *[pl.col("location_x_diff").filter(pl.col("event_name") == c).std().alias(f"{c}_ET_std_x{feature_suffix}") for c in LIST_EVENT_NAME_FEATURE],
            *[pl.col("location_x_diff").filter(pl.col("event_name") == c).max().alias(f"{c}_ET_max_x{feature_suffix}") for c in LIST_EVENT_NAME_FEATURE],
            *[pl.col("location_x_diff").filter(pl.col("event_name") == c).min().alias(f"{c}_ET_min_x{feature_suffix}") for c in LIST_EVENT_NAME_FEATURE],

            *[pl.col("level").filter(pl.col("level") == c).count().alias(f"{c}_LEVEL_count{feature_suffix}") for c in LEVELS],
            *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in LEVELS],
            *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c in LEVELS],
            *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for c in LEVELS],
            *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).median().alias(f"{c}_ET_median_{feature_suffix}") for c in LEVELS],
            *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for c in LEVELS],
            ]

        df = x.groupby(["session_id"], maintain_order=True).agg(aggs).sort("session_id")

        dict_agg_feature_by_level = {
            '0-4':[
                pl.col("elapsed_time").filter(((pl.col('fqid') == 'tunic') & (pl.col('event_name') == 'navigate_click')) | (pl.col('text_fqid') == 'tunic.historicalsociety.collection.tunic.slip')).apply(except_handling).alias("slip_click_duration"),
                pl.col("index").filter(((pl.col('fqid') == 'tunic') & (pl.col('event_name') == 'navigate_click')) | (pl.col('text_fqid') == 'tunic.historicalsociety.collection.tunic.slip')).apply(except_handling).alias("slip_click_indexCount"),
                pl.col("elapsed_time").filter(((pl.col('fqid') == 'plaque') & (pl.col('event_name') == 'navigate_click')) | (pl.col('text_fqid') == 'tunic.kohlcenter.halloffame.plaque.face.date')).apply(except_handling).alias("shirt_era_search_duration"),
                pl.col("index").filter(((pl.col('fqid') == 'plaque') & (pl.col('event_name') == 'navigate_click')) | (pl.col('text_fqid') == 'tunic.kohlcenter.halloffame.plaque.face.date')).apply(except_handling).alias("shirt_era_search_indexCount"),
            ],
            '5-12':[
                pl.col("elapsed_time").filter((pl.col("text")=="Here's the log book.")|(pl.col("fqid")=='logbook.page.bingo')).apply(lambda s: s.max()-s.min()).alias("logbook_bingo_duration"),
                pl.col("index").filter((pl.col("text")=="Here's the log book.")|(pl.col("fqid")=='logbook.page.bingo')).apply(lambda s: s.max()-s.min()).alias("logbook_bingo_indexCount"),
                pl.col("elapsed_time").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='reader'))|(pl.col("fqid")=="reader.paper2.bingo")).apply(lambda s: s.max()-s.min()).alias("reader_bingo_duration"),
                pl.col("index").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='reader'))|(pl.col("fqid")=="reader.paper2.bingo")).apply(lambda s: s.max()-s.min()).alias("reader_bingo_indexCount"),
                pl.col("elapsed_time").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='journals'))|(pl.col("fqid")=="journals.pic_2.bingo")).apply(lambda s: s.max()-s.min()).alias("journals_bingo_duration"),
                pl.col("index").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='journals'))|(pl.col("fqid")=="journals.pic_2.bingo")).apply(lambda s: s.max()-s.min()).alias("journals_bingo_indexCount"),
                pl.col("elapsed_time").filter(((pl.col('fqid') == 'businesscards') & (pl.col('event_name') == 'navigate_click')) | (pl.col('text_fqid') == 'tunic.humanecology.frontdesk.businesscards.card_bingo.bingo')).apply(except_handling).alias("businesscard_bingo_duration"),
                pl.col("index").filter(((pl.col('fqid') == 'businesscards') & (pl.col('event_name') == 'navigate_click')) | (pl.col('text_fqid') == 'tunic.humanecology.frontdesk.businesscards.card_bingo.bingo')).apply(except_handling).alias("businesscard_bingo_indexCount"),

                # add 20230423
                pl.col("elapsed_time").filter(((pl.col('text_fqid') == "tunic.historicalsociety.frontdesk.archivist.need_glass_0")) | (pl.col('text_fqid') == "tunic.historicalsociety.frontdesk.magnify")).apply(except_handling).alias("search_grass_duration"),
                pl.col("index").filter(((pl.col('text_fqid') == "tunic.historicalsociety.frontdesk.archivist.need_glass_0")) | (pl.col('text_fqid') == "tunic.historicalsociety.frontdesk.magnify")).apply(except_handling).alias("search_grass_indexCount"),

                # add 20230506
                pl.col("elapsed_time").filter(pl.col('level_group') == '0-4').last().alias('level_1_last'),
                pl.col("elapsed_time").filter(pl.col('level_group') == '5-12').first().alias('level_2_first'),

                *[pl.col(c).filter(pl.col('level_group') == '5-12').mean().alias(f"{c}_mean_level2") for c in LIST_NUMS],
                *[pl.col(c).filter(pl.col('level_group') == '5-12').median().alias(f"{c}_median_level2") for c in LIST_NUMS],
                *[pl.col(c).filter(pl.col('level_group') == '5-12').std().alias(f"{c}_std_level2") for c in LIST_NUMS],
                *[pl.col(c).filter(pl.col('level_group') == '5-12').min().alias(f"{c}_min_level2") for c in LIST_NUMS],
                *[pl.col(c).filter(pl.col('level_group') == '5-12').max().alias(f"{c}_max_level2") for c in LIST_NUMS],
                *[pl.col(c).filter(pl.col('level_group') == '5-12').sum().alias(f"{c}_sum_level2") for c in LIST_NUMS],

            ],
            '13-22':[
                pl.col("elapsed_time").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='reader_flag'))|(pl.col("fqid")=="tunic.library.microfiche.reader_flag.paper2.bingo")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("reader_flag_duration"),
                pl.col("index").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='reader_flag'))|(pl.col("fqid")=="tunic.library.microfiche.reader_flag.paper2.bingo")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("reader_flag_indexCount"),
                pl.col("elapsed_time").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='journals_flag'))|(pl.col("fqid")=="journals_flag.pic_0.bingo")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("journalsFlag_bingo_duration"),
                pl.col("index").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='journals_flag'))|(pl.col("fqid")=="journals_flag.pic_0.bingo")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("journalsFlag_bingo_indexCount"),
                pl.col("elapsed_time").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='tracks'))|(pl.col("text")=="That hoofprint doesn't match the flag!")).apply(except_handling).alias("tracks_duration"),
                pl.col("index").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='tracks'))|(pl.col("text")=="That hoofprint doesn't match the flag!")).apply(except_handling).alias("tracks_indexCount"),

                # add 20230423
                pl.col("elapsed_time").filter(((pl.col('event_name') == "person_click") & (pl.col('text_fqid') == "tunic.historicalsociety.cage.teddy.trapped")) | ((pl.col('event_name') == "navigate_click") & (pl.col('fqid') == "unlockdoor"))).apply(except_handling).alias("search_key_duration"),
                pl.col("index").filter(((pl.col('event_name') == "person_click") & (pl.col('text_fqid') == "tunic.historicalsociety.cage.teddy.trapped")) | ((pl.col('event_name') == "navigate_click") & (pl.col('fqid') == "unlockdoor"))).apply(except_handling).alias("search_key_indexCount"),

                # add 20230506
                pl.col("elapsed_time").filter(pl.col('level_group') == '5-12').last().alias('level_2_last'),
                pl.col("elapsed_time").filter(pl.col('level_group') == '13-22').first().alias('level_3_first'),

                *[pl.col(c).filter(pl.col('level_group') == '13-22').mean().alias(f"{c}_mean_level3") for c in LIST_NUMS],
                *[pl.col(c).filter(pl.col('level_group') == '13-22').median().alias(f"{c}_median_level3") for c in LIST_NUMS],
                *[pl.col(c).filter(pl.col('level_group') == '13-22').std().alias(f"{c}_std_level3") for c in LIST_NUMS],
                *[pl.col(c).filter(pl.col('level_group') == '13-22').min().alias(f"{c}_min_level3") for c in LIST_NUMS],
                *[pl.col(c).filter(pl.col('level_group') == '13-22').max().alias(f"{c}_max_level3") for c in LIST_NUMS],
                *[pl.col(c).filter(pl.col('level_group') == '13-22').sum().alias(f"{c}_sum_level3") for c in LIST_NUMS],

            ]
        }

        if use_extra:
            if grp == '0-4':
                aggs = dict_agg_feature_by_level['0-4']
                tmp = x.groupby(["session_id"], maintain_order=True).agg(aggs).sort("session_id")
                df = df.join(tmp, on="session_id", how='left')

            if grp == '5-12':
                aggs = dict_agg_feature_by_level['0-4'] + dict_agg_feature_by_level['5-12']

                tmp = (x.groupby(["session_id"], maintain_order=True)
                        .agg(aggs)
                        .sort("session_id")
                        .with_columns(level1_answer_time=pl.col('level_2_first') - pl.col('level_1_last'))
                        .drop(['level_1_last', 'level_2_first']))

                df = df.join(tmp, on="session_id", how='left')

            if grp == '13-22':
                aggs = dict_agg_feature_by_level['0-4'] + dict_agg_feature_by_level['5-12'] + dict_agg_feature_by_level['13-22']

                tmp = (x.groupby(["session_id"], maintain_order=True)
                        .agg(aggs)
                        .sort("session_id")
                        .with_columns(level1_answer_time=pl.col('level_2_first') - pl.col('level_1_last'))
                        .with_columns(level2_answer_time=pl.col('level_3_first') - pl.col('level_2_last'))
                        .drop(['level_1_last', 'level_2_first', 'level_2_last', 'level_3_first']))

                df = df.join(tmp, on="session_id", how='left')

        return df.to_pandas()

    @staticmethod
    def feature_engineer_pre_group(x, grp, use_extra, feature_suffix):
        """create aggregate features for polars."""
        def except_handling(s):
            try:
                return s.max() - s.min()
            except:
                return 0

        aggs = [
            pl.col("index").count().alias(f"session_number_{feature_suffix}"),
            pl.col("text_fqid").filter(pl.col("text_fqid") == 'text_fqid_None').count().alias(f'null_count'),
            pl.col('fqid').filter(pl.col('fqid').str.starts_with('to')).count().alias('page_change_count'),
            pl.col('index').filter(pl.col('text').str.contains('?', literal=True)).count().alias('question_count'),
            pl.col('index').filter((pl.col("event_name") == 'observation_click') & (pl.col('text_fqid').str.contains('block'))).count().alias('block_count'),
            pl.col('index').filter((pl.col("event_name") == 'person_click') & (pl.col('text_fqid').str.contains('recap'))).count().alias('recap_count'),
            pl.col('index').filter((pl.col("event_name") == 'person_click') & (pl.col('text_fqid').str.contains('lost'))).count().alias('lost_count'),

            *[pl.col(c).drop_nulls().n_unique().alias(f"{c}_unique_{feature_suffix}") for c in LIST_CATS],
            *[pl.col('index').filter(pl.col('text').str.contains(c, literal=True)).count().alias(f'word_{c}') for c in DIALOGS],
            *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c, literal=True))).mean().alias(f'word_mean_{c}') for c in DIALOGS],
            *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c, literal=True))).std().alias(f'word_std_{c}') for c in DIALOGS],
            *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c, literal=True))).max().alias(f'word_max_{c}') for c in DIALOGS],
            *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c, literal=True))).sum().alias(f'word_sum_{c}') for c in DIALOGS],
            *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c, literal=True))).median().alias(f'word_median_{c}') for c in DIALOGS],

            *[pl.col(c).mean().alias(f"{c}_mean_{feature_suffix}") for c in LIST_NUMS],
            *[pl.col(c).median().alias(f"{c}_median_{feature_suffix}") for c in LIST_NUMS],
            *[pl.col(c).std().alias(f"{c}_std_{feature_suffix}") for c in LIST_NUMS],
            *[pl.col(c).min().alias(f"{c}_min_{feature_suffix}") for c in LIST_NUMS],
            *[pl.col(c).max().alias(f"{c}_max_{feature_suffix}") for c in LIST_NUMS],
            *[pl.col(c).sum().alias(f"{c}_sum_{feature_suffix}") for c in LIST_NUMS],

            *[pl.col("event_name").filter(pl.col("event_name") == c).count().alias(f"{c}_event_name_counts{feature_suffix}")for c in LIST_EVENT_NAME_FEATURE],
            *[pl.col("elapsed_time_diff").filter(pl.col("event_name")==c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c in LIST_EVENT_NAME_FEATURE],
            *[pl.col("elapsed_time_diff").filter(pl.col("event_name")==c).median().alias(f"{c}_ET_median_{feature_suffix}") for c in LIST_EVENT_NAME_FEATURE],
            *[pl.col("elapsed_time_diff").filter(pl.col("event_name")==c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in LIST_EVENT_NAME_FEATURE],
            *[pl.col("elapsed_time_diff").filter(pl.col("event_name")==c).max().alias(f"{c}_ET_max_{feature_suffix}") for c in LIST_EVENT_NAME_FEATURE],

            *[pl.col("name").filter(pl.col("name") == c).count().alias(f"{c}_name_counts{feature_suffix}")for c in LIST_NAME_FEATURE],   
            *[pl.col("elapsed_time_diff").filter(pl.col("name")==c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c in LIST_NAME_FEATURE],
            *[pl.col("elapsed_time_diff").filter(pl.col("name")==c).median().alias(f"{c}_ET_median_{feature_suffix}") for c in LIST_NAME_FEATURE],
            *[pl.col("elapsed_time_diff").filter(pl.col("name")==c).max().alias(f"{c}_ET_max_{feature_suffix}") for c in LIST_NAME_FEATURE],
            *[pl.col("elapsed_time_diff").filter(pl.col("name")==c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in LIST_NAME_FEATURE],
            *[pl.col("elapsed_time_diff").filter(pl.col("name")==c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for c in LIST_NAME_FEATURE],

            *[pl.col("room_fqid").filter(pl.col("room_fqid") == c).count().alias(f"{c}_room_fqid_counts{feature_suffix}")for c in LIST_ROOM],
            *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in LIST_ROOM],
            *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c in LIST_ROOM],
            *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).median().alias(f"{c}_ET_median_{feature_suffix}") for c in LIST_ROOM],
            *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for c in LIST_ROOM],
            *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for c in LIST_ROOM],

            *[pl.col("fqid").filter(pl.col("fqid") == c).count().alias(f"{c}_fqid_counts{feature_suffix}")for c in LIST_FQID],
            *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in LIST_FQID],
            *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c in LIST_FQID],
            *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).median().alias(f"{c}_ET_median_{feature_suffix}") for c in LIST_FQID],
            *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for c in LIST_FQID],
            *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for c in LIST_FQID],

            *[pl.col("text_fqid").filter(pl.col("text_fqid") == c).count().alias(f"{c}_text_fqid_counts{feature_suffix}") for c in LIST_TEXT],
            *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in LIST_TEXT],
            *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c in LIST_TEXT],
            *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).median().alias(f"{c}_ET_median_{feature_suffix}") for c in LIST_TEXT],
            *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for c in LIST_TEXT],
            *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for c in LIST_TEXT],

            *[pl.col("location_x_diff").filter(pl.col("event_name") == c).mean().alias(f"{c}_ET_mean_x{feature_suffix}") for c in LIST_EVENT_NAME_FEATURE],
            *[pl.col("location_x_diff").filter(pl.col("event_name") == c).median().alias(f"{c}_ET_median_x{feature_suffix}") for c in LIST_EVENT_NAME_FEATURE],
            *[pl.col("location_x_diff").filter(pl.col("event_name") == c).std().alias(f"{c}_ET_std_x{feature_suffix}") for c in LIST_EVENT_NAME_FEATURE],
            *[pl.col("location_x_diff").filter(pl.col("event_name") == c).max().alias(f"{c}_ET_max_x{feature_suffix}") for c in LIST_EVENT_NAME_FEATURE],
            *[pl.col("location_x_diff").filter(pl.col("event_name") == c).min().alias(f"{c}_ET_min_x{feature_suffix}") for c in LIST_EVENT_NAME_FEATURE],

            *[pl.col("level").filter(pl.col("level") == c).count().alias(f"{c}_LEVEL_count{feature_suffix}") for c in LEVELS],
            *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in LEVELS],
            *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c in LEVELS],
            *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for c in LEVELS],
            *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).median().alias(f"{c}_ET_median_{feature_suffix}") for c in LEVELS],
            *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for c in LEVELS],
            ]

        df = x.groupby(["session_id"], maintain_order=True).agg(aggs).sort("session_id")

        dict_agg_feature_by_level = {
            '0-4':[
                pl.col("elapsed_time").filter(((pl.col('fqid') == 'tunic') & (pl.col('event_name') == 'navigate_click')) | (pl.col('text_fqid') == 'tunic.historicalsociety.collection.tunic.slip')).apply(except_handling).alias("slip_click_duration"),
                pl.col("index").filter(((pl.col('fqid') == 'tunic') & (pl.col('event_name') == 'navigate_click')) | (pl.col('text_fqid') == 'tunic.historicalsociety.collection.tunic.slip')).apply(except_handling).alias("slip_click_indexCount"),
                pl.col("elapsed_time").filter(((pl.col('fqid') == 'plaque') & (pl.col('event_name') == 'navigate_click')) | (pl.col('text_fqid') == 'tunic.kohlcenter.halloffame.plaque.face.date')).apply(except_handling).alias("shirt_era_search_duration"),
                pl.col("index").filter(((pl.col('fqid') == 'plaque') & (pl.col('event_name') == 'navigate_click')) | (pl.col('text_fqid') == 'tunic.kohlcenter.halloffame.plaque.face.date')).apply(except_handling).alias("shirt_era_search_indexCount"),
            ],
            '5-12':[
                pl.col("elapsed_time").filter((pl.col("text")=="Here's the log book.")|(pl.col("fqid")=='logbook.page.bingo')).apply(lambda s: s.max()-s.min()).alias("logbook_bingo_duration"),
                pl.col("index").filter((pl.col("text")=="Here's the log book.")|(pl.col("fqid")=='logbook.page.bingo')).apply(lambda s: s.max()-s.min()).alias("logbook_bingo_indexCount"),
                pl.col("elapsed_time").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='reader'))|(pl.col("fqid")=="reader.paper2.bingo")).apply(lambda s: s.max()-s.min()).alias("reader_bingo_duration"),
                pl.col("index").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='reader'))|(pl.col("fqid")=="reader.paper2.bingo")).apply(lambda s: s.max()-s.min()).alias("reader_bingo_indexCount"),
                pl.col("elapsed_time").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='journals'))|(pl.col("fqid")=="journals.pic_2.bingo")).apply(lambda s: s.max()-s.min()).alias("journals_bingo_duration"),
                pl.col("index").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='journals'))|(pl.col("fqid")=="journals.pic_2.bingo")).apply(lambda s: s.max()-s.min()).alias("journals_bingo_indexCount"),
                pl.col("elapsed_time").filter(((pl.col('fqid') == 'businesscards') & (pl.col('event_name') == 'navigate_click')) | (pl.col('text_fqid') == 'tunic.humanecology.frontdesk.businesscards.card_bingo.bingo')).apply(except_handling).alias("businesscard_bingo_duration"),
                pl.col("index").filter(((pl.col('fqid') == 'businesscards') & (pl.col('event_name') == 'navigate_click')) | (pl.col('text_fqid') == 'tunic.humanecology.frontdesk.businesscards.card_bingo.bingo')).apply(except_handling).alias("businesscard_bingo_indexCount"),

                # add 20230423
                pl.col("elapsed_time").filter(((pl.col('text_fqid') == "tunic.historicalsociety.frontdesk.archivist.need_glass_0")) | (pl.col('text_fqid') == "tunic.historicalsociety.frontdesk.magnify")).apply(except_handling).alias("search_grass_duration"),
                pl.col("index").filter(((pl.col('text_fqid') == "tunic.historicalsociety.frontdesk.archivist.need_glass_0")) | (pl.col('text_fqid') == "tunic.historicalsociety.frontdesk.magnify")).apply(except_handling).alias("search_grass_indexCount"),
            ],
            '13-22':[
                pl.col("elapsed_time").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='reader_flag'))|(pl.col("fqid")=="tunic.library.microfiche.reader_flag.paper2.bingo")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("reader_flag_duration"),
                pl.col("index").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='reader_flag'))|(pl.col("fqid")=="tunic.library.microfiche.reader_flag.paper2.bingo")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("reader_flag_indexCount"),
                pl.col("elapsed_time").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='journals_flag'))|(pl.col("fqid")=="journals_flag.pic_0.bingo")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("journalsFlag_bingo_duration"),
                pl.col("index").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='journals_flag'))|(pl.col("fqid")=="journals_flag.pic_0.bingo")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("journalsFlag_bingo_indexCount"),
                pl.col("elapsed_time").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='tracks'))|(pl.col("text")=="That hoofprint doesn't match the flag!")).apply(except_handling).alias("tracks_duration"),
                pl.col("index").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='tracks'))|(pl.col("text")=="That hoofprint doesn't match the flag!")).apply(except_handling).alias("tracks_indexCount"),

                # add 20230423
                pl.col("elapsed_time").filter(((pl.col('event_name') == "person_click") & (pl.col('text_fqid') == "tunic.historicalsociety.cage.teddy.trapped")) | ((pl.col('event_name') == "navigate_click") & (pl.col('fqid') == "unlockdoor"))).apply(except_handling).alias("search_key_duration"),
                pl.col("index").filter(((pl.col('event_name') == "person_click") & (pl.col('text_fqid') == "tunic.historicalsociety.cage.teddy.trapped")) | ((pl.col('event_name') == "navigate_click") & (pl.col('fqid') == "unlockdoor"))).apply(except_handling).alias("search_key_indexCount"),
            ]
        }

        if use_extra:
            if grp == '0-4':
                aggs = dict_agg_feature_by_level['0-4']
                tmp = x.groupby(["session_id"], maintain_order=True).agg(aggs).sort("session_id")
                df = df.join(tmp, on="session_id", how='left')

            if grp == '5-12':
                aggs = dict_agg_feature_by_level['5-12']
                tmp = x.groupby(["session_id"], maintain_order=True).agg(aggs).sort("session_id")
                df = df.join(tmp, on="session_id", how='left')

            if grp == '13-22':
                aggs = dict_agg_feature_by_level['13-22']
                tmp = x.groupby(["session_id"], maintain_order=True).agg(aggs).sort("session_id")
                df = df.join(tmp, on="session_id", how='left')

        return df.to_pandas()

    @staticmethod
    def time_feature(df):
        df["year"] = df["session_id"].apply(lambda x: int(str(x)[:2])).astype(np.uint8)
        df["month"] = df["session_id"].apply(lambda x: int(str(x)[2:4])+1).astype(np.uint8)
        df["day"] = df["session_id"].apply(lambda x: int(str(x)[4:6])).astype(np.uint8)
        df["hour"] = df["session_id"].apply(lambda x: int(str(x)[6:8])).astype(np.uint8)
        df["minute"] = df["session_id"].apply(lambda x: int(str(x)[8:10])).astype(np.uint8)
        df["second"] = df["session_id"].apply(lambda x: int(str(x)[10:12])).astype(np.uint8)
        return df

    @staticmethod
    def create_target_df(path_target_df):
        """create target dataframe"""
        targets = pd.read_csv(path_target_df)
        targets['session'] = targets.session_id.apply(lambda x: int(x.split('_')[0]))
        targets['q'] = targets.session_id.apply(lambda x: int(x.split('_')[-1][1:]))
        targets.set_index('session', inplace=True)
        targets.sort_index(inplace=True)
        targets = targets[['q', 'correct']]
        return targets

    @staticmethod
    def split_dataframe_by_level_group(df):
        """split data by level_group"""
        df1 = df.filter(pl.col("level_group") == '0-4')
        df2 = df.filter(pl.col("level_group") == '5-12')
        df3 = df.filter(pl.col("level_group") == '13-22')
        return df1, df2, df3

    @staticmethod
    def feature_selection(df, drop_threshold=0.9):
        """create use feature list"""
        nulls = df.isnull().sum().sort_values(ascending=False) / len(df)
        drops = list(nulls[nulls > drop_threshold].index)

        for col in df.columns:
            if df[col].nunique() == 1:
                drops.append(col)

        if df.index.name != 'session_id':
            df.set_index('session_id', inplace=True)
        list_use_feature = [c for c in df.columns if c not in drops+['level_group']]

        return list_use_feature

    @staticmethod
    def dump_feature_list_to_json(list_key, list_value, output_path):
        """dump json file that use feature list by level group"""
        dict_dump = {k: v for k, v in zip(list_key, list_value)}

        with open(output_path, mode='w') as fp:
            json.dump(dict_dump, fp)

    @staticmethod
    def create_compare_data(df, df_target, all_users):
        """create compare score data for search thresold"""
        df_true = df.copy()
        for k in range(1, 19):
            tmp = df_target.loc[df_target.q == k].set_index('session').loc[all_users]
            df_true[f'q_{k}'] = tmp.correct.values
        return df_true

    @staticmethod
    def search_best_threshold(df_preds, df_ture, start=0.5, end=0.7, step=0.005):
        """search best threshol f1score"""
        scores = []
        thresholds = []
        best_score = 0
        for threshold in np.arange(start, end, step):
            preds = (df_preds.values.reshape((-1)) > threshold).astype('int')
            m = f1_score(df_ture.values.reshape((-1)), preds, average='macro')
            scores.append(m)
            thresholds.append(threshold)
            if m > best_score:
                best_score = m
                best_threshold = threshold
            print(f'threshold: {threshold:.3f}, score: {m:.4f}')
        print(f'best threshold: {best_threshold:.4f}')
