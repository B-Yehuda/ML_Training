--- Part 1: General information of each community ---
with communities as (
    select community
         , count(*) as community_size
    from dev_yehuda.channel_community
    group by community
)

   , se_channels as (
    select channel_community.community
         , count(*) as se_channels
    from dev_yehuda.channel_community
             join bi_db.creators
                  on channel_community.channel = creators.provider_id and
                     creators.provider = 'twitch' and
                     creators.boss_channel_id is not null
    group by channel_community.community
)

   , se_activity as (
    select channel_community.community
         , count(invite_timestamp) as invites
         , sum(is_accept)          as accept
         , sum(is_started)         as start
         , sum(is_finished)        as won
    from dev_yehuda.channel_community
             join bi_db.se_invitations
                  on channel_community.channel = se_invitations.providerid
    where se_invitations.invite_timestamp is not null
      and se_invitations.invite_timestamp >= '2022-01-01'
      and se_invitations.provider = 'twitch'
      and se_invitations.se_platform in ('sesp', 'boss')
    group by channel_community.community
)

   , deployments as (
    select channel_community.community
         , count(*) as deployments
    from dev_yehuda.channel_community
             join bi_db.deployments
                  on channel_community.channel = deployments.provider_id
    where deployments.start_date is not null
      and deployments.start_date >= '2022-01-01'
      and deployments.provider = 'twitch'
      and deployments.deployment_source_system in ('sesp', 'boss')
      and deployments.status = 'won'
    group by channel_community.community
)

--- Part 2: Features information of each channel ---
   , feature_raid as (
    select latest_deployment.channel_id
         , 'raid'               as deployment_type
         , case
               when latest_deployment.tutorials_d7 > 0 and latest_deployment.deposits_count_d7 > 0
                   then 'tier_1'
               when latest_deployment.tutorials_d7 > 0 and latest_deployment.deposits_count_d7 = 0
                   then 'tier_2'
               when latest_deployment.tutorials_d7 = 0 and latest_deployment.deposits_count_d7 = 0
                   then 'tier_3'
               else 'other' end as feature
    from (
             select channel_id
                  , tutorials_d7
                  , deposits_count_d7
                  , row_number()
                    over (partition by channel_id order by deployment_start_date desc) as rnk
             from bi_db.kronos_model_training
             where deployment_start_date >= '2022-01-01'
         ) as latest_deployment
    where rnk = 1
)

   , feature_raid_1k_plus_payout as (
    select latest_deployment.channel_id
         , 'raid_1k_plus_payout' as deployment_type
         , case
               when latest_deployment.tutorials_d7 > 0 and latest_deployment.deposits_count_d7 > 0
                   then 'tier_1'
               when latest_deployment.tutorials_d7 > 0 and latest_deployment.deposits_count_d7 = 0
                   then 'tier_2'
               when latest_deployment.tutorials_d7 = 0 and latest_deployment.deposits_count_d7 = 0
                   then 'tier_3'
               else 'other' end  as feature
    from (
             select plarium_metrics.old_channel_id                                                                     as channel_id
                  , plarium_metrics.tutorials_d7
                  , plarium_metrics.deposits_count_d7
                  , row_number()
                    over (partition by plarium_metrics.channel_id order by plarium_metrics.deployment_start_date desc) as rnk
             from bi_db.plarium_metrics
                      join bi_db.boss_deployments
                           on plarium_metrics.deployment_id = boss_deployments.deployment_id
                               and boss_deployments.deployment_state in
                                   ('completed', 'payment_processing', 'adjustment_approval')
                               and boss_deployments.channel_platform = 'twitch'
             where boss_deployments.deployment_start_date >= '2022-01-01'
               and boss_deployments.deployment_total_payout_amount > 1000
               and lower(plarium_metrics.advertiser_name) = 'raid: shadow legends'
         ) as latest_deployment
    where rnk = 1
)

   , feature_hellofresh as (
    select latest_deployment.channel_id
         , 'hellofresh' as deployment_type
         , case
               when latest_deployment.conversions > 5
                   then 'tier_1' -- 'hellofresh_5_plus_conversions'
               when latest_deployment.conversions > 0 and latest_deployment.conversions <= 5
                   then 'tier_2' -- 'hellofresh_1_to_5_conversions'
               else 'tier_3' -- 'hellofresh_0_conversions'
        end             as feature
    from (
             select deployments.channel_id
                  , isnull(sum(v_hellofresh_events.conversions), 0)                                 as conversions
                  , row_number()
                    over (partition by deployments.channel_id order by deployments.start_date desc) as rnk
             from (
                      select channel_id
                           , start_date
                           , end_date
                      from bi_db.deployments
                      where start_date is not null
                        and start_date >= '2022-01-01'
                        and provider = 'twitch'
                        and deployment_source_system in ('sesp', 'boss')
                        and status = 'won'
                        and lower(advertiser_name) like '%hellofresh%'
                  ) as deployments
                      left join bi_db.v_hellofresh_events
                                on deployments.channel_id = v_hellofresh_events.channel_id and
                                   v_hellofresh_events.date between deployments.start_date and deployments.end_date
             group by deployments.channel_id
                    , deployments.start_date
         ) as latest_deployment
    where rnk = 1
)

   , feature_factor as (
    select latest_deployment.channel_id
         , 'factor' as deployment_type
         , case
               when latest_deployment.conversions > 5
                   then 'tier_1' -- 'factor_5_plus_conversions'
               when latest_deployment.conversions > 0 and latest_deployment.conversions <= 5
                   then 'tier_2' -- 'factor_1_to_5_conversions'
               else 'tier_3' -- 'factor_0_conversions'
        end         as feature
    from (
             select deployments.channel_id
                  , isnull(sum(v_hellofresh_events.conversions), 0)                                 as conversions
                  , row_number()
                    over (partition by deployments.channel_id order by deployments.start_date desc) as rnk
             from (
                      select channel_id
                           , start_date
                           , end_date
                      from bi_db.deployments
                      where start_date is not null
                        and start_date >= '2022-01-01'
                        and provider = 'twitch'
                        and deployment_source_system in ('sesp', 'boss')
                        and status = 'won'
                        and lower(advertiser_name) like '%factor%'
                  ) as deployments
                      left join bi_db.v_hellofresh_events
                                on deployments.channel_id = v_hellofresh_events.channel_id and
                                   v_hellofresh_events.date between deployments.start_date and deployments.end_date
             group by deployments.channel_id
                    , deployments.start_date
         ) as latest_deployment
    where rnk = 1
)

--- Part 3: Aggregate features data of channels and communities ---
   , channel_community_feature as (
    select channel_community.channel
         , channel_community.community
         , features.deployment_type
         , features.feature
    from dev_yehuda.channel_community
             join bi_db.creators
                  on channel_community.channel = creators.provider_id and
                     creators.provider = 'twitch' and
                     creators.boss_channel_id is not null
             join (select channel_id, deployment_type, feature
                   from feature_raid
                   union
                   select channel_id, deployment_type, feature
                   from feature_raid_1k_plus_payout
                   union
                   select channel_id, deployment_type, feature
                   from feature_hellofresh
                   union
                   select channel_id, deployment_type, feature
                   from feature_factor) as features
                  on creators.channel_id = features.channel_id
)

--    , agg_community_feature as (
--     select community
--          , deployment_type
--          , feature
--          , count(*) as feature_size
--     from channel_community_feature
--     group by community
--            , deployment_type
--            , feature
-- )
--
-- --- Part 4: Aggregate all data ---
-- select communities.community
--      , communities.community_size
--      , se_channels.se_channels
--      , se_activity.invites
--      , se_activity.accept
--      , 1.00 * se_activity.accept / se_activity.invites as acceptance_rate
--      , se_activity.start
--      , 1.00 * se_activity.start / se_activity.invites  as start_rate
--      , se_activity.won
--      , 1.00 * se_activity.won / se_activity.invites    as won_rate
--      , deployments.deployments
--      , agg_community_feature.deployment_type
--      , agg_community_feature.feature
--      , agg_community_feature.feature_size
-- from communities
--          left join se_channels
--                    on communities.community = se_channels.community
--          left join se_activity
--                    on communities.community = se_activity.community
--          left join deployments
--                    on communities.community = deployments.community
--          left join agg_community_feature
--                    on communities.community = agg_community_feature.community
-- where agg_community_feature.feature is not null

select *
from channel_community_feature
where channel_community_feature.feature is not null