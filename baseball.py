import polars as pl
from uncertainties import ufloat
import pandas as pd
import math
from tqdm.autonotebook import tqdm
import random


def rate_error(col, col_count):
    return 1.96 * (pl.col(col) * (1 - pl.col(col)) / pl.col(col_count)).sqrt()


def pipe_rate_error(_df: pl.LazyFrame | pl.DataFrame, col, col_count=None, out_name=None):
    if not out_name:
        out_name = f'{col}_error'
    if not col_count:
        col_count = f'{col}_count'
    if col_count not in _df.columns:
        _df = _df.with_columns(pl.col(col).count().alias(col_count))
    return _df.with_columns(rate_error(col, col_count).alias(out_name))


def make_fake_phase(_df: pl.LazyFrame, period: float = 26.7) -> pl.LazyFrame:
    return _df.with_columns(
        days_since=(pl.col('date') - pl.col('date').min()).dt.total_days()
    ).with_columns(phase=100 * (0.5 + 0.5 * (math.pi * pl.col('days_since') / period).cos()))


def make_one_df_range(
    _merged,
    threshold=5,
    phase=99,
    range_size=1,
    fake_moon_period=None,
    group_by=['Moon Status'],
    stat='large_victory_flag',
):
    merged = _merged.lazy()
    if fake_moon_period:
        merged = merged.pipe(make_fake_phase, fake_moon_period)
    return (
        merged
        .with_columns(
            (pl.col('phase').is_between(phase, phase + range_size)).alias('Moon Status'),
            large_victory_flag=pl.col('margin_of_victory').ge(threshold),
            #  team_quality_large_gap = (pl.col('py_win')-pl.col('py_win_home')).abs().gt(.10)
        )
        .group_by(group_by, maintain_order=True)
        .agg(
            pl.col(stat).mean().alias(f'{stat}_rate'),
            pl.col('phase').mean(),
            pl.col('gid').count().alias('count'),
            pl.col(stat).sum().alias(f'{stat}_count'),
            pl.col('daynight').eq('day').mean().alias('day_game_flag'),
        )
        .with_columns(blowout_threshold=threshold)
        .collect()
        .pipe(pipe_rate_error, f'{stat}_rate', 'count')
        .with_columns(phase_bin=phase, fake_moon_period=fake_moon_period)
    )


def make_phase_bin_data(merged, threshold=7, fake_moon_period=None, phase_range=range(100)):

    phase_vs_all = (
        (
            pl.concat([
                make_one_df_range(
                    merged, threshold=threshold, phase=phase, fake_moon_period=fake_moon_period
                )
                for phase in phase_range
            ]).pivot(
                on='Moon Status', index='phase_bin', values=['large_victory_rate', 'count']
            )
        )
        .pipe(pipe_rate_error, 'large_victory_rate_true', 'count_true')
        .pipe(pipe_rate_error, 'large_victory_rate_false', 'count_false')
    )
    phase_data_pandas = phase_vs_all.to_pandas()
    phase_data_pandas['large_victory_rate_in_phase_range'] = phase_data_pandas[
        ['large_victory_rate_true', 'large_victory_rate_true_error']
    ].apply(lambda x: ufloat(x.iloc[0], x.iloc[1]), axis=1)
    phase_data_pandas['large_victory_rate_out_of_phase_range'] = phase_data_pandas[
        ['large_victory_rate_false', 'large_victory_rate_false_error']
    ].apply(lambda x: ufloat(x.iloc[0], x.iloc[1]), axis=1)
    phase_data_pandas = phase_data_pandas.assign(
        ratio_ufloat=pd.col('large_victory_rate_in_phase_range')
        / pd.col('large_victory_rate_out_of_phase_range')
    ).assign(
        ratio=pd.col('ratio_ufloat').apply(lambda x: x.n),
        ratio_error=pd.col('ratio_ufloat').apply(lambda x: x.s),
        fake_moon_period=fake_moon_period,
    )
    return phase_data_pandas


def get_all_baseballdata(start_year=1900):
    return (
        pl
        .scan_csv(
            '/Volumes/BMD/Large Data Sets/baseball_data/gameinfo.csv',
            infer_schema_length=100000,
        )
        .with_columns(pl.col('date').cast(pl.String).str.to_date('%Y%m%d'))
        .filter(pl.col('date').gt(pl.date(start_year, 1, 1)))
        .with_columns(margin_of_victory=(pl.col('vruns') - pl.col('hruns')).abs())
    ).collect()


def make_moon(file='horizons_results.txt'):
    pandas_df = pd.read_table(file, sep=r'\s{4,}', parse_dates=True, engine='python')
    pandas_df.columns = ['date', 'phase']
    pandas_df['date'] = pd.to_datetime(pandas_df['date'])
    return pl.from_pandas(pandas_df).with_columns(pl.col('date').dt.date())


def get_merged_data():
    all_data = get_all_baseballdata()
    moon_data = make_moon()
    return all_data.join(moon_data, how='left', on=['date'])


def make_fake_vs_real_full_moons(merged, n=100, threshold=7, group=['Moon Status']):
    out = []
    for _ in tqdm(range(n)):
        random_period = random.randint(250000, 350000) / 10000
        out.append(
            make_one_df_range(
                merged, fake_moon_period=random_period, threshold=threshold, group_by=group
            )
        )
    out.append(
        make_one_df_range(
            merged,
            group_by=group,
            fake_moon_period=None,
            threshold=7,
        )
    )
    print(out[-1].columns)
    return (
        pl
        .concat(out)
        .pivot(
            on='Moon Status',
            index=['phase_bin', 'fake_moon_period']
            + (['season'] if 'season' in group else []),
            values=['large_victory_rate'],
            aggregate_function='first',
        )
        .with_columns(full_moon_ratio=pl.col('true') / 'false')
        .with_columns(pl.col('fake_moon_period').is_not_null().alias('fake moon'))
    )
