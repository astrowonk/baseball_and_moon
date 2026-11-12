"""
gid_stat_sums.parquet  - stat sums by gid

"""

import json

import polars as pl
from baseball import pipe_rate_error
from sqlalchemy import create_engine, text
from tqdm.autonotebook import tqdm
from uncertainties import ufloat

STAT_MAP = {
    'b_pa': "Plate Appearances by Team's Batters",
    'b_ab': "At Bats by Team's Batters",
    'b_r': 'Runs Scored',
    'b_h': 'Hits',
    'b_d': 'Doubles',
    'b_t': 'Triples',
    'b_hr': 'Home Runs',
    'b_rbi': 'Runs Batted In',
    'b_sh': 'Sacrifice Hits',
    'b_sf': 'Sacrifice Flies',
    'b_hbp': "Times Hit by Pitch for Team's Batters",
    'b_w': 'Batter Walks by Team',
    'b_iw': 'Intentional Walks',
    'b_k': 'Batter Strikeouts by Team',
    'b_sb': 'Stolen Bases',
    'b_cs': 'Times Caught Stealing',
    'b_gdp': "Times Grounded in Double Play by Team's Batters",
    'b_xi': "Times Reached Base on Catcher's Interference",
    'b_roe': 'Times Reached Base via Error',
    'p_ipouts': 'Outs Recorded by Team Pitchers (IP * 3)',
    'p_noout': "Batters Faced by Team's Pitchers in Any Innings in Which No Outs Were Recorded",
    'p_bfp': "Batters Faced by Team's Pitchers",
    'p_h': 'Hits Allowed',
    'p_d': 'Doubles Allowed',
    'p_t': 'Triples Allowed',
    'p_hr': 'Home Runs Allowed',
    'p_r': 'Runs Allowed',
    'p_er': 'Earned Runs Allowed',
    'p_w': 'Walks Allowed',
    'p_iw': 'Intentional Walks Allowed',
    'p_k': "Strikeouts by Team's Pitchers",
    'p_hbp': 'Hit by Pitches Allowed',
    'p_wp': 'Wild Pitches Allowed',
    'p_bk': 'Balks Allowed',
    'p_sh': 'Sacrifice Hits Allowed',
    'p_sf': 'Sacrifice Flies Allowed',
    'p_sb': 'Stolen Bases Allowed',
    'p_cs': 'Caught Stealing Allowed',
    'p_pb': 'Passed Balls Allowed',
    'd_po': 'Team Putouts',
    'd_a': 'Fielding Assists by Team',
    'd_e': 'Errors Committed by Team',
    'd_dp': 'Double Plays Turned by Team',
    'd_tp': 'Triple Plays Turned by Team',
    'd_pb': 'Passed Balls Allowed by Team',
    'd_wp': 'Wild Pitches Allowed by Team',
    'd_sb': 'Stolen Bases Allowed by Team',
    'd_cs': 'Caught Stealing Allowed by Team',
}

COLUMNS = [
    'b_t',
    'b_hr',
    'b_pa',
    'b_d',
    'd_dp',
    'p_k',
    'b_hbp',
    #    'b_cs',
    'b_sh',
    #   'b_sb',
    'b_gdp',
    'b_roe',
    'b_iw',
    'b_w',
    'p_wp',
]


def make_gid_lookup_parquet():
    pl.read_csv('teamstats.csv', infer_schema_length=100000, ignore_errors=True).filter(
        pl.col('stattype').eq('value')
    ).group_by('gid').agg(pl.col(COLUMNS).sum()).write_parquet('gid_stat_sums.parquet')


def get_stat_map():
    with open('stat_map.json', 'r') as f:
        return json.load(f)


def ufloat_div(x):
    if any(value is None for value in x.values()) or x['ratio_false'] == 0:
        return {'phase_ratio_value': None, 'phase_ratio_value_error': None}

    result = ufloat(x['ratio_true'], x['ratio_true_error']) / ufloat(
        x['ratio_false'], x['ratio_false_error']
    )
    return {'phase_ratio_value': result.n, 'phase_ratio_value_error': result.s}


def prop_error(_df):
    return _df.with_columns(
        mystruct=pl.struct(
            'ratio_true', 'ratio_true_error', 'ratio_false', 'ratio_false_error'
        ).map_elements(
            ufloat_div,
            return_dtype=pl.Struct([
                pl.Field('phase_ratio_value', pl.Float64),
                pl.Field('phase_ratio_value_error', pl.Float64),
            ]),
            skip_nulls=True,
        )
    ).unnest('mystruct')


def make_data_before_pivot(merged, gid_stat_sums, phase=100, phase_width=1) -> pl.LazyFrame:
    data = merged.lazy()
    if phase == 100:
        data = data.with_columns(theflag=pl.col('full_moon_flag'))
    elif phase == 0:
        data = data.with_columns(theflag=pl.col('new_moon_flag'))
    else:
        data = data.with_columns(
            theflag=pl.col('phase').is_between(
                phase - phase_width / 2, phase + phase_width / 2
            )
        )
    return (
        data
        .lazy()
        .join(gid_stat_sums.lazy(), on='gid')
        .group_by('season', 'theflag')
        .agg(pl.col(COLUMNS).sum())
        .unpivot(index=['season', 'theflag', 'b_pa'], variable_name='stat')
    )


def do_phacking(merged, gid_stat_sums, phase=100, phase_width=1):
    stat_map = STAT_MAP

    phack_data = (
        make_data_before_pivot(merged, gid_stat_sums, phase, phase_width=phase_width)
        .collect()
        .pivot(on=['theflag'], index=['season', 'stat'], values=['b_pa', 'value'])
        .with_columns(
            ratio_true=pl.col('value_true') / 'b_pa_true',
            ratio_false=pl.col('value_false') / 'b_pa_false',
        )
    ).drop_nulls()
    hacked_with_error = (
        phack_data
        .pipe(pipe_rate_error, 'ratio_true', col_count='b_pa_true')
        .pipe(pipe_rate_error, 'ratio_false', col_count='b_pa_false')
        .pipe(prop_error)
        .with_columns(pct_change=pl.col('phase_ratio_value') - 1)
    )
    return hacked_with_error.with_columns(
        phacked=(pl.col('pct_change').abs() >= pl.col('phase_ratio_value_error'))
        & pl.col('value_true').gt(0),
        stat_name=pl.col('stat').replace_strict(stat_map),
        phase_bin=phase,
    )


def make_gid_stat_sums_parquet():

    cols = [
        'b_t',
        'b_hr',
        'b_pa',
        'b_d',
        'd_dp',
        'p_k',
        'b_hbp',
        #    'b_cs',
        'b_sh',
        #   'b_sb',
        'b_gdp',
        'b_roe',
        'b_iw',
        'b_w',
        'p_wp',
    ]
    pl.read_csv('teamstats.csv', infer_schema_length=100000, ignore_errors=True).filter(
        pl.col('stattype').eq('value')
    ).group_by('gid').agg(pl.col(cols).sum()).write_parquet('gid_stat_sums.parquet')


class DatabaseMaker:
    all_stats = None

    def __init__(self):
        self.merged = pl.read_parquet('gameinfo1900_2025_with_moon.parquet')
        self.gid_stat_sums = pl.read_parquet('gid_stat_sums.parquet')

    def make_all_stats(self, phase_width=2):
        out = []
        for phase in tqdm([x for x in list(range(0, 101, phase_width))]):
            out.append(
                do_phacking(
                    self.merged, self.gid_stat_sums, phase=phase, phase_width=phase_width
                )
            )
        self.all_stats = pl.concat(out, how='diagonal_relaxed')

    def upload_database(self):
        assert self.all_stats is not None, 'Run make_all_stats first'
        con = create_engine('sqlite:///phack.db')
        print('shuffling')
        self.all_stats.sample(fraction=1, shuffle=True).with_columns(
            posted=pl.lit(None).cast(pl.Date)
        ).write_database('p_hack_results', connection=con, if_table_exists='replace')
        with con.connect() as connection:
            connection.execute(
                text("""CREATE INDEX IF NOT EXISTS datephack on p_hack_results(posted);""")
            )

    def upload_unpivotted(self, phase_width=2):
        unpivotted = pl.concat([
            make_data_before_pivot(
                self.merged, self.gid_stat_sums, phase=phase, phase_width=phase_width
            )
            .with_columns(phase_bin=pl.lit(phase))
            .collect()
            for phase in range(0, 101, 2)
        ])
        con = create_engine('sqlite:///phack.db')

        unpivotted.write_database('unpivotted', connection=con, if_table_exists='replace')
        with con.connect() as connection:
            connection.execute(
                text(
                    """CREATE INDEX if not exists up1 on unpivotted(season,phase_bin,stat);"""
                )
            )
