import polars as pl

from baseball import pipe_rate_error

from uncertainties import ufloat
import json


def get_stat_map():
    with open('stat_map.json', 'r') as f:
        return json.load(f)


def ufloat_div(x):
    new_ufloat_obj = ufloat(x['ratio_true'], x['ratio_true_error']) / ufloat(
        x['ratio_false'], x['ratio_false_error']
    )
    return {'phase_ratio_value': new_ufloat_obj.n, 'phase_ratio_value_error': new_ufloat_obj.s}


def prop_error(_df):
    return (
        _df
        .filter(pl.col('ratio_false').gt(0))
        .with_columns(
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
        )
        .unnest('mystruct')
    )


def do_phacking(merged, gid_stat_sums, phase=100):
    stat_map = get_stat_map()
    data = merged.lazy()
    if phase == 100:
        data = data.with_columns(theflag=pl.col('full_moon_flag'))
    elif phase == 0:
        data = data.with_columns(theflag=pl.col('new_moon_flag'))
    else:
        data = data.with_columns(theflag=pl.col('phase').is_between(phase - 0.5, phase + 0.5))
    phack_data = (
        data
        .lazy()
        .join(gid_stat_sums.lazy(), on='gid')
        .group_by('season', 'theflag')
        .agg(pl.col(cols).sum())
        .collect()
        .sort('season')
        .unpivot(index=['season', 'theflag', 'b_pa'], variable_name='stat')
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
