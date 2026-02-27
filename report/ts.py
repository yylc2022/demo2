import numpy as np
import pandas as pd
import os
# Try to import Pool from jindefund.dpool or jd.dpool; if missing, fall back to local jd_shim
try:
    from jindefund.dpool import Pool
except Exception:
    try:
        from jd.dpool import Pool
    except Exception:
        from jd_shim import Pool

from functools import reduce
# try import jd-provided helpers; fallback to jd_shim
try:
    from jd import gettab, rmean, troll, savetab, rstd
    from jd import tcorr, tcos, tmean, tcount, tstd, tcumsum, tdrawdown
    from jd import bts
    from jd.dtype import struct as ds
    from jd.pattern.time import convert_jinde_datetime_to_pandas_datetime_list
    from jd.io import log as fl
except Exception:
    # fallback
    from jd_shim import gettab, rmean, troll, savetab, rstd, fl, tcorr, tcos, tmean, tcount, tstd, tcumsum, tdrawdown, bts, ds, convert_jinde_datetime_to_pandas_datetime_list
import io_bokeh as fb

def calc_tcorr(symbol, meta):
    fl.init(os.path.join(meta.log_root, 'map_phase', '{}.log'.format(symbol)))
    fl.info("current symbol is: %s", symbol)
    label_list = meta.label_list
    signal = meta.signal
    mycorr = tcorr if meta.corr_type=='pearson' else tcos
    dtkey = meta.dttm_key
    signal = meta.signal
    last_keep_days = meta.last_keep_days
    
    tab_signal = gettab(
        freq=meta.freq, 
        shard=meta.shard, 
        start=meta.start, end=meta.end, 
        universe=[symbol],
        source='h5/gp/'+meta.signal_root,
        vector=None
    )

    tab_label = gettab(
        freq=meta.freq, 
        shard=meta.shard, 
        start=meta.start, end=meta.end, 
        universe=[symbol],
        source='h5/gp/'+meta.label_root,
        vector=None
    )

    tcorr_list = []

    while(not tab_signal.finish()):
        try:
            # print("Now:", tab_signal.now())
            df_labal_list = []
            
            tab_signal.load()
            tab_label.load()
            df_sig = tab_signal.concat_data[symbol]
            df_lab = tab_label.concat_data[symbol]

            # Calc auto calc
            df = pd.concat([df_sig[dtkey], df_sig[signal]], axis=1)
            df = df.set_index(dtkey)

            # Determine resampling freq
            if getattr(meta, 'freq', 'minute') == 'day':
                # Use Month Start to get enough daily points for correlation
                resample_freq = 'MS'
            else:
                resample_freq = '15T'

            # Helper for resampling
            def resample_calc(mdf, func, cols, freq=resample_freq):
                # mdf has integer index (Jinde format)
                # Convert to datetime index
                dti = convert_jinde_datetime_to_pandas_datetime_list(mdf.index.values)
                temp = mdf.copy()
                temp.index = dti
                # Group/Resample
                res = temp.groupby(pd.Grouper(freq=freq)).apply(func)
                # Convert index back to Jinde int
                if isinstance(res, pd.Series):
                    res = res.to_frame()

                # If apply returns MultiIndex, usually (Time, Columns) or just Time.
                # If we get (Time, ..), we want to keep Time index.

                # Back to integer index
                # YYYYMMDDHHMMSS000
                new_idx_str = res.index.strftime('%Y%m%d%H%M%S')
                # Pad with 000
                new_idx_int = (new_idx_str + '000').astype(np.int64)
                res.index = new_idx_int
                return res

            full_res_list = []
            for j in meta.auto_shift_bars:
                df['shift'] = df[signal].shift(j)

                # Function for autocorr
                def calc_auto(subdf):
                    if len(subdf) < 2: return np.nan
                    return mycorr(subdf[signal].values, subdf['shift'].values)

                label_res = resample_calc(df, calc_auto, [signal, 'shift'])
                label_res.columns = ['tautocorr({},{})'.format(signal, j)]

                # print("label {} finish, res:\n{}".format(label, label_res))
                df_labal_list.append(label_res)
                
            df_sig = tab_signal.load()[symbol]
            df_lab = tab_label.load()[symbol]
            
            # count of 15M count, mean, std of x,y
            for key in [signal, label_list[0]]:
                if(key == signal):
                    df = pd.concat([df_sig[dtkey], df_sig[key]], axis=1)
                else:
                    df = pd.concat([df_lab[dtkey], df_lab[key]], axis=1)

                df = df.set_index(dtkey)
                # print("11111:", label, df)

                # tcount
                def calc_count(subdf):
                    return tcount(subdf[key].values)
                label_res = resample_calc(df, calc_count, [key])
                label_res.columns = ['tcount({})'.format(key)]
                df_labal_list.append(label_res)
                
                # tmean
                def calc_mean(subdf):
                    return tmean(subdf[key].values)
                label_res = resample_calc(df, calc_mean, [key])
                label_res.columns = ['tmean({})'.format(key)]
                df_labal_list.append(label_res)
                
                # tstd
                def calc_std(subdf):
                    return tstd(subdf[key].values)
                label_res = resample_calc(df, calc_std, [key])
                label_res.columns = ['tstd({})'.format(key)]
                df_labal_list.append(label_res)

            for label in label_list:
                df = pd.concat([df_sig[dtkey], df_sig[signal], df_lab[label]], axis=1)
                df = df.set_index(dtkey)
                # print("11111:", label, df)

                def calc_ic(subdf):
                     if len(subdf) < 2: return np.nan
                     return mycorr(subdf[signal].values, subdf[label].values)

                label_res = resample_calc(df, calc_ic, [signal, label])
                label_res.columns = ['mycorr({},{})'.format(signal, label)]

                # print("label {} finish, res:\n{}".format(label, label_res))
                df_labal_list.append(label_res)

                # print(df_labal_list)
            final_df = pd.concat(df_labal_list, axis=1)
            # fl.info("final_df:%s", final_df)
            if(meta.shard=='forever'): # Bug because uppercase
                div = 10**20 # Ensure cut is 0
            elif(meta.shard=='year'): # Bug because uppercase
                div = 10**13

            # Note: tab_signal.now() returns shard index. If shard='year', it is e.g. 2020.
            cut = final_df.index // div
            # fl.info("cuttingggg=%s", cut)
            real_df = final_df[cut==tab_signal.now()]
            # fl.info("real_df:%s", real_df)
            tcorr_list.append(real_df)
        except Exception as e:
            fl.warn("tcorr_calc calc error: %s %s", symbol, tab_signal.now())
        try:
            tab_signal.roll(autoload = False, last_shard_keep_days = last_keep_days, dttm_col=dtkey)
            tab_label.roll(autoload = False, last_shard_keep_days = last_keep_days, dttm_col=dtkey)
        except Exception as e:
            fl.error("tcorr_calc rolling error: %s", symbol)
    # fl.info("tcorr_list type is %s", str(type(tcorr_list)))
    if(meta.shard=='forever'): # Bug because uppercase
        res = tcorr_list[0]
    else:
        res = pd.concat(tcorr_list)
    # fl.info("ts.calc_tcorr.res:%s", res)
    res[dtkey] = res.index.astype(int)
    res = res.reset_index(drop=True)
    fl.info("Try to save %s", symbol)
    savetab(res, os.path.join(meta.save_root, 'map_phase'),datetime_key=dtkey, symbol=symbol, shardby='forever')

def map_phase(meta):
    label = getattr(meta, 'label', None)
    with Pool(processes=meta.nprocess) as pool:
        for sb in meta.unv:
            pool.apply_async(calc_tcorr, (sb, meta,), label=label and f'{label}-{sb}' or None)
        pool.close()
        pool.join()

def minbar_table1(df, signal, default_label):
    statistic_list = ['mycorr({signal},{label})'.format(signal=signal, label=default_label)]
    for key in [signal, default_label]:
        for func in ['tcount', 'tmean', 'tstd']:
            s = "{func}({key})".format(func=func, key=key)
            statistic_list.append(s)
    showtab1 = df[statistic_list].replace([np.inf, -np.inf], np.nan).dropna().describe(percentiles=[0.05,0.95])
    return showtab1

def minbar_table2(df, signal, label_list):
    lblist = []
    for lb in label_list:
        lblist.append('mycorr({signal},{label})'.format(signal=signal, label=lb))
    showtab2 = df[lblist].replace([np.inf, -np.inf], np.nan).dropna().describe(percentiles=[0.05,0.95]).T
    return showtab2

def minbar_table3(tab, signal, default_label, symbol_list, dttm_key='dt'):
    res = {}
    des_columns = []
    des_content = []

    # Helper to find the mycorr column that targets default_label in a DataFrame
    def find_mycorr_col(df, target_label):
        # Prefer exact match of second argument inside mycorr(...,label)
        for c in df.columns:
            if c.startswith('mycorr(') and ',' in c and c.endswith(')'):
                try:
                    second = c.split(',', 1)[1].rstrip(')')
                    if second == target_label:
                        return c
                except Exception:
                    pass
        # Fallback: any column containing both 'mycorr' and the label string
        for c in df.columns:
            if 'mycorr' in c and target_label in c:
                return c
        return None

    # Build whole_market series by averaging per-symbol mycorr(target_label)
    # Collect per-symbol series where available
    loaded = tab.load()
    per_symbol_series = []
    for sym, sdf in loaded.items():
        try:
            col = find_mycorr_col(sdf, default_label)
            if col is None:
                continue
            tmp = sdf[[dttm_key, col]].copy()
            tmp = tmp.dropna(subset=[col]).set_index(dttm_key)[col]
            tmp.name = sym
            per_symbol_series.append(tmp)
        except Exception:
            continue

    if per_symbol_series:
        combined = pd.concat(per_symbol_series, axis=1)
        # align by index (dt) and take mean across symbols
        mean_series = combined.mean(axis=1)
        whole_sbdf = mean_series.reset_index()
        whole_sbdf.columns = [dttm_key, 'mycorr({},{})'.format(signal, default_label)]
    else:
        whole_sbdf = pd.DataFrame(columns=[dttm_key, 'mycorr({},{})'.format(signal, default_label)])

    # Process whole_market first
    for symbol in ['whole_market'] + list(symbol_list):
        try:
            if symbol == 'whole_market':
                sbdf = whole_sbdf.copy()
            else:
                # For specific symbol, pick its DataFrame and find the correct mycorr column
                df_symbol = loaded.get(symbol)
                if df_symbol is None:
                    sbdf = pd.DataFrame(columns=[dttm_key, 'mycorr({},{})'.format(signal, default_label)])
                else:
                    col = find_mycorr_col(df_symbol, default_label)
                    if col is None:
                        # no matching mycorr column; create empty
                        sbdf = pd.DataFrame(columns=[dttm_key, 'mycorr({},{})'.format(signal, default_label)])
                    else:
                        sbdf = df_symbol[[dttm_key, col]].copy()
                        # rename to consistent key name so later code can use same key
                        sbdf = sbdf.rename(columns={col: 'mycorr({},{})'.format(signal, default_label)})

            key = 'mycorr({},{})'.format(signal, default_label)
            # convert dt
            if dttm_key in sbdf.columns:
                try:
                    sbdf['dttm'] = convert_jinde_datetime_to_pandas_datetime_list(sbdf[dttm_key] + 10**9)
                except Exception:
                    # fallback: try interpreting as pandas datetime
                    try:
                        sbdf['dttm'] = pd.to_datetime(sbdf[dttm_key])
                    except Exception:
                        sbdf['dttm'] = pd.NaT
            else:
                sbdf['dttm'] = pd.NaT

            # Ensure key exists
            if key not in sbdf.columns:
                sbdf[key] = np.nan

            # compute accum (handle empty)
            try:
                sbdf['accum'] = tcumsum(sbdf[key].fillna(0).values)
            except Exception:
                sbdf['accum'] = np.nan

            res[symbol] = sbdf[['dttm', key, 'accum']]

            # describe
            des_df = sbdf[key].replace([np.inf, -np.inf], np.nan).dropna().describe(percentiles=[0.05, 0.95])

            # drawdown
            try:
                max_drawdown_amp, recovery_bars, max_drawdown_start_idx, longest_drawdown_bars, longest_drawdown_start_idx, in_max_drawdown = tdrawdown(sbdf['accum'].values)
            except Exception:
                max_drawdown_amp = np.nan
                recovery_bars = np.nan
                max_drawdown_start_idx = -1
                longest_drawdown_bars = np.nan
                longest_drawdown_start_idx = -1
                in_max_drawdown = False

            max_drawdown_start = 'NA' if max_drawdown_start_idx is None or max_drawdown_start_idx == -1 else str(sbdf['dttm'].iat[int(max_drawdown_start_idx)].strftime("%Y-%m-%d"))
            longest_drawdown_start = 'NA' if longest_drawdown_start_idx is None or longest_drawdown_start_idx == -1 else str(sbdf['dttm'].iat[int(longest_drawdown_start_idx)].strftime("%Y-%m-%d"))
            max_drawdown_end = not in_max_drawdown

            drawdown_df = pd.DataFrame([max_drawdown_amp,
                                        recovery_bars,
                                        max_drawdown_end,
                                        max_drawdown_start,
                                        longest_drawdown_bars,
                                        longest_drawdown_start], index=['MaxDdAmp', 'MaxDdRecBars', 'MaxDdEnd', 'MaxDdStart', 'LongDdRecBars', 'LongDdStart'])

            des_content.append(pd.concat([des_df, drawdown_df], axis=0))
            des_columns.append(symbol)
        except Exception as e:
            fl.error('Error processing symbol for table3: %s, %s', symbol, e)
            # Append placeholder
            des_content.append(pd.Series(dtype=float))
            des_columns.append(symbol)

    # build showtab3
    if des_content:
        showtab3 = pd.concat(des_content, axis=1)
        showtab3.columns = des_columns
        showtab3 = showtab3.T
        # protect against missing stats
        if 'std' in showtab3.columns and 'count' in showtab3.columns:
            showtab3['AnnStd'] = showtab3['std'] / np.power(showtab3['count'] / 12.0, 0.5)
            showtab3['AnnSharpe'] = showtab3['mean'] / showtab3['AnnStd']
    else:
        showtab3 = pd.DataFrame()

    # build showtab3plus: concat accum columns
    df_list = []
    column_list = []
    i = 0
    for sb, sbdf in res.items():
        i += 1
        if i >= 100:
            break
        try:
            df_temp = sbdf.set_index('dttm')[['accum']].rename(columns={'accum': sb})
            df_list.append(df_temp)
            column_list.append(sb)
        except Exception:
            continue
    if df_list:
        showtab3plus = pd.concat(df_list, axis=1)
    else:
        showtab3plus = pd.DataFrame()

    return showtab3, showtab3plus

def minbar_table4(df, signal, default_label, dttm_key='dt'):
    key = 'mycorr({signal},{label})'.format(signal=signal, label=default_label)
    def table4calc(ydf):
        ydfx = ydf.reset_index(drop=True)
        ydfx['dttm'] = convert_jinde_datetime_to_pandas_datetime_list(ydfx[dttm_key]+10**9)
    #     print("ydfx:",ydfx)
        max_drawdown_amp, recovery_bars, max_drawdown_start_idx, \
            longest_drawdown_bars, longest_drawdown_start_idx, \
            in_max_drawdown = tdrawdown(ydfx['accum'].values)
        max_drawdown_start = 'NA' if max_drawdown_start_idx is None or max_drawdown_start_idx==-1 else str(ydfx['dttm'][max_drawdown_start_idx].strftime("%Y-%m-%d"))
        longest_drawdown_start = 'NA' if longest_drawdown_start_idx is None or longest_drawdown_start_idx==-1 else str(ydfx['dttm'][longest_drawdown_start_idx].strftime("%Y-%m-%d"))
        max_drawdown_end = not in_max_drawdown
        drawdown_df = pd.DataFrame([max_drawdown_amp, 
            recovery_bars, max_drawdown_end, max_drawdown_start, 
            longest_drawdown_bars, longest_drawdown_start], index=['MaxDdAmp', 
            'MaxDdRecBars', 'MaxDdEnd', 'MaxDdStart', 'LongDdRecBars', 'LongDdStart'])
        des_df = ydfx[key].replace([np.inf, -np.inf], np.nan).dropna().describe(percentiles=[0.05,0.95])
        concat_df = pd.concat([des_df, drawdown_df], axis=0).T
        return concat_df
        
    df2 = df.groupby(df[dttm_key]).mean().reset_index()
    df2['accum'] = tcumsum(df2[key])
    # print(df2)
    showtab4 = df2[[dttm_key, key, 'accum']].groupby(df2[dttm_key] // 10**13).apply(table4calc)
    showtab4 = showtab4.reset_index(level=1, drop=True)
    showtab4['AnnStd'] = showtab4['std'] / np.power(showtab4['count'] / 12.0, 0.5)
    showtab4['AnnSharpe'] = showtab4['mean'] / showtab4['AnnStd']
    return showtab4

# Get Gain
def get_gain(symbol, 
    groupname, 
    stdp, 
    signal, 
    default_label,
    demean = False,
    dttm_key='dt', 
    hist_days=20,
    start = 2005,
    end = 2020,
    freq = 'minute',
    shard='year',
    signal_root='F:/onedrive/data2/res2',
    label_root=None,
    ):

    # fl.info("Inside symbol=%s, get_gain %s, %s, %s, %s, during=%s->%s:", str(symbol), groupname, stdp, signal, default_label, start, end)

    if(type(symbol)==str):
        tab = gettab(freq=shard, shard=shard, start=start, end=end, source='h5/gp/'+signal_root,
                     universe=[symbol], vector=None)
    elif(type(symbol)==list):
        tab = gettab(freq=shard, shard=shard, start=start, end=end, 
                     source='h5/gp/'+signal_root, universe=symbol, vector=None)
    dfl = []
    while (not tab.finish()):
        try:
            if(type(symbol)==str):
                df = tab.load()[symbol]
                # print("table5 signal, single=", df)
            elif(type(symbol)==list):
                # fl.info("Curr symbol=%s tab in signal load: %s %s", str(symbol), tab.now(), str(tab.load().keys()))
                df = pd.concat(tab.load(),axis=0) #.groupby(dttm_key).mean().reset_index()
                # fl.info("table5 signal, group= %s", df)
            dfl.append(df)
        except:
            pass
        tab.roll()
    try:
        df = pd.concat(dfl, axis=0)
    except Exception as e:
        raise Exception('')
    day_mean = df[[signal]].groupby(df[dttm_key] // 10**9).mean()
    if(freq!='day'): # Bug
        day_std = df[[signal]].groupby(df[dttm_key] // 10**9).std()
    else:
        day_std = df[[signal]].groupby(df[dttm_key] // 10**9).mean()
        
    day_mean_hist = pd.DataFrame(rmean(day_mean.values, hist_days), 
        index=day_mean.index, 
        columns=['day_mean_hist'])
    day_mean_hist = day_mean_hist.shift(1)

    if(freq!='day'):
        day_std_hist = pd.DataFrame(rmean(day_std.values, hist_days), 
            index=day_std.index, 
            columns=['day_std_hist'])
    else:
        day_std_hist = pd.DataFrame(rstd(day_std.values, hist_days), 
            index=day_std.index, 
            columns=['day_std_hist'])
    day_std_hist = day_std_hist.shift(1)

    df['date'] = df[dttm_key] // 10**9
    day_mean_hist['date'] = day_mean_hist.index
    day_std_hist['date'] = day_std_hist.index
    # signal = pd.merge(pd.merge(df, day_mean_hist, on=['date']), day_std_hist, on=['date'])
    sigdf = reduce(lambda left, right: pd.merge(left, right, 
        on='date', how='outer'), [df, day_mean_hist, day_std_hist])
    if(demean):
        sigdf['adjust_adx'] = (sigdf[signal] - sigdf['day_mean_hist']) / sigdf['day_std_hist']
    else:
        sigdf['adjust_adx'] = sigdf[signal] / sigdf['day_std_hist']
    
    if(type(symbol)==str):
        tab = gettab(freq=freq, shard=shard, start=start, end=end, 
                     source='h5/gp/'+label_root, universe=[symbol], vector=None)
    elif(type(symbol)==list):
        tab = gettab(freq=freq, shard=shard, start=start, end=end, 
                     source='h5/gp/'+label_root, universe=symbol, vector=None)
    dfl = []
    while (not tab.finish()):
        try:
            if(type(symbol)==str):
                df = tab.load()[symbol]
                # print("table5 label, single=", df)
            elif(type(symbol)==list):
                # fl.info("Curr symbol=%s tab in label load: %s %s", str(symbol), tab.now(), str(tab.load().keys()))
                df = pd.concat(tab.load(),axis=0) #.groupby(dttm_key).mean().reset_index()
                # fl.info("table5 label, group= %s", df)
            dfl.append(df)
        except:
            pass
        tab.roll()
    df = pd.concat(dfl, axis=0)
    df = df.reset_index(drop=True)
    # fl.info("get_gain_label: %s %s %s %s", groupname, stdp, default_label, df)

    sigdf['label'] = df[default_label]
    
    key = "gain_{}_{}".format(groupname, stdp)
    sigdf[key] = 0
    mapping = (np.abs(sigdf['adjust_adx'])>stdp)
    sigdf[key][mapping] = np.sign(sigdf[mapping]['adjust_adx']) * sigdf[mapping]['label']
    final = sigdf[[key]].groupby(sigdf[dttm_key] // 10**9).sum()
    # fl.info("get_gain_final: %s %s", groupname, df) 
    return final

def get_table5(signal, 
    default_label,
    chosen_symbol_list=['A', 'AG', 'HC', 'B', 'BB'], 
    whole_symbol_list=[],
    stdp_list=[1,2],
    demean = False,
    dttm_key='dt',
    hist_days=20,
    start=2005,
    end=2020,
    freq='minute',
    shard='year',
    signal_root='F:/onedrive/data2/res2',
    label_root=None,
    ):

    res = []
    for sp in stdp_list:
        resdf = get_gain(
            symbol=whole_symbol_list, 
            groupname='whole_market',
            stdp=sp,
            signal=signal, 
            default_label=default_label,
            demean=demean,
            dttm_key=dttm_key, 
            hist_days=hist_days,
            start=start,
            end=end,
            freq=freq,
            shard=shard,
            signal_root=signal_root,
            label_root=label_root,
            )
        # fl.info("get_table5 whole_market: %s %s", sp, resdf)
        res.append(resdf)

    for sb in chosen_symbol_list:
        for sp in stdp_list:
            resdf = get_gain(
                symbol=sb, 
                groupname=sb, 
                stdp=sp, 
                signal=signal, 
                default_label=default_label,
                demean = demean,
                dttm_key=dttm_key, 
                hist_days=hist_days,
                start = start,
                end = end,
                freq=freq,
                shard=shard,
                signal_root=signal_root,
                label_root=label_root,
                )
            # fl.info("get_table5: %s %s", sb, resdf)
            res.append(resdf)
    
    table5 = pd.concat(res, axis = 1)
    return table5

def minbar_table5(signal, default_label,
    chosen_symbol_list=['A', 'AG', 'HC', 'B', 'BB'], 
    whole_symbol_list=[],
    stdp_list=[1,2],
    demean = False,
    dttm_key='dt',
    hist_days=20,
    start=2005,
    end=2020,
    freq='minute',
    shard='year',
    signal_root='F:/onedrive/data2/res2',
    label_root=None,
    ):


    xdf = get_table5(signal=signal, 
        default_label=default_label,
        chosen_symbol_list=chosen_symbol_list, 
        whole_symbol_list=whole_symbol_list,
        stdp_list=stdp_list,
        demean = demean,
        dttm_key=dttm_key,
        hist_days=hist_days,
        start=start,
        end=end,
        freq=freq,
        shard=shard,
        signal_root=signal_root,
        label_root=label_root,
    )

    # fl.info("minbar_table5 xdf: %s", xdf)

    xdf2 = xdf.copy(True)
    xdf2.index = convert_jinde_datetime_to_pandas_datetime_list(xdf.index*10**9)
    des_df = xdf.replace([np.inf, -np.inf], np.nan).dropna().describe(percentiles=[0.05,0.95])
    
    accum = pd.DataFrame(tcumsum(xdf2), index=xdf2.index, columns=xdf2.columns)
    
    max_drawdown_amp, recovery_bars, max_drawdown_start_idx, \
        longest_drawdown_bars, longest_drawdown_start_idx, \
        in_max_drawdown = tdrawdown(accum.values)
    
    max_drawdown_start = []
    for idx in list(max_drawdown_start_idx):
        if(idx==-1):
            max_drawdown_start.append("NA")
        else:
            max_drawdown_start.append(str(xdf2.index[idx].strftime("%Y-%m-%d")))

    longest_drawdown_start = []
    for idx in list(longest_drawdown_start_idx):
        if(idx==-1):
            longest_drawdown_start.append("NA")
        else:
            longest_drawdown_start.append(str(xdf2.index[idx].strftime("%Y-%m-%d")))

    max_drawdown_end = []
    for bl in list(in_max_drawdown):
        max_drawdown_end.append(not bl)

    drawdown_df = pd.DataFrame(np.stack([np.asarray(max_drawdown_amp), 
                                         np.asarray(recovery_bars), 
                                         np.asarray(max_drawdown_start), \
                                         np.asarray(longest_drawdown_bars), 
                                         np.asarray(longest_drawdown_start), \
                                         np.asarray(max_drawdown_end)], axis=1).T,
                                         index = ['MaxDdAmp', 
        'MaxDdRecBars', 'MaxDdStart', 'LongDdRecBars', 'LongDdStart','MaxDdEnd'],
        columns = accum.columns)

    concat_df = pd.concat([des_df, drawdown_df], axis=0).T
    # fl.info("concat_df: %s", concat_df)
    showtab5 = concat_df
    showtab5plus = accum

    showtab5['AnnStd'] = showtab5['std'] / np.power(showtab5['count'] / 12.0, 0.5)
    showtab5['AnnSharpe'] = showtab5['mean'] / showtab5['AnnStd']

    return showtab5, showtab5plus

def minbar_table6(df, signal, auto_shift_bars):
    key_list=[]
    for j in auto_shift_bars:
        key_list.append('tautocorr({},{})'.format(signal, j))
    showtab6 = df[key_list].replace([np.inf, -np.inf], np.nan).dropna().describe(percentiles=[0.05,0.95]).T
    return showtab6

def draw_minbar(meta):
    tab = gettab(
        freq=meta.freq,
        shard='forever',
        start=0,
        end=1,
        universe=meta.unv,
        source='h5/gp/'+meta.save_root+'\\map_phase',
        vector=None
    )
    # dataroot=os.path.join(meta.save_root, 'map_phase'),
    df_list = []
    for symbol in meta.unv:
        try:
            df_list.append(tab.load()[symbol])
        except Exception as e:
            fl.exception('df_list append fail.')
    try:
        df = pd.concat(df_list, axis=0).reset_index(drop=True)
    except:
        fl.exception('df_list concat fail.')
    showtab1 = minbar_table1(df, meta.signal, meta.label_list[0])
    showtab2 = minbar_table2(df, meta.signal, meta.label_list)
    showtab3, showtab3plus = minbar_table3(tab, meta.signal, meta.label_list[0],
                                           symbol_list=meta.unv, dttm_key=meta.dttm_key)
    showtab4 = minbar_table4(df, meta.signal, meta.label_list[0], dttm_key=meta.dttm_key)
    showtab5, showtab5plus = minbar_table5(
        signal=meta.signal,
        default_label=meta.label_list[0],
        demean=meta.demean,
        chosen_symbol_list=meta.extreme_symbol_list,
        whole_symbol_list=meta.unv,
        stdp_list=meta.stdp_list,
        dttm_key=meta.dttm_key,
        hist_days=meta.hist_days,
        start=meta.start,
        end=meta.end,
        freq=meta.freq,
        shard=meta.shard,
        signal_root=meta.signal_root,
        label_root=meta.label_root,
    )
    showtab6=minbar_table6(df, meta.signal, meta.auto_shift_bars)

    plot_list = []
    plot_list = plot_list + fb.plot('Time Series Signal Statistic', [('table', showtab1)])
    plot_list = plot_list + fb.plot('TS Correlation Analysis', [('table', showtab2)])
    plot_list = plot_list + fb.plot('TS Correlation Analysis', [('table', showtab3)])
    plot_list = plot_list + fb.plot('TS Cumulative Correlation Analysis', [('line', showtab3plus)])
    plot_list = plot_list + fb.plot('TS Yearly Correlation Analysis', [('table', showtab4)])
    plot_list = plot_list + fb.plot('Extreme Accum', [('line', showtab5plus)])
    plot_list = plot_list + fb.plot('Extreme', [('table', showtab5)])
    plot_list = plot_list + fb.plot('TS Auto Correlation Analysis', [('table', showtab6)])
    fb.save_html(os.path.join(meta.save_root, 
        "ts_report", 
        meta.signal, 
        "{}_{}.html".format(meta.freq, meta.shard)
        ), 
    plot_list)

    fl.info("Plase see html under %s", meta.save_root)

def minbar_gen(**kwargs):
    meta = ds.load(kwargs)
    fl.init(os.path.join(meta.log_root, "ts_report_minbar_gen_main.log"))
    meta.freq = 'minute'
    meta.shard = 'year'
    map_phase(meta) # for debug, comment it
    draw_minbar(meta)

def daybar_gen(**kwargs):
    meta = ds.load(kwargs)
    fl.init(os.path.join(meta.log_root, "ts_report_minbar_gen_main.log"))
    meta.freq = 'day'
    meta.shard = 'forever'
    meta.start = 0
    meta.end = 1
    map_phase(meta) # for debug, comment it
    draw_minbar(meta)

