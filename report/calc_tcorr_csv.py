import pandas as pd
import numpy as np
import os


# ----------------- Helper Functions -----------------

def tmean(x):
    return np.mean(x)


def tstd(x):
    return np.std(x)


def tcount(x):
    return len(x)


def tcorr(x, y):
    """Pearson correlation"""
    if len(x) < 2: return np.nan
    return pd.Series(x).corr(pd.Series(y))


def tcos(x, y):
    """Cosine similarity"""
    if len(x) < 2: return np.nan
    dot = np.dot(x, y)
    norma = np.linalg.norm(x)
    normb = np.linalg.norm(y)
    if norma == 0 or normb == 0:
        return np.nan
    return dot / (norma * normb)


def tcumsum(x):
    return np.cumsum(x)


def tdrawdown(xs):
    """
    Calculate drawdown statistics.
    Returns:
    min_dd, rec_bars, start_idx, longest_duration, real_longest_start, in_max_dd
    """
    xs = np.array(xs)
    if len(xs) == 0:
        return 0, 0, -1, 0, -1, False

    hwm = np.maximum.accumulate(xs)
    drawdowns = xs - hwm

    # 1. Max Drawdown Amp
    min_dd = np.min(drawdowns)
    if min_dd == 0:
        return 0, 0, -1, 0, -1, False

    min_dd_idx = np.argmin(drawdowns)

    # 2. Max Drawdown Start
    peak_val = hwm[min_dd_idx]
    candidates = np.where(xs[:min_dd_idx + 1] == peak_val)[0]
    if len(candidates) > 0:
        start_idx = candidates[-1]
    else:
        start_idx = 0

    # 3. Recovery Bars for Max DD
    recovery_candidates = np.where(xs[min_dd_idx:] >= peak_val)[0]
    if len(recovery_candidates) > 0:
        end_idx = min_dd_idx + recovery_candidates[0]
        rec_bars = end_idx - start_idx
        in_max_dd = False
    else:
        rec_bars = len(xs) - 1 - start_idx
        in_max_dd = True

    # 4. Longest Drawdown
    is_dd = drawdowns < 0
    framed = np.concatenate(([0], is_dd.astype(int), [0]))
    abs_diff = np.abs(np.diff(framed))
    ranges = np.where(abs_diff == 1)[0].reshape(-1, 2)

    longest_duration = 0
    longest_start = -1

    for start, end in ranges:
        duration = end - start
        if duration > longest_duration:
            longest_duration = duration
            longest_start = start

    real_longest_start = longest_start - 1 if longest_start > 0 else 0

    return min_dd, rec_bars, start_idx, longest_duration, real_longest_start, in_max_dd


# ----------------- Table Functions -----------------

def minbar_table1(data_dict, signal_col, label_col):
    """
    Calculate statistics on the RAW whole sample (all symbols, all timestamps).
    Replaces older logic which used monthly aggregated stats.
    """
    print("--- Generating minbar_table1 (Whole Sample Statistics) ---")

    # Collect all raw data
    extracted_data = []
    for sym, df in data_dict.items():
        temp = df[[signal_col, label_col]].copy()
        extracted_data.append(temp)

    if not extracted_data:
        return pd.DataFrame()

    full_raw_df = pd.concat(extracted_data, axis=0)

    # Describe automatically calculates count, mean, std, min, max, percentiles
    stats = full_raw_df.describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99])

    # Calculate Skew and Kurtosis
    skew = full_raw_df.skew()
    kurt = full_raw_df.kurt()

    # Add Skew/Kurt rows to the stats dataframe
    stats.loc['skew'] = skew
    stats.loc['kurtosis'] = kurt

    return stats


def minbar_table2(df, signal, label_list):
    lblist = []
    for lb in label_list:
        lblist.append('mycorr({signal},{label})'.format(signal=signal, label=lb))

    lblist = [c for c in lblist if c in df.columns]

    # Return summary table AND raw series for plotting
    raw_ic_series = df[lblist].replace([np.inf, -np.inf], np.nan).dropna()

    showtab2 = raw_ic_series.describe(percentiles=[0.05, 0.95]).T

    # If there are multiple label columns, we might need to decide which one to plot or plot them all.
    # Usually label_list has only one item here based on usage.

    return showtab2, raw_ic_series


def minbar_table3(all_results_dict, signal, default_label):
    res = {}  # Store accum series for showtab3plus
    des_content = []
    des_columns = []

    col_name = f'mycorr({signal},{default_label})'

    # Whole Market (average)
    extracted_series = []
    for sym, df in all_results_dict.items():
        if col_name in df.columns:
            s = df[col_name]
            s.name = sym
            extracted_series.append(s)

    if extracted_series:
        combined = pd.concat(extracted_series, axis=1)
        mean_series = combined.mean(axis=1)
        whole_market_df = pd.DataFrame(mean_series, columns=[col_name])
    else:
        whole_market_df = pd.DataFrame(columns=[col_name])

    process_list = ['whole_market'] + sorted(list(all_results_dict.keys()))

    for symbol in process_list:
        try:
            if symbol == 'whole_market':
                sbdf = whole_market_df.copy()
            else:
                sbdf = all_results_dict[symbol].copy()

            if col_name not in sbdf.columns:
                sbdf[col_name] = np.nan

            sbdf = sbdf.dropna(subset=[col_name])
            sbdf['accum'] = tcumsum(sbdf[col_name].values)

            # Store accumulation for plotting
            res[symbol] = sbdf['accum']

            des_df = sbdf[col_name].describe(percentiles=[0.05, 0.95])

            max_dd, rec_bars, start_idx, long_dd, long_start, in_dd = tdrawdown(sbdf['accum'].values)

            dates = sbdf.index
            max_dd_start = str(dates[start_idx].date()) if (start_idx >= 0 and start_idx < len(dates)) else 'NA'
            long_dd_start = str(dates[long_start].date()) if (long_start >= 0 and long_start < len(dates)) else 'NA'
            max_dd_end = not in_dd

            drawdown_stats = pd.Series([
                max_dd, rec_bars, max_dd_end, max_dd_start, long_dd, long_dd_start
            ], index=['MaxDdAmp', 'MaxDdRecBars', 'MaxDdEnd', 'MaxDdStart', 'LongDdRecBars', 'LongDdStart'])

            full_stats = pd.concat([des_df, drawdown_stats])
            des_content.append(full_stats)
            des_columns.append(symbol)

        except Exception as e:
            print(f"Error in table3 for {symbol}: {e}")

    showtab3plus = pd.DataFrame()
    if res:
        # Align all accum series
        showtab3plus = pd.concat(res, axis=1)

    if des_content:
        showtab3 = pd.concat(des_content, axis=1)
        showtab3.columns = des_columns
        showtab3 = showtab3.T

        if 'std' in showtab3.columns and 'mean' in showtab3.columns and 'count' in showtab3.columns:
            try:
                showtab3['AnnStd'] = showtab3['std'].astype(float) / np.power(showtab3['count'].astype(float) / 12.0,
                                                                              0.5)
                showtab3['AnnSharpe'] = showtab3['mean'].astype(float) / showtab3['AnnStd']
            except:
                pass
        return showtab3, showtab3plus
    return pd.DataFrame(), pd.DataFrame()


def minbar_table4(df, signal, default_label):
    if isinstance(df.index, pd.DatetimeIndex):
        mean_df = df.groupby(level=0).mean()
    else:
        return pd.DataFrame()

    key = 'mycorr({signal},{label})'.format(signal=signal, label=default_label)
    if key not in mean_df.columns:
        return pd.DataFrame()

    mean_df['accum'] = tcumsum(mean_df[key].values)

    def table4calc(ydf):
        ydfx = ydf.copy()
        max_dd, rec_bars, start_idx, long_dd, long_start, in_dd = tdrawdown(ydfx['accum'].values)

        dates = ydfx.index
        max_dd_start = str(dates[start_idx].date()) if (start_idx >= 0 and start_idx < len(dates)) else 'NA'
        long_dd_start = str(dates[long_start].date()) if (long_start >= 0 and long_start < len(dates)) else 'NA'
        max_dd_end = not in_dd

        drawdown_df = pd.Series([
            max_dd, rec_bars, max_dd_end, max_dd_start, long_dd, long_dd_start
        ], index=['MaxDdAmp', 'MaxDdRecBars', 'MaxDdEnd', 'MaxDdStart', 'LongDdRecBars', 'LongDdStart'])

        des_df = ydfx[key].describe(percentiles=[0.05, 0.95])
        return pd.concat([des_df, drawdown_df])

    res = mean_df.groupby(mean_df.index.year).apply(table4calc)

    if 'std' in res.columns:
        res['AnnStd'] = res['std'] * np.sqrt(12)
        res['AnnSharpe'] = res['mean'] / res['AnnStd']

    return res


def minbar_table4_monthly(df, signal, default_label):
    if isinstance(df.index, pd.DatetimeIndex):
        mean_df = df.groupby(level=0).mean()
    else:
        return pd.DataFrame()

    key = 'mycorr({signal},{label})'.format(signal=signal, label=default_label)
    if key not in mean_df.columns:
        return pd.DataFrame()

    def table4calc(ydf):
        ydfx = ydf.copy()
        # Accumulate the returns for this specific month across years to see if the "Month X" strategy works
        ydfx['accum'] = tcumsum(ydfx[key].values)

        max_dd, rec_bars, start_idx, long_dd, long_start, in_dd = tdrawdown(ydfx['accum'].values)

        dates = ydfx.index
        max_dd_start = str(dates[start_idx].date()) if (start_idx >= 0 and start_idx < len(dates)) else 'NA'
        long_dd_start = str(dates[long_start].date()) if (long_start >= 0 and long_start < len(dates)) else 'NA'
        max_dd_end = not in_dd

        drawdown_df = pd.Series([
            max_dd, rec_bars, max_dd_end, max_dd_start, long_dd, long_dd_start
        ], index=['MaxDdAmp', 'MaxDdRecBars', 'MaxDdEnd', 'MaxDdStart', 'LongDdRecBars', 'LongDdStart'])

        des_df = ydfx[key].describe(percentiles=[0.05, 0.95])
        return pd.concat([des_df, drawdown_df])

    res = mean_df.groupby(mean_df.index.month).apply(table4calc)

    if 'std' in res.columns:
        res['AnnStd'] = res['std'] * np.sqrt(12)
        res['AnnSharpe'] = res['mean'] / res['AnnStd']

    return res


def minbar_table6(df, signal, auto_shift_bars):
    key_list = []
    for j in auto_shift_bars:
        key_list.append('tautocorr({signal},{j})'.format(signal=signal, j=j))

    key_list = [c for c in key_list if c in df.columns]

    showtab6 = df[key_list].replace([np.inf, -np.inf], np.nan).dropna().describe(percentiles=[0.05, 0.95]).T

    # Prepare plot data: Lag vs Mean
    plot_data = pd.DataFrame()
    try:
        if 'mean' in showtab6.columns:
            tmp = showtab6[['mean']].copy()
            # Extract lag k from index string 'tautocorr(signal,k)'
            import re
            def extract_lag(s):
                m = re.search(r'tautocorr\([^,]+,\s*(\d+)\)', s)
                return int(m.group(1)) if m else np.nan

            tmp['lag'] = tmp.index.map(extract_lag)
            tmp = tmp.dropna().sort_values('lag').set_index('lag')
            plot_data = tmp
    except Exception as e:
        print(f"Error preparing table6 plot data: {e}")

    return showtab6, plot_data


# ----------------- Table 5 (Strategy) -----------------

def get_gain_csv(data_dict, target, stdp, signal_col, label_col, hist_days=20):
    if target == 'whole_market':
        df_list = []
        for sym, df in data_dict.items():
            temp = df.copy()
            temp['symbol'] = sym
            df_list.append(temp)
        if not df_list:
            return pd.DataFrame()
        df = pd.concat(df_list, axis=0)
    else:
        if target not in data_dict:
            return pd.DataFrame()
        df = data_dict[target].copy()

    # DEBUG
    # print(f"DEBUG: get_gain_csv target={target} columns={df.columns}")

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            return pd.DataFrame()

    df['date_key'] = df.index.date

    if signal_col not in df.columns:
        # Maybe index name is conflicting or something?
        # Try to see if it's in index? No.
        raise ValueError(f"Column not found: {signal_col}. Available: {df.columns.tolist()}")

    # Calculate daily stats for signal standardization
    daily_stats = df.groupby('date_key')[signal_col].agg(['mean', 'std'])

    # Shifted rolling window for history
    if daily_stats['std'].isna().all() or (daily_stats['std'] == 0).all():
        # Fallback for single symbol (or constant cross-section): calculate rolling std of the mean (signal value)
        daily_stats['day_mean_hist'] = daily_stats['mean'].rolling(window=hist_days).mean().shift(1)
        daily_stats['day_std_hist'] = daily_stats['mean'].rolling(window=hist_days).std().shift(1)
    else:
        # Cross-sectional logic (for whole_market)
        daily_stats['day_mean_hist'] = daily_stats['mean'].rolling(window=hist_days).mean().shift(1)
        # Use mean of x-sectional std as 'history std'
        daily_stats['day_std_hist'] = daily_stats['std'].rolling(window=hist_days).mean().shift(1)

    # Merge back to minutes/intraday data
    df = df.merge(daily_stats[['day_mean_hist', 'day_std_hist']], left_on='date_key', right_index=True, how='left')

    # Calculate standardized signal
    # Avoid division by zero
    df['day_std_hist'] = df['day_std_hist'].replace(0, np.nan)
    df['adjust_adx'] = df[signal_col] / df['day_std_hist']

    # Strategy logic
    key = f"gain_{target}_{stdp}"
    df[key] = 0.0

    mask = np.abs(df['adjust_adx']) > stdp
    valid_mask = mask & df['adjust_adx'].notna() & df[label_col].notna()

    # If standard deviation is missing, we can't trade
    trade_rets = np.sign(df.loc[valid_mask, 'adjust_adx']) * df.loc[valid_mask, label_col]
    df.loc[valid_mask, key] = trade_rets

    # Calculate Win Rate Stats
    total_trades = len(trade_rets)
    win_trades = (trade_rets > 0).sum()
    win_rate = win_trades / total_trades if total_trades > 0 else 0.0

    # Aggregating daily returns
    final = df.groupby('date_key')[[key]].sum()
    final.index = pd.to_datetime(final.index)

    stats = {
        'win_rate': win_rate,
        'win_trades': win_trades,
        'total_trades': total_trades
    }

    return final, stats


def minbar_table5_csv(data_dict, signal_col, label_col, stdp_list=[1, 2], chosen_symbols=None, target_symbol='000852.SH'):
    print("--- Generating minbar_table5 (Strategy Backtest) ---")

    res_list = []
    stats_dict = {} # Key: column name, Value: {win_rate, ...}

    # 1. Whole Market
    for sp in stdp_list:
        try:
            resdf, stats = get_gain_csv(data_dict, 'whole_market', sp, signal_col, label_col)
            if not resdf.empty:
                res_list.append(resdf)
                stats_dict[resdf.columns[0]] = stats
        except Exception as e:
            print(f"Error calculating gain for whole_market stdp={sp}: {e}")

    # 2. Chosen Symbols
    if chosen_symbols is None:
        all_syms = sorted(list(data_dict.keys()))
        # Prioritize target_symbol if exists
        targets = [s for s in all_syms if target_symbol in s]
        if targets:
            chosen_symbols = targets
        else:
            chosen_symbols = all_syms[:5]

    for sb in chosen_symbols:
        for sp in stdp_list:
            try:
                resdf, stats = get_gain_csv(data_dict, sb, sp, signal_col, label_col)
                if not resdf.empty:
                    res_list.append(resdf)
                    stats_dict[resdf.columns[0]] = stats
            except Exception as e:
                print(f"Error calculating gain for {sb} stdp={sp}: {e}")

    if not res_list:
        return pd.DataFrame(), pd.DataFrame()

    table5 = pd.concat(res_list, axis=1)

    # Stats
    des_df = table5.replace([np.inf, -np.inf], np.nan).dropna().describe(percentiles=[0.05, 0.95])

    # Add Win Rate Row to des_df (or helper)
    # des_df has columns matching table5 columns
    # We want to add a row 'WinRate'

    win_rate_row = pd.Series(index=table5.columns, dtype=float)
    trade_count_row = pd.Series(index=table5.columns, dtype=float)

    for col in table5.columns:
        if col in stats_dict:
            win_rate_row[col] = stats_dict[col]['win_rate']
            trade_count_row[col] = stats_dict[col]['total_trades']

    # Append to des_df (need to transpose, add, transpose back OR just append if des_df index is rows)
    # des_df is (stats x columns). index is ['count', 'mean', ...]
    des_df.loc['WinRate'] = win_rate_row
    des_df.loc['TradeCount'] = trade_count_row

    # Accumulation (Equity Curve)
    accum = table5.fillna(0).cumsum()

    # Drawdown Metrics
    dd_metrics = []
    for col in accum.columns:
        vals = accum[col].values
        try:
            m_amp, m_rec, m_start_idx, l_bars, l_start_idx, in_dd = tdrawdown(vals)
            dates = accum.index
            m_start = str(dates[m_start_idx].date()) if m_start_idx != -1 else "NA"
            l_start = str(dates[l_start_idx].date()) if l_start_idx != -1 else "NA"
            dd_metrics.append([m_amp, m_rec, m_start, l_bars, l_start, not in_dd])
        except Exception:
            dd_metrics.append([np.nan, np.nan, "NA", np.nan, "NA", "NA"])

    drawdown_df = pd.DataFrame(dd_metrics,
                               index=accum.columns,
                               columns=['MaxDdAmp', 'MaxDdRecBars', 'MaxDdStart', 'LongDdRecBars', 'LongDdStart',
                                        'MaxDdEnd']).T

    showtab5 = pd.concat([des_df, drawdown_df], axis=0).T

    if 'std' in showtab5.columns and 'mean' in showtab5.columns:
        # Assuming daily returns
        showtab5['AnnStd'] = showtab5['std'] * np.sqrt(250)
        try:
            # Handle potential division by zero
            showtab5['AnnSharpe'] = (showtab5['mean'] * 250).div(showtab5['AnnStd']).replace([np.inf, -np.inf], np.nan)
        except Exception:
            showtab5['AnnSharpe'] = np.nan

    return showtab5, accum


def calc_daily_cross_sectional_ic(data_dict, lookback=30):
    """
    Calculate daily cross-sectional IC and return the last N days.
    """
    df_list = []
    for sym, df in data_dict.items():
        temp = df[['signal', 'label']].copy()
        # temp['symbol'] = sym # Not strictly needed for correlation
        df_list.append(temp)

    if not df_list:
        return pd.DataFrame()

    # Concatenate all symbols, aligning on index (Date)
    full_df = pd.concat(df_list, axis=0)

    # Group by Date and calculate correlation
    def daily_ic(g):
        if len(g) < 2: return np.nan
        return g['signal'].corr(g['label'])

    daily_series = full_df.groupby(full_df.index).apply(daily_ic)
    daily_series = daily_series.dropna()

    daily_series.name = 'Daily_Cross_Sectional_IC'

    return daily_series.tail(lookback)


def minbar_table7_csv(data_dict, signal_col, label_col, chosen_symbols=None, target_symbol='000852.SH'):
    """
    Simple Long-Short Strategy (Sign Strategy).
    Position = sign(signal).
    Return = Position * label.
    """
    print("--- Generating minbar_table7 (Simple Long-Short Strategy) ---")

    res_list = []

    # 1. Whole Market Strategy
    try:
        # For whole market, we strategy on each asset then aggregate
        # Re-use get_gain_csv with stdp=0 which effectively does sign(signal) * return
        # But get_gain_csv does standardization. User asked for "pred positive..., negative...", simple sign.
        # Let's write a simple calculator for this specific request to be sure.

        # Aggregate all data
        df_list = []
        for sym, df in data_dict.items():
            temp = df.copy()
            temp['symbol'] = sym
            df_list.append(temp)

        if df_list:
            whole_df = pd.concat(df_list, axis=0)
            # Ensure datetime index
            if not isinstance(whole_df.index, pd.DatetimeIndex):
                whole_df.index = pd.to_datetime(whole_df.index)

            whole_df['date_key'] = whole_df.index.date

            key = 'gain_whole_market_simple'
            whole_df[key] = np.sign(whole_df[signal_col].fillna(0)) * whole_df[label_col].fillna(0)

            # Aggregate by date
            res_whole = whole_df.groupby('date_key')[[key]].sum()
            res_whole.index = pd.to_datetime(res_whole.index)
            res_list.append(res_whole)
    except Exception as e:
        print(f"Error table7 whole_market: {e}")

    # 2. Chosen Symbols
    if chosen_symbols is None:
        all_syms = sorted(list(data_dict.keys()))
        targets = [s for s in all_syms if target_symbol in s]
        if targets:
            chosen_symbols = targets
        else:
            chosen_symbols = all_syms[:5]

    for sb in chosen_symbols:
        try:
            if sb not in data_dict: continue
            df = data_dict[sb].copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df['date_key'] = df.index.date

            key = f'gain_{sb}_simple'
            key_bench = f'bench_{sb}'

            # Simple sign strategy
            sig = df[signal_col].fillna(0)
            lab = df[label_col].fillna(0)

            df[key] = np.sign(sig) * lab
            df[key_bench] = lab

            res_sb = df.groupby('date_key')[[key, key_bench]].sum()
            res_sb.index = pd.to_datetime(res_sb.index)
            res_list.append(res_sb)
        except Exception as e:
            print(f"Error table7 {sb}: {e}")

    if not res_list:
        return pd.DataFrame(), pd.DataFrame()

    table7 = pd.concat(res_list, axis=1)

    # Stats
    des_df = table7.replace([np.inf, -np.inf], np.nan).dropna().describe(percentiles=[0.05, 0.95])

    # Cumulative Sum
    accum = table7.fillna(0).cumsum()

    # Drawdown metrics (reusing tdrawdown)
    dd_metrics = []
    for col in accum.columns:
        vals = accum[col].values
        try:
            m_amp, m_rec, m_start_idx, l_bars, l_start_idx, in_dd = tdrawdown(vals)
            dates = accum.index
            m_start = str(dates[m_start_idx].date()) if m_start_idx != -1 else "NA"
            l_start = str(dates[l_start_idx].date()) if l_start_idx != -1 else "NA"
            dd_metrics.append([m_amp, m_rec, m_start, l_bars, l_start, not in_dd])
        except Exception:
            dd_metrics.append([np.nan, np.nan, "NA", np.nan, "NA", "NA"])

    drawdown_df = pd.DataFrame(dd_metrics,
                               index=accum.columns,
                               columns=['MaxDdAmp', 'MaxDdRecBars', 'MaxDdStart', 'LongDdRecBars', 'LongDdStart',
                                        'MaxDdEnd']).T

    showtab7 = pd.concat([des_df, drawdown_df], axis=0).T

    if 'std' in showtab7.columns and 'mean' in showtab7.columns:
        showtab7['AnnStd'] = showtab7['std'] * np.sqrt(250)
        try:
            showtab7['AnnSharpe'] = (showtab7['mean'] * 250).div(showtab7['AnnStd']).replace([np.inf, -np.inf], np.nan)
        except:
            showtab7['AnnSharpe'] = np.nan

    return showtab7, accum


# ----------------- Table 8 (Label Accumulation) -----------------

def minbar_table8_csv(data_dict, label_col, target_symbol='000852.SH'):
    """
    Accumulation of Label Returns (Benchmark).
    """
    print("--- Generating minbar_table8 (Label Accumulation, All Symbols) ---")
    res_list = []

    # 1. Whole Market (Average Return)
    try:
        df_list = []
        for sym, df in data_dict.items():
            temp = df.copy()
            temp['symbol'] = sym
            df_list.append(temp)
        if df_list:
            whole_df = pd.concat(df_list, axis=0)
            if not isinstance(whole_df.index, pd.DatetimeIndex):
                whole_df.index = pd.to_datetime(whole_df.index)
            whole_df['date_key'] = whole_df.index.date

            daily = whole_df.groupby('date_key')[label_col].mean()
            daily.name = 'whole_market_label'
            daily.index = pd.to_datetime(daily.index)
            res_list.append(daily)
    except Exception as e:
        print(f"Error table8 whole_market: {e}")

    # 2. Individual Symbols
    # Filter to target_symbol and whole_market to be safe and consistent with "Others only stats target_symbol".

    targets = [s for s in data_dict.keys() if target_symbol in s]
    if not targets: targets = list(data_dict.keys())[:5]

    for sym in targets:
        try:
            df = data_dict[sym].copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df['date_key'] = df.index.date

            d = df.groupby('date_key')[label_col].sum()
            d.name = f"{sym}_label"
            d.index = pd.to_datetime(d.index)
            res_list.append(d)
        except:
            pass

    if not res_list: return pd.DataFrame()

    table8 = pd.concat(res_list, axis=1).fillna(0).cumsum()
    return table8


def minbar_table9_csv(data_dict, signal_col, target_symbol='000852.SH'):
    """
    Calculates statistics and cumulative returns for the signal of each symbol.
    Includes max drawdown calculation.
    """
    print("--- Generating minbar_table9 (Signal Analysis) ---")
    res_list = []
    stats_list = []
    index_list = []

    # 1. Whole Market (Average Signal)
    try:
        df_list = []
        for sym, df in data_dict.items():
            df_list.append(df[[signal_col]])
        if df_list:
            whole_df = pd.concat(df_list, axis=0)
            if not isinstance(whole_df.index, pd.DatetimeIndex):
                whole_df.index = pd.to_datetime(whole_df.index)
            whole_df['date_key'] = whole_df.index.date

            daily_mean_signal = whole_df.groupby('date_key')[signal_col].mean()
            daily_mean_signal.name = 'whole_market_signal'
            daily_mean_signal.index = pd.to_datetime(daily_mean_signal.index)
            res_list.append(daily_mean_signal)
    except Exception as e:
        print(f"Error table9 whole_market: {e}")

    # 2. Individual Symbols
    all_symbols = sorted(list(data_dict.keys()))
    for sym in all_symbols:
        try:
            df = data_dict[sym].copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df['date_key'] = df.index.date

            daily_signal = df.groupby('date_key')[signal_col].sum() # Summing daily signals
            daily_signal.name = f"{sym}_signal"
            daily_signal.index = pd.to_datetime(daily_signal.index)
            res_list.append(daily_signal)
        except Exception as e:
            print(f"Error processing signal for {sym} in table9: {e}")

    if not res_list:
        return pd.DataFrame(), pd.DataFrame()

    # Combine all signal series
    signal_returns = pd.concat(res_list, axis=1).fillna(0)
    signal_accum = signal_returns.cumsum()

    # Calculate statistics including drawdown for each series
    for col in signal_accum.columns:
        series_accum = signal_accum[col]
        series_raw = signal_returns[col]

        stats = series_raw.describe(percentiles=[0.05, 0.95])

        max_dd, rec_bars, start_idx, long_dd, long_start, in_dd = tdrawdown(series_accum.values)

        dates = series_accum.index
        max_dd_start = str(dates[start_idx].date()) if (start_idx >= 0 and start_idx < len(dates)) else 'NA'
        long_dd_start = str(dates[long_start].date()) if (long_start >= 0 and long_start < len(dates)) else 'NA'
        max_dd_end = not in_dd

        drawdown_stats = pd.Series([
            max_dd, rec_bars, max_dd_end, max_dd_start, long_dd, long_dd_start
        ], index=['MaxDdAmp', 'MaxDdRecBars', 'MaxDdEnd', 'MaxDdStart', 'LongDdRecBars', 'LongDdStart'])

        full_stats = pd.concat([stats, drawdown_stats])
        stats_list.append(full_stats)
        index_list.append(col)

    stats_df = pd.concat(stats_list, axis=1).T
    stats_df.index = index_list

    return stats_df, signal_accum


# ----------------- HTML Report Generation -----------------

try:
    import io_bokeh as fb
    from bokeh.models.widgets import Div
    from bokeh.plotting import figure  # Ensure this is imported
    from bokeh.models import ColumnDataSource, Range1d

    HAS_BOKEH = True
except ImportError:
    print("Warning: io_bokeh or bokeh not found. HTML generation will use fallback.")
    HAS_BOKEH = False


def generate_html_report_csv(output_dir, report_name,
                             t1, t2, t3, t3plus, t4, t5, t5plus, t6, t7, t7plus, t9, t9plus, t6plus, ic_raw_data=None,
                             t4_monthly=pd.DataFrame(), last_30_days_ic=pd.Series(dtype=float),
                             rolling_ic_comparison=pd.DataFrame(), target_symbol='000852.SH'):
    save_path = os.path.join(output_dir, f"{report_name}.html")

    if HAS_BOKEH:
        try:
            plot_list = []

            # --- Custom HTML Header and Styles ---
            header_html = f"""
            <style>
              body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; margin: 0; padding: 20px; background-color: #f5f5f5; }}
              .report-container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
              .main-title {{ text-align: center; font-size: 32px; font-weight: 800; color: #1a1a1a; margin-bottom: 12px; }}
              .sub-title {{ text-align: center; font-size: 20px; font-weight: 700; color: #4a4a4a; margin-bottom: 30px; }}

              .section-header {{
                margin-top: 50px;
                margin-bottom: 30px;
                padding-bottom: 15px;
                border-bottom: 2px solid #eaeaea;
                text-align: center;
              }}
              .section-title {{ font-size: 28px; font-weight: 800; color: #2c3e50; margin-bottom: 10px; }}
              .section-desc {{ font-size: 16px; color: #666; line-height: 1.6; max-width: 900px; margin: 0 auto; }}

              .chart-container {{ margin: 20px 0; }}
            </style>

            <div class='report-container'>
              <div class='main-title'>Alpha Factor Analysis Report</div>
              <div class='sub-title'>Time Series & Correlation Analysis</div>
              <div style='text-align: center; color: #666; margin-bottom: 40px; font-size: 14px;'>
                {target_symbol}(TCN-model)
              </div>
            """
            plot_list.append(Div(text=header_html, sizing_mode='stretch_width'))

            # --- Section 1: Signal Statistics ---
            section1_html = """
            <div class='section-header'>
                <div class='section-title'>Part 1: Signal Statistics & Auto-Correlation</div>
                <div class='section-desc'>Analysis of signal distribution statistics and time-series auto-correlation properties.</div>
            </div>
            """
            plot_list.append(Div(text=section1_html, sizing_mode='stretch_width'))

            if not t1.empty:
                plot_list.append(
                    Div(text="<div style='font-weight: 600; font-size: 16px; margin: 20px 0 10px;'>1.1 Signal Statistics (Table 1 - Whole Sample)</div>",
                        sizing_mode='stretch_width'))
                plot_list += fb.plot('', [('table', t1)])  # removed title from plot to use custom div

            if not t6.empty:
                plot_list.append(
                    Div(text="<div style='font-weight: 600; font-size: 16px; margin: 20px 0 10px;'>1.2 Signal Auto Correlation (Table 6)</div>",
                        sizing_mode='stretch_width'))
                plot_list += fb.plot('', [('table', t6)])

            if not t6plus.empty:
                plot_list.append(
                    Div(text="<div style='font-weight: 600; font-size: 16px; margin: 20px 0 10px;'>1.3 Auto Correlation Decay</div>",
                        sizing_mode='stretch_width'))
                plot_list += fb.plot('TS Auto Correlation Mean (Lag 1-10)', [('line', t6plus)])

            # --- Section 2: IC Performance ---
            section2_html = """
            <div class='section-header'>
                <div class='section-title'>Part 2: IC Performance Analysis</div>
                <div class='section-desc'>Evaluation of Information Coefficient (IC) across time and symbols.</div>
            </div>
            """
            plot_list.append(Div(text=section2_html, sizing_mode='stretch_width'))

            if not t2.empty:
                plot_list.append(
                    Div(text="<div style='font-weight: 600; font-size: 16px; margin: 20px 0 10px;'>2.1 IC Summary (Table 2)</div>",
                        sizing_mode='stretch_width'))
                plot_list += fb.plot('', [('table', t2)])

            if ic_raw_data is not None and not ic_raw_data.empty:
                try:
                    # Assume first column is the IC series 'mycorr(signal,label)'
                    series_name = ic_raw_data.columns[0]
                    data = ic_raw_data[series_name].dropna().values

                    if len(data) > 1:
                        # 1. Statistics
                        mu = np.mean(data)
                        sigma = np.std(data)

                        # 2. Histogram (Count)
                        hist, edges = np.histogram(data, density=False, bins=50)

                        # 4. Plot using quad for correct alignment on a numeric axis
                        p = figure(
                            title=f"IC 频率分布直方图: {series_name} (Mean={mu:.4f}, Std={sigma:.4f})",
                            tools="pan,wheel_zoom,box_zoom,reset,save",
                            background_fill_color="#fafafa", height=400, width=1000)

                        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                               fill_color="navy", line_color="white", alpha=0.6, legend_label="IC 频次")


                        p.yaxis.axis_label = "频次 (Count)"
                        p.xaxis.axis_label = "IC 值"
                        p.legend.location = "top_right"
                        p.grid.grid_line_color = "white"

                        plot_list.append(p)
                except Exception as e:
                    print(f"Error plotting IC histogram: {e}")

            if not t3.empty:
                plot_list.append(
                    Div(text="<div style='font-weight: 600; font-size: 16px; margin: 20px 0 10px;'>2.2 IC by Symbol (Table 3)</div>",
                        sizing_mode='stretch_width'))
                plot_list += fb.plot('', [('table', t3)])

            if not t3plus.empty:
                plot_list.append(
                    Div(text="<div style='font-weight: 600; font-size: 16px; margin: 20px 0 10px;'>2.3 Cumulative IC Series</div>",
                        sizing_mode='stretch_width'))
                plot_list += fb.plot('Cumulative IC Series $\sum Corr(S_t, L_t)$', [('line', t3plus)])

            if not t4.empty:
                desc_t4 = """
                <div style='background-color: #f8f9fa; border-left: 4px solid #4a90e2; padding: 10px 15px; margin: 20px 0; font-size: 14px; color: #555;'>
                    <strong>Yearly IC Analysis (Table 4):</strong><br>
                    Shows the average IC for each year to analyze the stability of the factor's performance over time.
                </div>
                """
                plot_list.append(Div(text=desc_t4, sizing_mode='stretch_width'))

                plot_list.append(
                    Div(text="<div style='font-weight: 600; font-size: 16px; margin: 20px 0 10px;'>2.4 Yearly IC Analysis (Table 4)</div>",
                        sizing_mode='stretch_width'))
                plot_list += fb.plot('', [('table', t4)])

                # Plot Yearly IC Mean Bar Chart
                try:
                    if 'mean' in t4.columns:
                        years = [str(y) for y in t4.index]
                        means = t4['mean'].values

                        # Conditional colors: Red for positive, Green for negative (China market convention)
                        colors = ['#ef5350' if x >= 0 else '#66bb6a' for x in means]

                        source = ColumnDataSource(data=dict(years=years, means=means, colors=colors))

                        p_year = figure(x_range=years, height=350, title="Yearly IC Mean",
                                        tools="pan,wheel_zoom,box_zoom,reset,save",
                                        tooltips=[("Year", "@years"), ("IC Mean", "@means{0.0000}")],
                                        x_axis_label="Year", y_axis_label="IC Mean",
                                        background_fill_color="#fafafa")

                        p_year.vbar(x='years', top='means', width=0.8, color='colors', source=source)

                        p_year.xgrid.grid_line_color = None
                        p_year.y_range.start = min(0, min(means) * 1.2)
                        p_year.y_range.end = max(0, max(means) * 1.2)
                        # Rotate x-axis labels
                        p_year.xaxis.major_label_orientation = np.pi / 4

                        plot_list.append(p_year)
                except Exception as e:
                    print(f"Error plotting Table 4 chart: {e}")

            if not t4_monthly.empty:
                desc_t4m = """
                <div style='background-color: #f8f9fa; border-left: 4px solid #4a90e2; padding: 10px 15px; margin: 20px 0; font-size: 14px; color: #555;'>
                    <strong>Monthly IC Analysis (Seasonality):</strong><br>
                    Shows the average IC for each calendar month (Jan-Dec) aggregated across all years. 
                    This helps identify if the factor performs better in certain months.
                </div>
                """
                plot_list.append(Div(text=desc_t4m, sizing_mode='stretch_width'))

                plot_list.append(
                    Div(text="<div style='font-weight: 600; font-size: 16px; margin: 20px 0 10px;'>2.5 Monthly IC Analysis (Seasonality)</div>",
                        sizing_mode='stretch_width'))
                plot_list += fb.plot('', [('table', t4_monthly)])

                # Plot Monthly IC Mean Bar Chart
                try:
                    if 'mean' in t4_monthly.columns:
                        months = [str(m) for m in t4_monthly.index]
                        means = t4_monthly['mean'].values

                        # Conditional colors
                        colors = ['#ef5350' if x >= 0 else '#66bb6a' for x in means]

                        source = ColumnDataSource(data=dict(months=months, means=means, colors=colors))

                        p_month = figure(x_range=months, height=350, title="Monthly IC Mean (Seasonality)",
                                         tools="pan,wheel_zoom,box_zoom,reset,save",
                                         tooltips=[("Month", "@months"), ("IC Mean", "@means{0.0000}")],
                                         x_axis_label="Month", y_axis_label="IC Mean",
                                         background_fill_color="#fafafa")

                        p_month.vbar(x='months', top='means', width=0.8, color='colors', source=source)

                        p_month.xgrid.grid_line_color = None
                        p_month.y_range.start = min(0, min(means) * 1.2)
                        p_month.y_range.end = max(0, max(means) * 1.2)

                        plot_list.append(p_month)
                except Exception as e:
                    print(f"Error plotting Table 4 Monthly chart: {e}")

            if not last_30_days_ic.empty:
                desc_recent = """
                <div style='background-color: #f8f9fa; border-left: 4px solid #4a90e2; padding: 10px 15px; margin: 20px 0; font-size: 14px; color: #555;'>
                    <strong>Recent 60 Days IC Performance:</strong><br>
                    Shows the daily Cross-Sectional IC (or Rolling IC) for the last 60 trading days.
                </div>
                """
                plot_list.append(Div(text=desc_recent, sizing_mode='stretch_width'))

                plot_list.append(
                    Div(text="<div style='font-weight: 600; font-size: 16px; margin: 20px 0 10px;'>2.6 Recent 60 Days Daily IC (Cross Level)</div>",
                        sizing_mode='stretch_width'))
                try:
                    # prepare data
                    dates = [d.strftime('%Y-%m-%d') for d in last_30_days_ic.index]
                    vals = last_30_days_ic.values

                    p_recent = figure(x_range=dates, height=350, title="Daily Cross-Sectional IC (Last 60 Days)",
                                      tools="pan,wheel_zoom,box_zoom,reset,save",
                                      tooltips=[("Date", "@x"), ("IC", "@y{0.0000}")],
                                      x_axis_label="Date", y_axis_label="IC",
                                      background_fill_color="#fafafa", width=1000)

                    p_recent.line(dates, vals, line_width=2, color="navy", legend_label="Daily IC")
                    p_recent.circle(dates, vals, size=6, color="red", fill_color="white", legend_label="Daily IC")

                    p_recent.xaxis.major_label_orientation = np.pi / 4
                    p_recent.legend.location = "top_left"

                    plot_list.append(p_recent)
                except Exception as e:
                    print(f"Error plotting Last 60 Days IC: {e}")

            if not rolling_ic_comparison.empty:
                desc_rolling = """
                <div style='background-color: #f8f9fa; border-left: 4px solid #4a90e2; padding: 10px 15px; margin: 20px 0; font-size: 14px; color: #555;'>
                    <strong>Rolling IC Analysis (30D vs 60D):</strong><br>
                    Comparison of 30-day and 60-day Rolling Information Coefficient over the full history.
                </div>
                """
                plot_list.append(Div(text=desc_rolling, sizing_mode='stretch_width'))

                plot_list.append(
                    Div(text="<div style='font-weight: 600; font-size: 16px; margin: 20px 0 10px;'>2.7 Rolling IC (30D vs 60D)</div>",
                        sizing_mode='stretch_width'))
                plot_list += fb.plot('Rolling IC (30D vs 60D)', [('line', rolling_ic_comparison)])

            # --- Section 3: Return Analysis ---
            section3_html = """
            <div class='section-header'>
                <div class='section-title'>Part 3: Return & Strategy Analysis</div>
                <div class='section-desc'>Backtest performance of signal-based strategies.</div>
            </div>
            """
            plot_list.append(Div(text=section3_html, sizing_mode='stretch_width'))

            if not t5.empty:
                plot_list.append(
                    Div(text="<div style='font-weight: 600; font-size: 16px; margin: 20px 0 10px;'>3.1 Standardized Strategy Stats (Table 5)</div>",
                        sizing_mode='stretch_width'))
                plot_list += fb.plot('', [('table', t5)])

            if not t5plus.empty:
                plot_list.append(
                    Div(text="<div style='font-weight: 600; font-size: 16px; margin: 20px 0 10px;'>3.2 Standardized Strategy Cumulative Return</div>",
                        sizing_mode='stretch_width'))
                plot_list += fb.plot('Strategy Performance (stdp filter)', [('line', t5plus)])

            # Simple LS Strategy
            desc_ls = """
            <div style='background-color: #f8f9fa; border-left: 4px solid #4a90e2; padding: 10px 15px; margin: 20px 0; font-size: 14px; color: #555;'>
                <strong>Simple Long-Short (Sign Strategy):</strong><br>
                Position = sign(Signal). Return = Position * Label.<br>
                This represents the raw directional power of the signal without standardization or thresholding.
            </div>
            """
            plot_list.append(Div(text=desc_ls, sizing_mode='stretch_width'))

            if not t7.empty:
                plot_list.append(
                    Div(text="<div style='font-weight: 600; font-size: 16px; margin: 20px 0 10px;'>3.3 Simple LS Stats (Table 7)</div>",
                        sizing_mode='stretch_width'))
                plot_list += fb.plot('', [('table', t7)])

            if not t7plus.empty:
                # Add explanation for Simple LS
                ls_desc = """
                <div style='background-color: #f0f7ff; padding: 15px; border-radius: 8px; margin: 20px 0;'>
                    <h4>Simple Long-Short Strategy (Sign Strategy)</h4>
                    <p>This strategy takes a position based purely on the sign of the signal, without scaling by signal strength.
                    It represents the raw directional predictive power.</p>
                    <p><strong>Formula:</strong> $R_{strategy} = \text{sign}(S_t) \times R_{t+1}$</p>
                    <p>Where $S_t$ is the signal at time $t$, and $R_{t+1}$ is the return at $t+1$.</p>
                </div>
                """
                plot_list.append(Div(text=ls_desc, sizing_mode='stretch_width'))

                plot_list.append(
                    Div(text="<div style='font-weight: 600; font-size: 16px; margin: 20px 0 10px;'>3.4 Simple LS Cumulative Return</div>",
                        sizing_mode='stretch_width'))

                # Check if we have 000852 specific columns to plot Strategy vs Benchmark
                # t7plus columns are like: 'gain_whole_market_simple', 'gain_000852.SH_simple', 'bench_000852.SH'

                # 1. Plot All Strategies
                strat_cols = [c for c in t7plus.columns if 'bench' not in c]
                if strat_cols:
                    plot_list += fb.plot('Simple LS Strategy Performance (All Symbols)', [('line', t7plus[strat_cols])])

                # 2. Plot target_symbol Strategy vs Benchmark
                # Find columns related to target_symbol
                targets = [c for c in t7plus.columns if target_symbol in c]
                if targets:
                    strat_col = next((c for c in targets if 'simple' in c), None)
                    bench_col = next((c for c in targets if 'bench' in c), None)

                    if strat_col and bench_col:
                        subset = t7plus[[strat_col, bench_col]].copy()
                        subset.columns = [f'Strategy ({target_symbol})', f'Benchmark ({target_symbol})']

                        plot_list.append(
                        Div(text=f"<div style='font-weight: 600; font-size: 16px; margin: 20px 0 10px;'>3.5 Strategy vs Benchmark ({target_symbol})</div>",
                            sizing_mode='stretch_width'))

                        plot_list += fb.plot(f'Simple LS Strategy vs Benchmark ({target_symbol})', [('line', subset)])

            # Signal Analysis (New Table 9)
            if not t9.empty:
                plot_list.append(
                    Div(text="<div style='font-weight: 600; font-size: 16px; margin: 20px 0 10px;'>3.6 Signal Analysis (Table 9)</div>",
                        sizing_mode='stretch_width'))
                plot_list += fb.plot('', [('table', t9)])

            if not t9plus.empty:
                plot_list.append(
                    Div(text="<div style='font-weight: 600; font-size: 16px; margin: 20px 0 10px;'>3.7 Signal Cumulative Return</div>",
                        sizing_mode='stretch_width'))
                plot_list += fb.plot('Signal Cumulative Return', [('line', t9plus)])

            plot_list.append(Div(text="</div>", sizing_mode='stretch_width'))  # Close report-container

            fb.save_html(save_path, plot_list)
            print(f"Report saved to: {save_path}")
            return
        except Exception as e:
            print(f"Error using io_bokeh: {e}. Falling back to Pandas HTML.")
            import traceback
            traceback.print_exc()

    # Fallback HTML generation
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("<html><head><title>TS Report</title>")
        f.write(
            "<style>table {border-collapse: collapse; width: 100%;} th, td {border: 1px solid black; padding: 8px;} th {background-color: #f2f2f2;}</style>")
        f.write("</head><body>")
        f.write("<h1>Time Series Analysis Report</h1>")

        def write_table(name, df):
            if df.empty: return
            f.write(f"<h2>{name}</h2>")
            f.write(df.to_html())

        write_table('Time Series Signal Statistic (Table 1 - Whole Sample)', t1)
        write_table('TS Correlation Analysis Summary (Table 2)', t2)
        write_table('TS Correlation Analysis By Symbol (Table 3)', t3)
        write_table('TS Yearly Correlation Analysis (Table 4)', t4)
        write_table('TS Monthly Correlation Analysis (Seasonality)', t4_monthly)
        write_table('Extreme Stats (Strategy) (Table 5)', t5)
        write_table('Simple LS Stats (Sign Strategy) (Table 7)', t7)
        # Table 8? Fallback usually doesn't show lines.

        if not t6plus.empty:
            write_table('TS Auto Correlation Mean (Lag 1-10)', t6plus)

        write_table('Signal Auto Correlation Analysis (Table 6)', t6)

        f.write("</body></html>")
    print(f"Fallback report saved to: {save_path}")


# ----------------- Main -----------------

def _calculate_autocorr_for_lag_csv(df, signal_col, lag, corr_func, resample_freq):
    """
    Helper function to calculate auto-correlation for a single lag from CSV data.

    Formula: corr(signal, signal.shift(lag))

    :param df: DataFrame with signal data and datetime index.
    :param signal_col: The name of the signal column.
    :param lag: The lag to use for the shift.
    :param corr_func: The correlation function to use (e.g., tcorr).
    :param resample_freq: The resampling frequency string (e.g., 'MS').
    :return: A Series with the auto-correlation result.
    """
    df_temp = df.copy()
    df_temp['shift'] = df_temp[signal_col].shift(lag)

    def calc_auto(subdf):
        valid = subdf[[signal_col, 'shift']].dropna()
        if len(valid) < 2: return np.nan
        return corr_func(valid[signal_col].values, valid['shift'].values)

    res = df_temp.groupby(pd.Grouper(freq=resample_freq)).apply(calc_auto)
    res.name = f'tautocorr({signal_col},{lag})'
    return res


def calc_tcorr_from_csv(
        signal_csv_path,
        label_csv_path,
        output_dir='output',
        auto_shift_bars=list(range(0, 11)),
        corr_type='pearson',
        resample_freq='MS',
        target_symbol='000852.SH'  #  在此处或调用时指定品种
):
    print(f"Loading signal data from: {signal_csv_path}")
    try:
        df_sig = pd.read_csv(signal_csv_path, encoding='gbk', index_col=0, parse_dates=True)
    except:
        df_sig = pd.read_csv(signal_csv_path, encoding='utf-8', index_col=0, parse_dates=True)

    print(f"Loading label data from: {label_csv_path}")
    try:
        df_lab = pd.read_csv(label_csv_path, encoding='gbk', index_col=0, parse_dates=True)
    except:
        df_lab = pd.read_csv(label_csv_path, encoding='utf-8', index_col=0, parse_dates=True)

    def clean_symbol(col_name):
        return col_name.split('-')[0]

    sig_map = {c: clean_symbol(c) for c in df_sig.columns}
    lab_map = {c: clean_symbol(c) for c in df_lab.columns}
    df_sig.rename(columns=sig_map, inplace=True)
    df_lab.rename(columns=lab_map, inplace=True)

    common_symbols = sorted(list(set(df_sig.columns) & set(df_lab.columns)))

    if not common_symbols:
        print("Error: No common symbols found.")
        return

    print(f"Found {len(common_symbols)} common symbols to process.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mycorr = tcorr if corr_type == 'pearson' else tcos

    all_results = []
    all_results_dict = {}
    raw_data_dict = {}  # Store raw data for strategy backtest

    for symbol in common_symbols:
        try:
            s_sig = df_sig[symbol]
            s_lab = df_lab[symbol]

            df = pd.concat([s_sig, s_lab], axis=1, join='inner')
            df.columns = ['signal', 'label']

            if df.empty:
                print(f"Warning: No overlapping dates for {symbol}, skipping.")
                continue

            # Store raw data for table 5
            raw_data_dict[symbol] = df.copy()

            df_result_list = []

            # Part A: Auto Correlation
            for j in auto_shift_bars:
                res = _calculate_autocorr_for_lag_csv(df, 'signal', j, mycorr, resample_freq)
                df_result_list.append(res)

            # Part B: Stats
            for key in ['signal', 'label']:
                def calc_count(subdf):
                    return tcount(subdf[key].dropna().values)

                res = df.groupby(pd.Grouper(freq=resample_freq)).apply(calc_count)
                res.name = f'tcount({key})'
                df_result_list.append(res)

                def calc_mean(subdf):
                    vals = subdf[key].dropna().values
                    if len(vals) == 0: return np.nan
                    return tmean(vals)

                res = df.groupby(pd.Grouper(freq=resample_freq)).apply(calc_mean)
                res.name = f'tmean({key})'
                df_result_list.append(res)

                def calc_std(subdf):
                    vals = subdf[key].dropna().values
                    if len(vals) < 2: return np.nan
                    return tstd(vals)

                res = df.groupby(pd.Grouper(freq=resample_freq)).apply(calc_std)
                res.name = f'tstd({key})'
                df_result_list.append(res)

            # Part C: IC
            def calc_ic(subdf):
                valid = subdf[['signal', 'label']].dropna()
                if len(valid) < 2: return np.nan
                return mycorr(valid['signal'].values, valid['label'].values)

            res = df.groupby(pd.Grouper(freq=resample_freq)).apply(calc_ic)
            res.name = f'mycorr(signal,label)'
            df_result_list.append(res)

            final_df = pd.concat(df_result_list, axis=1)

            save_path = os.path.join(output_dir, f'{symbol}.csv')
            final_df.to_csv(save_path)

            all_results.append(final_df)
            all_results_dict[symbol] = final_df

        except Exception as e:
            print(f"Exception processing {symbol}: {e}")

    print(f"Processing complete. Results saved to {output_dir}")

    if all_results:
        print("\n--- Generating minbar_table1 and minbar_table2 ---")
        full_df = pd.concat(all_results, axis=0)

        # Table 1: Now using raw data
        tab1 = minbar_table1(raw_data_dict, 'signal', 'label')
        print("\n[minbar_table1]")
        print(tab1)
        tab1.to_csv(os.path.join(output_dir, 'minbar_table1.csv'))

        # Calculate Daily Cross-Sectional IC (Full Sample) if possible
        # (or Rolling IC if only 1 symbol)

        # Check if we should use Cross-Sectional IC (many symbols)
        use_cross_sectional = len(common_symbols) > 1

        daily_ic_df = pd.DataFrame()

        if use_cross_sectional:
            print("Calculating Daily Cross-Sectional IC for Table 2...")
            # Concatenate all raw data
            all_raw_list = []
            for sym, df in raw_data_dict.items():
                temp = df[['signal', 'label']].copy()
                # temp['symbol'] = sym # implicit in structure if needed
                all_raw_list.append(temp)

            if all_raw_list:
                 # Align on index
                combined_raw = pd.concat(all_raw_list, axis=0) # This stacks them. We need to pivot or groupby date

                # Using groupby Date
                def calc_daily_xs_ic(g):
                    if len(g) < 2: return np.nan
                    return g['signal'].corr(g['label'])

                daily_ic_series = combined_raw.groupby(combined_raw.index).apply(calc_daily_xs_ic)
                daily_ic_df = daily_ic_series.to_frame(name='mycorr(signal,label)')
        else:
            # Single symbol case: Use Rolling IC?
            # But "Full Sample" might just mean "Use daily data for everything"?
            # But user specifically complained about IC frequency/count.
            # If it is single symbol, 30-day rolling correlation is a standard proxy for "Time varying IC".
             print("Single symbol detected. Using Rolling IC (20 days) for Table 2 to provide full sample distribution.")
             if common_symbols:
                 sym = common_symbols[0]
                 df = raw_data_dict[sym]
                 # Rolling correlation: 20 days
                 rolling_ic = df['signal'].rolling(window=20).corr(df['label'])
                 daily_ic_df = rolling_ic.to_frame(name='mycorr(signal,label)')

        if not daily_ic_df.empty:
            # Replace full_df usage for Table 2 with this new daily_ic_df
            print("\n--- Generating minbar_table2 (Daily IC Stats) ---")
            tab2, ic_raw_data = minbar_table2(daily_ic_df, 'signal', ['label'])
        else:
            # Fallback to original monthly
            tab2, ic_raw_data = minbar_table2(full_df, 'signal', ['label'])

        print("\n[minbar_table2]")
        print(tab2)
        tab2.to_csv(os.path.join(output_dir, 'minbar_table2.csv'))

        # --- Calculate Rolling IC (30D & 60D) Comparison ---
        print("\n--- Calculating Rolling IC (30D & 60D) ---")
        rolling_ic_comparison = pd.DataFrame()
        try:
            if use_cross_sectional:
                # Use daily_ic_series calculated earlier
                if 'daily_ic_series' in locals() and not daily_ic_series.empty:
                    ric30 = daily_ic_series.rolling(window=30).mean()
                    ric60 = daily_ic_series.rolling(window=60).mean()
                    rolling_ic_comparison = pd.concat([ric30, ric60], axis=1)
                    rolling_ic_comparison.columns = ['Rolling IC 30D', 'Rolling IC 60D']
            else:
                 # Single Symbol: Rolling Correlation
                 if common_symbols:
                     sym = common_symbols[0]
                     df = raw_data_dict[sym]
                     ric30 = df['signal'].rolling(window=30).corr(df['label'])
                     ric60 = df['signal'].rolling(window=60).corr(df['label'])
                     rolling_ic_comparison = pd.concat([ric30, ric60], axis=1)
                     rolling_ic_comparison.columns = ['Rolling IC 30D', 'Rolling IC 60D']

            rolling_ic_comparison = rolling_ic_comparison.dropna()
        except Exception as e:
            print(f"Error calculating Rolling IC Comparison: {e}")


        # Table 3
        try:
            print("\n--- Generating minbar_table3 ---")
            tab3, tab3plus = minbar_table3(all_results_dict, 'signal', 'label')
            print("\n[minbar_table3]")
            print(tab3)
            tab3.to_csv(os.path.join(output_dir, 'minbar_table3.csv'))

            # Save tab3plus (accumulated returns) for plotting
            tab3plus.to_csv(os.path.join(output_dir, 'minbar_table3_plus.csv'))
        except Exception as e:
            print(f"CRITICAL ERROR generating minbar_table3: {e}")
            import traceback
            traceback.print_exc()

        # Table 4
        try:
            print("\n--- Generating minbar_table4 (Annual Stats) ---")
            tab4 = minbar_table4(full_df, 'signal', 'label')
            print("\n[minbar_table4]")
            print(tab4)
            tab4.to_csv(os.path.join(output_dir, 'minbar_table4.csv'))
        except Exception as e:
            print(f"Error table4: {e}")
            import traceback
            traceback.print_exc()

        # Table 4 Monthly (Seasonality)
        try:
            print("\n--- Generating minbar_table4_monthly (Monthly Stats) ---")
            t4_monthly = minbar_table4_monthly(full_df, 'signal', 'label')
            print("\n[minbar_table4_monthly]")
            print(t4_monthly)
            t4_monthly.to_csv(os.path.join(output_dir, 'minbar_table4_monthly.csv'))
        except Exception as e:
            print(f"Error table4_monthly: {e}")
            import traceback
            traceback.print_exc()
            t4_monthly = pd.DataFrame()

        # Table 5
        try:
            print("\n--- Generating minbar_table5 (Strategy Backtest) ---")
            # Use raw_data_dict instead of all_results_dict
            tab5, tab5plus = minbar_table5_csv(raw_data_dict, 'signal', 'label', stdp_list=[1, 2], target_symbol=target_symbol)
            print("\n[minbar_table5]")
            print(tab5)
            tab5.to_csv(os.path.join(output_dir, 'minbar_table5.csv'))

            # Save tab5plus (equity curve) for plotting
            tab5plus.to_csv(os.path.join(output_dir, 'minbar_table5_plus.csv'))
        except Exception as e:
            print(f"Error table5: {e}")
            import traceback
            traceback.print_exc()

        # Table 7 (Simple Long-Short)
        try:
            print("\n--- Generating minbar_table7 (Simple Long-Short) ---")
            tab7, tab7plus = minbar_table7_csv(raw_data_dict, 'signal', 'label', target_symbol=target_symbol)
            print("\n[minbar_table7]")
            print(tab7)
            tab7.to_csv(os.path.join(output_dir, 'minbar_table7.csv'))
            tab7plus.to_csv(os.path.join(output_dir, 'minbar_table7_plus.csv'))
        except Exception as e:
            print(f"Error table7: {e}")
            import traceback
            traceback.print_exc()
            tab7 = pd.DataFrame()
            tab7plus = pd.DataFrame()

        # Table 6
        try:
            print("\n--- Generating minbar_table6 (Autocorr) ---")
            tab6, tab6plus = minbar_table6(full_df, 'signal', auto_shift_bars)
            print("\n[minbar_table6]")
            print(tab6)
            tab6.to_csv(os.path.join(output_dir, 'minbar_table6.csv'))
            tab6plus.to_csv(os.path.join(output_dir, 'minbar_table6_plus.csv'))
        except Exception as e:
            print(f"Error table6: {e}")
            import traceback
            traceback.print_exc()  # Added traceback
            tab6 = pd.DataFrame()
            tab6plus = pd.DataFrame()  # Ensure variable exists

        # Table 9 (Signal Analysis)
        try:
            print("\n--- Generating minbar_table9 (Signal Analysis) ---")
            tab9, tab9plus = minbar_table9_csv(raw_data_dict, 'signal', target_symbol=target_symbol)
            print("\n[minbar_table9]")
            print(tab9)
            tab9.to_csv(os.path.join(output_dir, 'minbar_table9.csv'))
            tab9plus.to_csv(os.path.join(output_dir, 'minbar_table9_plus.csv'))
        except Exception as e:
            print(f"Error table9: {e}")
            tab9, tab9plus = pd.DataFrame(), pd.DataFrame()

        # Daily IC (Last 60 days)
        last_30_ic = pd.Series(dtype=float)
        try:
            print("\n--- Calculating Last 60 Days Cross-Sectional IC ---")
            if not daily_ic_df.empty:
                last_30_ic = daily_ic_df['mycorr(signal,label)'].tail(60)
            else:
                last_30_ic = calc_daily_cross_sectional_ic(raw_data_dict, lookback=60)
            print(last_30_ic)
        except Exception as e:
            print(f"Error calculating daily IC: {e}")

        # HTML Report
        try:
            print("\n--- Generating HTML Report ---")
            generate_html_report_csv(output_dir, "tcorr_report",
                                     tab1, tab2,
                                     tab3 if 'tab3' in locals() else pd.DataFrame(),
                                     tab3plus if 'tab3plus' in locals() else pd.DataFrame(),
                                     tab4 if 'tab4' in locals() else pd.DataFrame(),
                                     tab5 if 'tab5' in locals() else pd.DataFrame(),
                                     tab5plus if 'tab5plus' in locals() else pd.DataFrame(),
                                     tab6 if 'tab6' in locals() else pd.DataFrame(),
                                     tab7 if 'tab7' in locals() else pd.DataFrame(),
                                     tab7plus if 'tab7plus' in locals() else pd.DataFrame(),
                                     tab9 if 'tab9' in locals() else pd.DataFrame(),
                                     tab9plus if 'tab9plus' in locals() else pd.DataFrame(),
                                     tab6plus if 'tab6plus' in locals() else pd.DataFrame(),
                                     ic_raw_data if 'ic_raw_data' in locals() else pd.DataFrame(),
                                     t4_monthly if 't4_monthly' in locals() else pd.DataFrame(),
                                     last_30_days_ic=last_30_ic,
                                     rolling_ic_comparison=rolling_ic_comparison if 'rolling_ic_comparison' in locals() else pd.DataFrame(),
                                     target_symbol=target_symbol)
        except Exception as e:
            print(f"Error generating HTML report: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    signal_file = os.path.join(r'C:\Users\yylc0\Desktop\arbeite\TCN_project\project\outputs\inference_predictions.csv')
    label_file = os.path.join(r'C:\Users\yylc0\Desktop\arbeite\TCN_project\project\outputs\inference_actual_returns.csv')

    #  在这里修改 target_symbol 来选择你想要的品种
    calc_tcorr_from_csv(signal_file, label_file, target_symbol='000852.SH')
