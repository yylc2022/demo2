import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import re
import gc
import os

def str_to_pa_type(dtype_str):
    dtype_map = {
        'int8': pa.int8(),
        'int16': pa.int16(),
        'int32': pa.int32(),
        'int64': pa.int64(),
        'uint8': pa.uint8(),
        'uint16': pa.uint16(),
        'uint32': pa.uint32(),
        'uint64': pa.uint64(),
        'float16': pa.float16(),
        'float32': pa.float32(),
        'float64': pa.float64(),
        'bool': pa.bool_(),
    }
    # 处理定长字节串 S10/S20
    m = re.search(r'S(\d+)', dtype_str)
    if m:
        #兼容新老的pyarrow版本
        if hasattr(pa,"fixed_size_binary"):
            return pa.fixed_size_binary(int(m.group(1)))

        return pa.binary(int(m.group(1)))
    # 处理定长unicode字符串 U10/U20
    m = re.search(r'U(\d+)', dtype_str)
    if m:
        return pa.string()  # pyarrow没有定长unicode，可以用pa.string()代替
    # 处理基础类型
    return dtype_map.get(dtype_str, pa.string())

# 示例
# dtypes = ['uint16', 'float64', 'S10', 'U20', 'bool', 'foo', 'string']

# for dtype in dtypes:
#     print(f"{dtype} -> {str_to_pa_type(dtype)}")

# table format



def save(
    df,
    path,
    typed=[],
    compression: str = "zstd",
    compression_level: int = 6,
    row_group_size: int = 5_000_000,
):
    """
    Save pandas DataFrame to parquet.

    - compression: 'zstd' (default), 'brotli', 'gzip', 'snappy', None
    - compression_level: codec level (if pyarrow supports it); e.g. zstd 1~22
    - row_group_size: rows per row group (default 5,000,000)
    """
    # 1) schema
    types = []
    for key in df.columns:
        types.append((key, str_to_pa_type(str(df[key].dtype))))
    schema = pa.schema(types)

    # 2) pandas -> arrow table
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)

    # 3) write
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        write_kwargs = dict(
            row_group_size=row_group_size,
            compression=compression,
        )
        # 兼容不同 pyarrow 版本：新版本支持 compression_level
        try:
            pq.write_table(table, f, compression_level=compression_level, **write_kwargs)
        except TypeError:
            pq.write_table(table, f, **write_kwargs)

    # 4) cleanup
    del table, df
    gc.collect()
