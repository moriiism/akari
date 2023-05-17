def getColNameLst():
    # get the following column name list
    # ['x00y00', 'x00y01', ..., 'x00y04', 
    # 'x01y00', 'x01y01', ..., 'x01y04',
    # ...
    # 'x04y00', 'x04y01', ..., 'x04y04']

    colname_lst = []
    npixx = 5
    npixy = 5
    for ipixx in range(npixx):
        for ipixy in range(npixy):
            colname_str = f"x{ipixx:02d}y{ipixy:02d}"
            colname_lst.append(colname_str)

    return colname_lst
