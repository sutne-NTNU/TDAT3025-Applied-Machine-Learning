# takes a two-dimensional array as input and lets you set spacing, wether to flip rows/colums
# and if the array contains headers for either the rows or columns (or both)
def printArray(array, spacing=5, flip=False, columnHeaders=None, rowHeaders=None):
    printFormat = "%-" + str(spacing) + "s"
    # flip array if necesary
    if flip:
        array = flipArray(array)

    # Print header
    if columnHeaders:
        start = " "
        if len(columnHeaders) > len(array[0]):
            start = columnHeaders[0]
            del (columnHeaders[0])
        elif len(rowHeaders) > len(array):
            start = rowHeaders[0]
            del (rowHeaders[0])
        if rowHeaders:
            header = (printFormat + "| ") % start
        else:
            header = start
        for item in columnHeaders:
            header += printFormat % item
        underlinedHeader = "\u0332".join(str(header.format()))
        print(underlinedHeader)

    # Print array itself
    for (i, row) in enumerate(array):
        # Check to prepend row header
        if rowHeaders:
            strRow = (printFormat + "| ") % str(rowHeaders[i])
        else:
            strRow = ""

        for item in row:
            strRow += printFormat % item
        print(strRow)


# Return the same array but the rows and columns are switched
def flipArray(array):
    flipped = []
    for i in range(0, len(array[0])):
        row = []
        for j in range(0, len(array)):
            row.append(array[j][i])
        flipped.append(row)
    return flipped
