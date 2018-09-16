





def bleed():
    sheets = []
    no_sheets = raw_input()
    for _ in range(no_sheets):
        temp = []
        k = raw_input()
        r,c = k.split(",")
        dots = raw_input()
        for _ in range(dots):
            k1 = raw_input()
            i, j, d = k1.aplit(",")
            temp.append([i,j,d])
        sheets.append(temp[:])
    print sheets
    exit()
    res = []
    for sheet in sheets:
        matrix = [[0]*cols for _ in range(rows)]
        for rows_, cols_, dark in sheet:
            doBfs(rows_, cols_, matrix, dark)
            res.append(sum(map(sum,matrix)))


def doBfs(i,j,matrix,dark):

    qu = [(i,j)]
    visited = [(i,j)]
    count = 1
    while qu:
        qu_len = len(qu)
        for _ in range(qu_len):
            a,b = qu.pop(0)
            for p,q in ((a+1,b),(a-1,b),(a,b+1),(a,b-1)):
                if(p>=0 and q >=0 and p<rows and q<cols and (p,q) not in visited):
                    matrix[p][q] = max(matrix[p][q],dark-count)
                    visited.append((p,q))
            if (dark-count <=0): return
            count +=1


bleed()
