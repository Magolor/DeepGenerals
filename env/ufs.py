class UnionFindSet(object):
    def __init__(self, objs):
        self.fa = {obj:obj for obj in objs}

    def __len__(self):
        return len(self.fa)

    def getfa(self, x):
        if self.fa[x]!=x:
            self.fa[x] = self.getfa(self.fa[x])
        return self.fa[x]

    def union(self, x, y):
        x = self.getfa(x); y = self.getfa(y)
        if x!=y:
            self.fa[x] = y; return True
        else:
            return False

    def connected(self, x,y):
        x = self.getfa(x); y = self.getfa(y); return x==y