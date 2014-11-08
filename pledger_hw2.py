#!/usr/bin/python3

import argparse,itertools,collections,math
from multiprocessing import Pool


class HMMDict(collections.OrderedDict):

    def __init__(self,*args,**kwargs):
        collections.OrderedDict.__init__(self,*args)
        self.default = kwargs['default'] if 'default' in kwargs else (lambda: None)

    def __getitem__(self,key):
        if key not in self.keys():
            self[key]=self.default()
        return collections.OrderedDict.__getitem__(self,key)

    def normalized(self, S,func=(lambda x:x)):

        d = HMMDict(default=self.default)
        c = sum(self[b] for b in S)

        for b in S:
            d[b] = func(self[b]/c)

        return d

class HMMVector(HMMDict):
    def __init__(self,*args,**kwargs):
        self.d = d = 0 if 'default' not in kwargs else kwargs['default']
        kwargs['default'] = (lambda: d)
        self.A = None if 'A' not in kwargs else kwargs['A']

        HMMDict.__init__(self,*args,**kwargs)

    def normalized(self, func=(lambda x:x)):
        d = HMMVector(A=self.A,default=self.default)
        c = sum(self[b] for b in self.A)

        for b in self.A:
            d[b] = func(self[b]/c)

        return d

class HMMMatrix(HMMDict):

    def __init__(self,*args,**kwargs):

        self.d = d = 0 if 'default' not in kwargs else kwargs['default']

        self.A = None if 'A' not in kwargs else kwargs['A']
        self.B = None if 'B' not in kwargs else kwargs['B']

        kwargs['default'] = (lambda: HMMVector(default=d,A=self.B))

        HMMDict.__init__(self,*args,**kwargs)

    def __repr__(self):

        if self.A is not None and self.B is not None:
            A,B = (self.A,self.B)
            s=","+",".join(repr(s) for s in B)

            for a in A:
                s+= "\n"+repr(a)
                for b in B:
                    s+=","+str(self[a][b])
            return "HMMMatrix([\n"+s+"\n])"
        else:
            return HMMDict.__repr__(self)

    def normalized(self,func=(lambda x:x)):
        m = HMMMatrix(A=self.A,B=self.B,default=self.d)
        for a in self.A:
            m[a] = self[a].normalized(func)
        return m




class HMM(object):

    def __init__(self, **kwargs):

        self.order = 1 if 'order' not in kwargs else kwargs['order']

        self.S = list([] if 'S' not in kwargs else kwargs['S']) # Internal states
        self.V = list([] if 'V' not in kwargs else kwargs['V']) # External states

        self.PiS = []

        self.Sn = self.S
        for i in range(1,self.order+1):
            self.Sn = ["".join(t) for t in itertools.product(self.S, repeat=i)]
            self.PiS += self.Sn

        # If N is the order, {X_t}\subset S is the internal state sequence
        # and {Y_t}\subset V is the external state sequence then...
        # A[i][j] = P(X_t=i|<X_t-N,...,X_t-1> =j)
        # B[i][j] = P(Y_t=i|X_t =j)
        # Pi[i]   = P(X_0=i)
        self.A = None
        self.B = None
        self.AT = None
        self.BT = None
        self.Pi = None

        self.Acounts = HMMMatrix(A=self.S, B=self.PiS, default=1)
        self.Bcounts = HMMMatrix(A=self.S, B=self.V, default=1)
        self.ATcounts = HMMMatrix(A=self.S, B=self.PiS, default=1)
        self.BTcounts = HMMMatrix(A=self.V, B=self.S, default=1)
        self.PiCounts = HMMVector(A=self.PiS, default=1)

    def train(self, Q, O):
        """ 
        Takes sequences Q=(q1,q2,...,qN) and O=(v1,v2,...,vN) 
        and appends appropriate counts to Acounts, Bcounts, and 
        PiCounts.
        """
        if len(Q) is not len(O):

            raise Exception('Invalid training data!')

        # print("Q: %r \n O: %r "%(Q,O))

        prev = None
        for i in range(len(Q)):
            # i in [0,len(inp)).
            
            i0 = max(0,i-self.order+1)

            qi,vi = pair = (Q[i],O[i])

            if i==0:
                # We need to add to initial distribution
                self.PiCounts[qi[-1]] += 1
            else:
                self.Acounts[qi[-1]][prev[0]] += 1
                self.ATcounts[prev[0]][qi[-1]] += 1

            self.Bcounts[vi[-1]][qi[-1]] += 1
            self.BTcounts[qi[-1]][vi[-1]] += 1

            
            
            
            
            prev = pair
            
    
    def normalize(self):
        self.A = self.Acounts.normalized(math.log)
        self.B = self.Bcounts.normalized(math.log)
        self.AT = self.ATcounts.normalized(math.log)
        self.BT = self.BTcounts.normalized(math.log)
        self.Pi = self.PiCounts.normalized(math.log)


    def dump(self):
        print('S   = %r'%self.S)
        print('Sn  = %r'%self.Sn)
        print('V   = %r'%self.V)
        print('A_N = %r'%(self.Acounts).keys())
        print('B_N = %r'%(self.Bcounts).keys())
        print('Pi_N= %r'%(self.PiCounts))

class Viterbi(object):

    def __init__(self,hmm):
        self.hmm=hmm

    def correct(self,Y):
        """
        Y is a **word** to be corrected.
        """

        self.hmm.normalize()

        hmm = self.hmm

        V = HMMMatrix(A=range(len(Y)), B=hmm.S)

        path = HMMDict()
        ptr = HMMMatrix(A=hmm.S, B=range(len(Y)))

        for k in hmm.S:
            V[0][k] = hmm.BT[k][Y[0]] + hmm.Pi[k]
            path[k] = [k]
            ptr[k][0] = k

        for t in range(1,len(Y)):
            newpath = HMMDict()
            yt = Y[t]

            for k in hmm.S:
                (p,sk) = max((hmm.BT[k][yt] + V[t-1][x] + hmm.AT[x][k], x) for x in hmm.S)
                V[t][k] =  p
                newpath[k] = path[sk] + [k]
                ptr[k][t] = sk
            
            path = newpath

        n = max(hmm.order,len(Y))-1

        self.V = V

        (p,sk) = max((V[n][s],s) for s in hmm.S)
        return (p,"".join(path[sk]))


def main():
    parser = argparse.ArgumentParser(
        description='Match two or more strings or lists as best as possible.')
    parser.add_argument('infile', type=argparse.FileType('r'),
                        help='file to read data from')
    parser.add_argument('--debug','-d',action='store_true', help='Output debug information')
    parser.add_argument('--order','-o',type=int, default=1, help='Order of hmm to generate')
    parser.add_argument('--count','-c',type=int,default=-1, help='Number of items to correct')
    parser.add_argument('--test','-t',action='store_true', help='Test with training data instead of test data')
    parser.add_argument('word',nargs='*',help='Words to correct')

    args = parser.parse_args()

    print("Transposing the fucking data format...", end="", flush=True)

    space = "abcdefghijklmnopqrstuvwxyz"[0:26:1]

    hmm = HMM( S=space, V=space, order=args.order)

    inputs = ["", ""]
    outputs = ["", ""]

    
    mode = 0
    for line in args.infile:
        if '.' in line:
            mode = 1
        else:
            inputs[mode] += line[2]
            outputs[mode] += line[0]

    inputs[0]  = inputs[0].split('_')
    outputs[0] = outputs[0].split('_')

    if args.test:
        inputs[1]  = inputs[0]
        outputs[1] = outputs[0]
    else:
        inputs[1]  = inputs[1].split('_')
        outputs[1] = outputs[1].split('_')
    print("DONE")

    print("Training the fucking HMM..............", end="", flush=True)
    for i in range(len(inputs[0])):
        hmm.train(outputs[0][i], inputs[0][i])
    print("DONE")

    if args.debug:
        hmm.dump()


    v = Viterbi(hmm)
    if len(args.word)==0:
        counts = {True:0,False:0}
        N = args.count if args.count != -1 else len(inputs[1])
        W = []
        print("Correcting %s fucking terms..........."%(N), end="", flush=True)
        
        for i in range(N):

            (p,w) = v.correct(inputs[1][i])
            W += [w]
            if args.debug or p==0:
                print("\n\n")
                print(v.V)
                print("%r => %s,%r | %r"%(inputs[1][i],p,w,outputs[1][i]))
            for j in range(len(w)):
                counts[outputs[1][i][j]==w[j]] += 1

        print("DONE")
        print(" ".join(W))
        print("%r => Error rate: %s"%(counts,counts[False]/(counts[False]+counts[True])))
    else:
        for w in args.word:
            (p,c) = v.correct(w)
            print("%r => %s:%r"%(w,p,c))



if __name__ == "__main__":

    main()
