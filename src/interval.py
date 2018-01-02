class Interval:
    def __init__(self,a,b):
        self.start=a
        self.stop=b
        self.is_trivial=(a==b)

def interval(t):
    def mkinterval(str):
        try:
            val=t(str)
            return Interval(val,val)
        except ValueError:
            str1,str2=str.split(':')
            val1=t(str1)
            val2=t(str2)
            return Interval(val1,val2)
    return mkinterval

