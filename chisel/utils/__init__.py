def utf8_to_int(utext):
    return ord(utext)

def int_to_utf8(n):
    over10 = "abcdef"
    return eval("u'\\u" + "".join(map(lambda m:str(m) if m<10 else over10[m-10],[(n / (16**3) )% 16, (n / (16**2) )% 16, (n / 16 )% 16, n % 16])) + "'")
    